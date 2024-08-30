import os
import numpy as np
import torch
from maskrcnn_benchmark.modeling import registry
from torch import nn
from torch.nn import functional as F
import random

from maskrcnn_benchmark.modeling.utils import cat
from .utils_relation import layer_init, get_box_info, get_box_pair_info
from maskrcnn_benchmark.data import get_dataset_statistics

from .model_msg_passing import IMPContext
from .model_motifs import LSTMContext, FrequencyBias
from .model_vctree import VCTreeLSTMContext

from maskrcnn_benchmark.modeling.roi_heads.relation_head.clip import clip
from maskrcnn_benchmark.modeling.roi_heads.relation_head.clip.simple_tokenizer import SimpleTokenizer as clip_tokenizer
from maskrcnn_benchmark.modeling.roi_heads.relation_head.clip.text_encoder import TextEncoder
from maskrcnn_benchmark.modeling.roi_heads.relation_head.clip.prompt import Clip_PromptLearner


@registry.ROI_RELATION_PREDICTOR.register("HP_Predictor")
class HP_Predictor(nn.Module):
    def __init__(self, config, in_channels):
        super(HP_Predictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        self.out_dir = config.OUTPUT_DIR

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
            'att_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        # init contextual lstm encoding
        self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

        self.rel_classes = rel_classes
        self.obj_classes = obj_classes

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_text = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_image = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        with torch.no_grad():
            clip_model, _ = clip.load('RN101', device='cpu')

        self.text_encoder = TextEncoder(clip_model)

        self.prompt_learner = Clip_PromptLearner(config, rel_classes, clip_tokenizer(), clip_model)
        self.prompt_learner_kl = Clip_PromptLearner(config, rel_classes, clip_tokenizer(), clip_model)
        self.prompt_learner_harder = Clip_PromptLearner(config, rel_classes, clip_tokenizer(), clip_model)

        relations = torch.arange(len(rel_classes))
        objs = torch.arange(len(obj_classes))
        self.obj_text = clip.tokenize([obj_classes[i.data] for i in objs])
        self.relation_text = clip.tokenize([rel_classes[i.data] for i in relations])
        self.down_dim = nn.Linear(self.pooling_dim, 512)  # R50: 1024 R101:512
        layer_init(self.down_dim, xavier=True)
        self.a1 = config.MODEL.ROI_RELATION_HEAD.HP.A1
        self.a2 = config.MODEL.ROI_RELATION_HEAD.HP.A2
        self.frozen()

    def frozen(self):
        for name, param in self.text_encoder.named_parameters():
            param.requires_grad_(False)

    def get_rel_dist_ctp(self, pair_pred, prod_rep, prompt_learner, down_dim, logit_scale):
        tokenized_prompts = self.relation_text.to(prod_rep.device)
        text_features = self.text_encoder(prompt_learner, tokenized_prompts)
        image_features = down_dim(prod_rep)
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        rel_dists = logit_scale.exp() * image_features @ text_features.T
        return rel_dists

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        add_losses = {}

        # encode context infomation
        obj_dists, obj_preds, edge_ctx, _ = self.context_layer(roi_features, proposals, rel_pair_idxs, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        if self.use_vision:
            if self.union_single_not_match:
                prod_rep = prod_rep * self.up_dim(union_features)
            else:
                prod_rep = prod_rep * union_features

        rel_dists = self.get_rel_dist_ctp(pair_pred, prod_rep,
                                          self.prompt_learner(),
                                          self.down_dim,
                                          self.logit_scale)

        if self.a1 != 0 or self.a2 != 0 and not self.training:
            rel_dists_hard = self.get_rel_dist_ctp(pair_pred, prod_rep,
                                                   self.prompt_learner_kl(),
                                                   self.down_dim,
                                                   self.logit_scale_image)
            rel_dists_harder = self.get_rel_dist_ctp(pair_pred, prod_rep,
                                                     self.prompt_learner_harder(),
                                                     self.down_dim,
                                                     self.logit_scale_image)
            rel_dists[:, 1:] = rel_dists[:, 1:] + self.a1 * rel_dists_hard[:, 1:] + self.a2 * rel_dists_harder[:, 1:]

        rel_dists_f = self.freq_bias.index_with_labels(pair_pred.long())
        rel_dists = rel_dists + rel_dists_f

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
            return (obj_dists, att_dists), rel_dists, add_losses
        else:
            return obj_dists, rel_dists, add_losses