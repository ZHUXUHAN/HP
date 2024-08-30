MODEL_NAME='hp_model/'
CHCEKPOINTS_PATH='your_model_path'
DATASETS='VG'
NUM_GPUS=1
CUDA_VISIBLE_DEVICES=0
PREDICTOR='HP_Predictor'

if [ $DATASETS = "VG" ]; then
    USE_BIAS=True
    CONFIG_FILE="configs/e2e_relation_X_101_32_8_FPN_1x.yaml"
    PRETRAINED_DETECTOR_CKPT='pretrained_faster_rcnn/model_final.pth'
fi

python -m torch.distributed.launch --master_port 10022 --nproc_per_node=${NUM_GPUS} tools/relation_test_net.py \
--config-file ${CONFIG_FILE} \
MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
MODEL.ROI_RELATION_HEAD.PREDICTOR ${PREDICTOR} \
TEST.IMS_PER_BATCH ${NUM_GPUS} \
GLOVE_DIR ./datasets/vg/glove \
MODEL.PRETRAINED_DETECTOR_CKPT ${PRETRAINED_DETECTOR_CKPT} \
OUTPUT_DIR "$CHCEKPOINTS_PATH"/${MODEL_NAME}/ \
SOLVER.PRE_VAL False \
TEST.IMS_PER_BATCH 1 \
MODEL.ROI_RELATION_HEAD.HP.A1 0.2 \
MODEL.ROI_RELATION_HEAD.HP.A2 0.1 \
MODEL.ROI_RELATION_HEAD.REL_NMS False

