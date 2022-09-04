DATASET_ROOT=../../dataset/VOC/VOCdevkit/VOC2012
SESSION="logs/sipe"

# Step 1. Train SIPE for localization maps.

# 1.1 train sipe
python train_resnet50_SIPE.py --session_name ${SESSION} --dataset_root ${DATASET_ROOT}
date
# 1.2 obtain localization maps
python make_cam.py --session_name ${SESSION} --dataset_root ${DATASET_ROOT}
date
# 1.3 evaluate localization maps
python eval_cam.py --session_name ${SESSION} --dataset_root ${DATASET_ROOT} --gt_path ${DATASET_ROOT}/SegmentationClass
date

# Step 2. Train IRN for pseudo labels.

# 2.1 generate ir label
python cam2ir.py --session_name ${SESSION} --dataset_root ${DATASET_ROOT}
date
# 2.2 train irn
python train_irn.py --session_name ${SESSION} --voc12_root ${DATASET_ROOT}
date
# 2.3 make pseudo labels
python make_seg_labels.py --session_name ${SESSION} --voc12_root ${DATASET_ROOT}
date

# 2.4 eval
python eval_sem_seg.py --sem_seg_out_dir ${SESSION}/pseudo_label --voc12_root ${DATASET_ROOT} --gt_dir ${DATASET_ROOT}/SegmentationClass
date