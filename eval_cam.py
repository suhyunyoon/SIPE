import numpy as np
import os
import argparse
from PIL import Image
from tool import pyutils
from data import data_voc, data_coco
from tqdm import tqdm
from torch.utils.data import DataLoader
from chainercv.evaluations import calc_semantic_segmentation_confusion

def calc_miou(confusion):
    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    iou = gtjresj / denominator
    print({'iou': iou, 'miou': np.nanmean(iou)})

def run(args, predict_dir, num_cls, dataset, dataset_ulb=None):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    if dataset_ulb is not None:
        dataloader_ulb = DataLoader(dataset_ulb, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    
    def get_preds_masks(dataloader):
        preds = []
        masks = []
        for iter, pack in tqdm(enumerate(dataloader)):
            cam_dict = np.load(os.path.join(predict_dir, pack['name'][0] + '.npy'), allow_pickle=True).item()
            cams = cam_dict['IS_CAM']
            keys = cam_dict['keys']
            cls_labels = np.argmax(cams, axis=0)
            cls_labels = keys[cls_labels]
            preds.append(cls_labels.copy())

            mask = np.array(Image.open(os.path.join(args.gt_path,  pack['name'][0] + '.png')))
            masks.append(mask.copy())
        return preds, masks
    
    preds, masks = get_preds_masks(dataloader)
    if dataset_ulb is not None:
        preds_ulb, masks_ulb = get_preds_masks(dataloader_ulb)
    else:
        preds_ulb, masks_ulb = [], []
    print(len(preds) + len(preds_ulb), 'Images:')

    if len(preds_ulb) > 0:
        print(len(preds), 'Labeled Images:')
    confusion = calc_semantic_segmentation_confusion(preds, masks)[:num_cls, :num_cls]
    calc_miou(confusion)

    if len(preds_ulb) > 0:
        print(len(preds_ulb), 'Unlabeled Images:')
        confusion_ulb = calc_semantic_segmentation_confusion(preds_ulb, masks_ulb)[:num_cls, :num_cls]
        calc_miou(confusion_ulb)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="voc", type=str)
    parser.add_argument("--dataset_root", default="../PascalVOC2012/VOCdevkit/VOC2012", type=str)
    parser.add_argument("--gt_path", default='../PascalVOC2012/VOCdevkit/VOC2012/SegmentationClass', type=str)
    parser.add_argument('--session_name', default="exp", type=str)
    parser.add_argument("--lb_list", default=None, type=str) ###
    parser.add_argument("--ulb_list", default=None, type=str) ###
    args = parser.parse_args()

    assert args.dataset in ['voc', 'coco'], 'Dataset must be voc or coco in this project.'

    if args.dataset == 'voc':
        dataset_root = args.dataset_root #'../PascalVOC2012/VOCdevkit/VOC2012'
        num_cls = 21
        train_list = data_voc.load_img_name_list('data/train_voc.txt')

        if args.lb_list is None:
            dataset = data_voc.VOC12ImageDataset('data/train_' + args.dataset + '.txt', voc12_root=dataset_root, img_normal=None, to_torch=False)
        else:
            dataset = data_voc.VOC12ImageDataset(args.lb_list, voc12_root=dataset_root, img_normal=None, to_torch=False)
            dataset.img_name_list = list(set(dataset.img_name_list) & set(train_list))
        if args.ulb_list is not None:
            dataset_ulb = data_voc.VOC12ImageDataset(args.ulb_list, voc12_root=dataset_root, img_normal=None, to_torch=False)
            dataset_ulb.img_name_list = list(set(dataset_ulb.img_name_list) & set(train_list))
        else:
            dataset_ulb = None

    elif args.dataset == 'coco':
        # args.gt_path = "../ms_coco_14&15/SegmentationClass/train2014/"
        dataset_root = args.dataset_root #"../ms_coco_14&15/images"
        train_list = 'data/train_coco.txt'
        num_cls = 81
        train_list = data_coco.load_img_name_list('data/train_coco.txt')
        
        if args.lb_list is None:
            dataset = data_coco.COCOImageDataset('data/train_' + args.dataset + '.txt', coco_root=dataset_root, img_normal=None, to_torch=False)
        else:
            dataset = data_coco.COCOImageDataset(args.lb_list, coco_root=dataset_root, img_normal=None, to_torch=False)
            dataset.img_name_list = list(set(dataset.img_name_list) & set(train_list))
        if args.ulb_list is not None:
            dataset_ulb = data_coco.COCOImageDataset(args.ulb_list, coco_root=dataset_root, img_normal=None, to_torch=False)
            dataset_ulb.img_name_list = list(set(dataset_ulb.img_name_list) & set(train_list))
        else:
            dataset_ulb = None

    pyutils.Logger(os.path.join(args.session_name, 'eval.log'))
    run(args, args.session_name + "/npy", num_cls, dataset, dataset_ulb)