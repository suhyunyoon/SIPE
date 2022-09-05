from builtins import bool
import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import importlib
import os
import imageio
import argparse
from data import data_voc, data_coco
from tool import torchutils, pyutils

cudnn.enabled = True

def overlap(img, hm):
    hm = plt.cm.jet(hm)[:, :, :3]
    hm = np.array(Image.fromarray((hm*255).astype(np.uint8), 'RGB').resize((img.shape[1], img.shape[0]), Image.BICUBIC)).astype(np.float)*2
    if hm.shape == np.array(img).astype(np.float).shape:
        out = (hm + np.array(img).astype(np.float)) / 3
        out = (out / np.max(out) * 255).astype(np.uint8)
    else:
        print(hm.shape)
        print(np.array(img).shape)
    return out

def draw_heatmap(norm_cam, gt_label, orig_img, save_path, img_name):
    gt_cat = np.where(gt_label==1)[0]
    for _, gt in enumerate(gt_cat):
        heatmap = overlap(orig_img, norm_cam[gt])
        cam_viz_path = os.path.join(save_path, img_name + '_{}.png'.format(gt))
        imageio.imsave(cam_viz_path, heatmap)


def _work(process_id, model, dataset, dataset_ulb, args):
    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    # Batch size = 1
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)
    
    if dataset_ulb is not None: ###
        databin_ulb = dataset_ulb[process_id]
        data_loader_ulb = DataLoader(databin_ulb, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):

        model.cuda()

        def save_cams(databin, data_loader, unlabeled=False):
            for iter, pack in enumerate(data_loader):
                img_name = pack['name'][0]
                label = pack['label'][0]
                # Removing label of Unlabled data
                if unlabeled:
                    # label = torch.ones_like(label) # all pred(low performance)
                    # hard label prediction
                    scores = [model.get_featnscore(img[0].cuda(non_blocking=True))[1] for img in pack['img']]
                    label = (torch.sigmoid(sum(scores)) >= 0.5).float().detach().cpu()
                size = pack['size']
                label = F.pad(label, (1, 0), 'constant', 1.0)

                outputs = [model(img[0].cuda(non_blocking=True), label.cuda(non_blocking=True).unsqueeze(-1).unsqueeze(-1)) for img in pack['img']]

                # multi-scale fusion
                IS_CAM_list = [output[1].cpu() for output in outputs]
                IS_CAM_list = [F.interpolate(torch.unsqueeze(o, 1), size, mode='bilinear', align_corners=False) for o in IS_CAM_list]
                IS_CAM = torch.sum(torch.stack(IS_CAM_list, 0), 0)[:,0]
                IS_CAM /= F.adaptive_max_pool2d(IS_CAM, (1, 1)) + 1e-5
                IS_CAM = IS_CAM.cpu().numpy()

                # visualize IS-CAM
                if args.visualize:
                    orig_img = np.array(Image.open(pack['img_path'][0]).convert('RGB'))
                    draw_heatmap(IS_CAM.copy(), label, orig_img, os.path.join(args.session_name, 'visual'), img_name)

                # save IS_CAM
                valid_cat = torch.nonzero(label)[:, 0].cpu().numpy()
                IS_CAM = IS_CAM[valid_cat]
                np.save(os.path.join(args.session_name, 'npy', img_name + '.npy'),  {"keys": valid_cat, "IS_CAM": IS_CAM})

                if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                    print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')

        save_cams(databin, data_loader)
        if dataset_ulb is not None: ###
            save_cams(databin_ulb, data_loader_ulb, unlabeled=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="network.resnet50_SIPE", type=str)
    parser.add_argument("--num_workers", default=os.cpu_count()//2, type=int)
    parser.add_argument("--session_name", default="exp", type=str)
    parser.add_argument("--ckpt", default="final.pth", type=str)
    parser.add_argument("--visualize", default=True, type=bool)
    parser.add_argument("--dataset_root", default="../PascalVOC2012/VOCdevkit/VOC2012", type=str)
    parser.add_argument("--dataset", default="voc", type=str)
    parser.add_argument("--lb_list", default=None, type=str) ###
    parser.add_argument("--ulb_list", default=None, type=str) ###

    args = parser.parse_args()

    os.makedirs(os.path.join(args.session_name, 'npy'), exist_ok=True)
    os.makedirs(os.path.join(args.session_name, 'visual'), exist_ok=True)
    pyutils.Logger(os.path.join(args.session_name, 'infer.log'))
    print(vars(args))

    assert args.dataset in ['voc', 'coco'], 'Dataset must be voc or coco in this project.'

    if args.dataset == 'voc':
        dataset_root = args.dataset_root #'../PascalVOC2012/VOCdevkit/VOC2012'
        model = getattr(importlib.import_module(args.network), 'CAM')(num_cls=21)
        if args.lb_list is None:
            dataset = data_voc.VOC12ClsDatasetMSF('data/train_' + args.dataset + '.txt', voc12_root=dataset_root, scales=(1.0, 0.5, 1.5, 2.0))
        else:
            dataset = data_voc.VOC12ClsDatasetMSF(args.lb_list, voc12_root=dataset_root, scales=(1.0, 0.5, 1.5, 2.0))
        if args.ulb_list is not None:
            dataset_ulb = data_voc.VOC12ClsDatasetMSF(args.ulb_list, voc12_root=dataset_root, scales=(1.0, 0.5, 1.5, 2.0))
        else:
            dataset_ulb = None

    elif args.dataset == 'coco':
        dataset_root = args.dataset_root #"../ms_coco_14&15/images"
        model = getattr(importlib.import_module(args.network), 'CAM')(num_cls=81)
        
        if args.lb_list is None:
            dataset = data_coco.COCOClsDatasetMSF('data/train_' + args.dataset + '.txt', coco_root=dataset_root, scales=(1.0, 0.5, 1.5, 2.0))
        else:
            dataset = data_coco.COCOClsDatasetMSF(args.lb_list, coco_root=dataset_root, scales=(1.0, 0.5, 1.5, 2.0))
        if args.ulb_list is not None:
            dataset_ulb = data_coco.COCOClsDatasetMSF(args.ulb_list, coco_root=dataset_root, scales=(1.0, 0.5, 1.5, 2.0))
        else:
            dataset_ulb = None

    checkpoint = torch.load(args.session_name + '/ckpt/' + args.ckpt)
    model.load_state_dict(checkpoint['net'], strict=True)
    model.eval()

    n_gpus = torch.cuda.device_count()

    dataset = torchutils.split_dataset(dataset, n_gpus)
    if dataset_ulb is not None: ###
        dataset_ulb = torchutils.split_dataset(dataset_ulb, n_gpus)

    print('[ ', end='') 
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, dataset_ulb, args), join=True)
    print(']')

    torch.cuda.empty_cache()