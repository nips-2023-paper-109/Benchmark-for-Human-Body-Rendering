import numpy as np
from lib.config import cfg
from skimage.metrics import structural_similarity as compare_ssim
import os
import cv2
from termcolor import colored
import json
import torch

import lpips

class Evaluator:
    def __init__(self):
        self.mse = []
        self.psnr = []
        self.ssim = []
        self.lpips = []
        self.lpips_metric = lpips.LPIPS(net='alex')

    def psnr_metric(self, img_pred, img_gt):
        mse = np.mean((img_pred - img_gt)**2)
        psnr = -10 * np.log(mse) / np.log(10)
        return psnr

    def ssim_metric(self, rgb_pred, rgb_gt, batch):
        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = batch['H'].item(), batch['W'].item()
        mask_at_box = mask_at_box.reshape(H, W)

        # convert the pixels into an image
        img_pred = np.zeros((H, W, 3))
        img_pred[mask_at_box] = rgb_pred
        img_gt = np.zeros((H, W, 3))
        img_gt[mask_at_box] = rgb_gt

        orig_img_pred = img_pred.copy()
        orig_img_gt = img_gt.copy()

        if 'crop_bbox' in batch:
            img_pred = fill_image(img_pred, batch)
            img_gt = fill_image(img_gt, batch)

        result_dir = os.path.join(cfg.result_dir, 'comparison')
        os.system('mkdir -p {}'.format(result_dir))
        frame_index = batch['frame_index'].item()
        view_index = batch['cam_ind'].item()

        # cv2.imwrite(
        #     '{}/frame{:04d}_view{:04d}.png'.format(result_dir, frame_index,
        #                                            view_index),
        #     (img_pred[..., [2, 1, 0]] * 255))
        # cv2.imwrite(
        #     '{}/frame{:04d}_view{:04d}_gt.png'.format(result_dir, frame_index,
        #                                               view_index),
        #     (img_gt[..., [2, 1, 0]] * 255))

        # if 'normal_map' in batch.keys():
        #     normal_pred = np.zeros((H, W, 3))
        #     normal_pred[mask_at_box] = (batch['normal_map'] + 1)/2
        #     cv2.imwrite(
        #         '{}/frame{:04d}_view{:04d}_pred_normal.png'.format(result_dir, frame_index,
        #                                                 view_index),
        #         (normal_pred[..., [2, 1, 0]] * 255))

        # crop the object region
        x, y, w, h = cv2.boundingRect(mask_at_box.astype(np.uint8))
        # img_pred = orig_img_pred[y:y + h, x:x + w]
        # img_gt = orig_img_gt[y:y + h, x:x + w]
        # compute the ssim
        # ssim = compare_ssim(img_pred, img_gt, multichannel=True)
        ssim = compare_ssim(orig_img_pred[y:y + h, x:x + w], orig_img_gt[y:y + h, x:x + w], multichannel=True)


        # cv2.putText(img_pred, 'PSNR: {:.4}, SSIM: {:.4}'.format(self.psnr[-1], ssim), \
        #     (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (1, 0, 0), 1)

        selected_ref_imgs = batch['ref_imgs'][batch['selected_ref_ind']].permute(0,2,3,1).detach().cpu().numpy()

        vis_list = [selected_ref_imgs[0],
                    selected_ref_imgs[1],
                    selected_ref_imgs[2],
                    img_gt,
                    img_pred]

        vis = np.hstack(vis_list)

        cv2.imwrite(
            '{}/frame{:04d}_view{:04d}.png'.format(result_dir, frame_index,
                                                   view_index),
            (vis[..., [2, 1, 0]] * 255))

        return ssim

    def evaluate(self, output, batch):
        rgb_pred = output['rgb_map'][0].detach().cpu().numpy()
        rgb_gt = batch['rgb'][0].detach().cpu().numpy()

        if rgb_gt.sum() == 0:
            return

        mse = np.mean((rgb_pred - rgb_gt)**2)
        self.mse.append(mse)

        psnr = self.psnr_metric(rgb_pred, rgb_gt)
        self.psnr.append(psnr)

        if 'normal_map' in output.keys():
            batch['normal_map'] = output['normal_map'][0].detach().cpu().numpy()

        ssim = self.ssim_metric(rgb_pred, rgb_gt, batch)
        self.ssim.append(ssim)

        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
        mask_at_box = mask_at_box.reshape(H, W)
        # convert the pixels into an image
        white_bkgd = int(cfg.white_bkgd)
        img_pred = np.zeros((H, W, 3)) + white_bkgd
        img_pred[mask_at_box] = rgb_pred
        img_gt = np.zeros((H, W, 3)) + white_bkgd
        img_gt[mask_at_box] = rgb_gt

        x, y, w, h = cv2.boundingRect(mask_at_box.astype(np.uint8))
        img_pred = img_pred[y:y + h, x:x + w]
        img_gt = img_gt[y:y + h, x:x + w]
        lpips = self.lpips_metric(torch.from_numpy(img_pred).permute(2,0,1)[None].to(torch.float32)*2-1, \
            torch.from_numpy(img_gt).permute(2,0,1)[None].to(torch.float32)*2-1)
        self.lpips.append(lpips.detach().numpy())


    def summarize(self):
        result_dir = cfg.result_dir
        print(
            colored('the results are saved at {}'.format(result_dir),
                    'yellow'))

        # result_path = os.path.join(cfg.result_dir, 'metrics.npy')
        # os.system('mkdir -p {}'.format(os.path.dirname(result_path)))

        os.system('mkdir -p {}'.format(result_dir))
        metrics = {'mse': np.mean(self.mse).item(), 
            'psnr': np.mean(self.psnr).item(), 
            'ssim': np.mean(self.ssim).item(),
            'lpips': np.mean(self.lpips).item()}
        # np.save(result_path, metrics)  
        
        metrics_json = json.dumps(metrics)
        f = open(os.path.join(result_dir, 'metrics.json'), 'w')
        f.write(metrics_json)
        f.close()

        print('mse: {}'.format(np.mean(self.mse)))
        print('psnr: {}'.format(np.mean(self.psnr)))
        print('ssim: {}'.format(np.mean(self.ssim)))
        print('lpips: {}'.format(np.mean(self.lpips)))
        self.mse = []
        self.psnr = []
        self.ssim = []
        self.lpips = []


def fill_image(img, batch):
    orig_H, orig_W = batch['orig_H'].item(), batch['orig_W'].item()
    full_img = np.zeros((orig_H, orig_W, 3))
    bbox = batch['crop_bbox'][0].detach().cpu().numpy()
    height = bbox[1, 1] - bbox[0, 1]
    width = bbox[1, 0] - bbox[0, 0]
    full_img[bbox[0, 1]:bbox[1, 1],
             bbox[0, 0]:bbox[1, 0]] = img[:height, :width]
    return full_img
