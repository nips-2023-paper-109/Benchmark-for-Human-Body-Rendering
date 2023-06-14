import os
import skimage
import json

from skimage.metrics import structural_similarity

import torch
import numpy as np
from tqdm import tqdm
import cv2

from core.data import create_dataloader
from core.nets import create_network
from core.utils.train_util import cpu_data_to_gpu
from core.utils.image_util import ImageWriter, to_8b_image, to_8b3ch_image

from configs import cfg
from torch.utils.tensorboard import SummaryWriter

# from third_parties.lpips import LPIPS
import lpips as lp

EXCLUDE_KEYS_TO_GPU = ['frame_name',
                       'img_width', 'img_height', 'ray_mask']

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def load_network():
    model = create_network()

    ## kk
    ckpt_path = os.path.join(cfg.logdir, f'{cfg.load_net}.tar')
    
    # wild_ckpt = 'experiments/human_nerf/wild/monocular/adventure/latest.tar'
    # ckpt_path = wild_ckpt

    # ckpt_path = 'experiments/human_nerf/zju_mocap/p387/adventure/iter_30000.tar'
    ##

    ckpt = torch.load(ckpt_path, map_location='cuda:0')
    model.load_state_dict(ckpt['network'], strict=False)
    print('load network from ', ckpt_path)
    return model.cuda().deploy_mlps_to_secondary_gpus()

def unpack_alpha_map(alpha_vals, ray_mask, width, height):
    alpha_map = np.zeros((height * width), dtype='float32')
    alpha_map[ray_mask] = alpha_vals
    return alpha_map.reshape((height, width))

def unpack_to_image(width, height, ray_mask, bgcolor,
                    rgb, alpha, truth=None):
    rgb_image = np.full((height * width, 3), bgcolor, dtype='float32')
    truth_image = np.full((height * width, 3), bgcolor, dtype='float32')

    rgb_image[ray_mask] = rgb
    rgb_image = to_8b_image(rgb_image.reshape((height, width, 3)))

    if truth is not None:
        truth_image[ray_mask] = truth
        truth_image = to_8b_image(truth_image.reshape((height, width, 3)))

    alpha_map = unpack_alpha_map(alpha, ray_mask, width, height)
    alpha_image = to_8b3ch_image(alpha_map)

    return rgb_image, alpha_image, truth_image

def psnr_metric(img_pred, img_gt):
    ''' Caculate psnr metric
        Args:
            img_pred: ndarray, W*H*3, range 0-1
            img_gt: ndarray, W*H*3, range 0-1

        Returns:
            psnr metric: scalar
    '''
    mse = np.mean((img_pred - img_gt) ** 2)
    psnr = -10 * np.log(mse) / np.log(10)
    return psnr.item()


def lpips_metric(model, pred, target):
    # convert range from 0-1 to -1-1
    processed_pred = torch.from_numpy(pred).float().unsqueeze(0).to(cfg.primary_gpus[0]) * 2. - 1.
    processed_target=torch.from_numpy(target).float().unsqueeze(0).to(cfg.primary_gpus[0]) * 2. - 1.

    lpips_loss = model(processed_pred.permute(0, 3, 1, 2),
                       processed_target.permute(0, 3, 1, 2))
    return torch.mean(lpips_loss).cpu().detach().item()

def eval_model(render_folder_name='eval', show_truth=True, show_alpha=True):
    cfg.perturb = 0.

    # cfg.bbox_offset = 0.05

    model = load_network()
    test_loader = create_dataloader('eval_pose')
    writer = ImageWriter(
        output_dir=os.path.join(cfg.logdir, cfg.load_net),
        exp_name=render_folder_name)
    log_dir = os.path.join(cfg.logdir, cfg.load_net, render_folder_name, 'log')
    swriter = SummaryWriter(log_dir)

    model.eval()
    PSNRA = []
    SSIMA = []
    LPIPSA = []
    # create lpip model and config
    # lpips_model = LPIPS(net='vgg')
    lpips_model = lp.LPIPS(net='alex')

    set_requires_grad(lpips_model, requires_grad=False)
    lpips_model.to(cfg.primary_gpus[0])

    # ray_mask_path = '../neuralbody/data/result/if_nerf/genebody_{}/ray_masks'.format(cfg.subject)
    ray_mask_path = '../neuralbody/data/result/if_nerf/humman_{}/ray_masks'.format(cfg.subject)
    # mask_name = 'frame0000_view0001_raymask.png'


    for idx, batch in enumerate(tqdm(test_loader)):
        for k, v in batch.items():
            batch[k] = v[0]

        data = cpu_data_to_gpu(
            batch,
            exclude_keys=EXCLUDE_KEYS_TO_GPU + ['target_rgbs'])

        with torch.no_grad():
            net_output = model(**data, iter_val=cfg.eval_iter)

        rgb = net_output['rgb']
        alpha = net_output['alpha']

        width = batch['img_width']
        height = batch['img_height']
        ray_mask = batch['ray_mask']

        # *_img: ndarray, (512, 512, 3), value range 0-255
        rgb_img, alpha_img, truth_img = \
            unpack_to_image(
                width, height, ray_mask, np.array(cfg.bgcolor) / 255.,
                rgb.data.cpu().numpy(),
                alpha.data.cpu().numpy(),
                batch['target_rgbs'])
        # convert image to 0-1
        rgb_img_norm = rgb_img / 255.
        truth_img_norm = truth_img / 255.
        # caculate the metric
        # psnr = psnr_metric(rgb_img_norm, truth_img_norm)
        # ssim = structural_similarity(rgb_img_norm, truth_img_norm, multichannel=True)
        # lpips = lpips_metric(model=lpips_model, pred=rgb_img_norm, target=truth_img_norm)

        # psnr = psnr_metric(rgb.cpu().numpy(), batch['target_rgbs'].numpy())

        ray_mask_name = 'frame{}_view00{:02d}_raymask.png'.format(batch['frame_name'][8:12], int(batch['frame_name'][-2:]))
        
        ray_mask = cv2.imread(os.path.join(ray_mask_path, ray_mask_name))[..., 0] > 0
        
        psnr = psnr_metric(rgb_img_norm[ray_mask], truth_img_norm[ray_mask])

        # ray_mask = ray_mask.cpu().numpy().reshape(512, 512)

        x, y, w, h = cv2.boundingRect(ray_mask.astype(np.uint8))
        img_pred_rect = rgb_img_norm[y:y + h, x:x + w]
        img_gt_rect = truth_img_norm[y:y + h, x:x + w]
        ssim = structural_similarity(img_pred_rect, img_gt_rect, multichannel=True)
        lpips = lpips_metric(model=lpips_model, pred=img_pred_rect, target=img_gt_rect)

        # import IPython; IPython.embed(); exit()

        imgs = [rgb_img[y:y + h, x:x + w]]
        if show_truth:
            imgs.append(truth_img[y:y + h, x:x + w])
        if show_alpha:
            imgs.append(alpha_img[y:y + h, x:x + w])
        img_out = np.concatenate(imgs, axis=1)
        writer.append(img_out, img_name=batch['frame_name'])

        swriter.add_scalar("psnr", psnr, idx)
        swriter.add_scalar("ssim", ssim, idx)
        swriter.add_scalar("lpips", lpips, idx)
        PSNRA.append(psnr)
        SSIMA.append(ssim)
        LPIPSA.append(lpips)

    psnr_final = np.mean(PSNRA).item()
    ssim_final = np.mean(SSIMA).item()
    lpips_final = np.mean(LPIPSA).item()

    metrics = {'psnr': psnr_final, 
                'ssim': ssim_final,
                'lpips': lpips_final}
        
    metrics_json = json.dumps(metrics)
    f = open(os.path.join(cfg.logdir, cfg.load_net, render_folder_name, 'log', 'metrics.json'), 'w')
    f.write(metrics_json)
    f.close()

    print(f"PSNR is {psnr_final}, SSIM is {ssim_final}, LPIPS is {lpips_final}")
    swriter.add_scalar("summary", psnr_final,1)
    swriter.add_scalar("summary", ssim_final, 2)
    swriter.add_scalar("summary", lpips_final, 3)
    writer.finalize()
    swriter.close()

    # psnr_per_view = np.array(PSNRA).reshape(-1, 19).mean(0)
    # print("psnr_per_view: ", psnr_per_view)


if __name__ == '__main__':
    eval_model(render_folder_name='eval_pose')

