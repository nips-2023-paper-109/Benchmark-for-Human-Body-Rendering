import matplotlib.pyplot as plt
import numpy as np
from lib.config import cfg
import cv2
import os
from termcolor import colored
import torch


class Visualizer:
    def __init__(self):
        data_dir = 'data/trained_model/deform/{}'.format(cfg.exp_name)
        print(colored('the results are saved at {}'.format(data_dir),
                      'yellow'))

    def visualize(self, output, batch):
        if torch.is_tensor(output['rgb_map']):
            rgb_pred = output['rgb_map'][0].detach().cpu().numpy()

        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = batch['H'].item(), batch['W'].item()
        mask_at_box = mask_at_box.reshape(H, W)

        img_pred = np.zeros((H, W, 3))
        img_pred[mask_at_box] = rgb_pred
        img_pred = img_pred[..., [2, 1, 0]]

        # view_index = batch['view_index'].item()
        frame_root = 'data/trained_model/deform/{}/epoch_{:04d}'.format(
            cfg.exp_name, batch['epoch'])
        os.system('mkdir -p {}'.format(frame_root))
        frame_index = batch['frame_index'].item()
        cv2.imwrite(
            os.path.join(frame_root, 'frame{:04d}.png'.format(frame_index)),
            img_pred * 255)
        
        ### KK ###
        ## visualiza normal map in posed space
        # nml_pred = output['normal_map'][0].detach().cpu().numpy()
        # normal_pred = np.zeros((H, W, 3))
        # normal_pred[mask_at_box] = (nml_pred + 1)/2
        # normal_pred = normal_pred[..., [2, 1, 0]]

        # img_normal_pred = np.hstack([img_pred, normal_pred])

        # cv2.imwrite(os.path.join(frame_root, 'frame{:04d}.png'.format(frame_index)),
        #             img_normal_pred * 255)