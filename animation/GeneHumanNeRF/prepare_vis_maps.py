import os
import numpy as np
import cv2

from tqdm import tqdm
# import sys
# sys.path.append('./')

import pickle

import torch
import torch.nn.functional as F
from lib.config import cfg
# from .nerf_net_utils import *
from lib.utils.blend_utils import *

from lib.utils import sample_utils

from lib.utils.file_utils import list_files, split_path

from lib.third_parties.smpl.smpl_numpy import SMPL

MODEL_DIR = 'lib/third_parties/smpl/models'


def get_vis_map(pvertices, faces, Rh, Th, cam_R, cam_T):
        '''
            Args:
                - pvertices, [N_batch, N_points, 3]
                - faces, [N_faces, 3]
                - Rh, [3, 3]
                - Th, [1, 3]
                - cam_R, [3, 3]
                - cam_T, [1, 3]

            return:
                - visibility, [N_points]
        '''

        w_pvertices = pose_points_to_world_points(pvertices, Rh[None], Th[None])

        R, T = cam_R, cam_T

        c_pvertices = torch.matmul(w_pvertices, R[None].transpose(2, 1)) + T[None]

        ### put the vertices with the (-1, 1)
        c_bounds, _ = torch.abs(c_pvertices[0]).max(0)
        c_pvertices = c_pvertices[0]/(c_bounds[None, ...]+1e-6)

        (xy, z) = c_pvertices.split([2, 1], dim=1)
        visibility = sample_utils.get_visibility(xy, z, faces[:, [0, 2, 1]]).flatten()

        return visibility == 0

if __name__ == "__main__":
    '''
    运行方式：
    python prepare_vis_maps.py --cfg_file configs/sdf_pdf/zju_mocap.yaml prepare_vis_map 377
    '''

    data_root = 'data/humman_eval'
    # data_root = cfg.train_dataset.data_root
    
    subjects = os.listdir(data_root)
    # subjects = ['387_eval', '393_eval', '394_eval']

    # subjects = [str(cfg.prepare_vis_map)]

    for subject in subjects:
        dataset_path = os.path.join(data_root, subject)
        out_path = os.path.join(dataset_path, 'vis_maps')
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        img_paths = list_files(os.path.join(dataset_path, 'images'),
                               exts=['.png'])
        ims = [split_path(ipath)[1] for ipath in img_paths]

        cameras = None
        with open(os.path.join(dataset_path, 'cameras.pkl'), 'rb') as f: 
            cameras = pickle.load(f)

        mesh_infos = None
        with open(os.path.join(dataset_path, 'mesh_infos.pkl'), 'rb') as f:   
            mesh_infos = pickle.load(f)

        smpl_model = SMPL(sex='neutral', model_dir=MODEL_DIR)
        faces = smpl_model.faces

        print('Preparing {} ...'.format(subject))
        for frame_name in tqdm(ims):

            RT = cameras[frame_name]['extrinsics'].astype(np.float32).copy()

            Rh = mesh_infos[frame_name]['Rh'].astype('float32')
            Th = mesh_infos[frame_name]['Th'].astype('float32')

            Rh = cv2.Rodrigues(Rh)[0].astype(np.float32)

            pvertices = mesh_infos[frame_name]['vertices'].astype('float32')
            
            vis_map_args = {
                            'pvertices': torch.from_numpy(pvertices[None]).cuda(),
                            'faces': torch.from_numpy(faces).long().cuda(),
                            'Rh': torch.from_numpy(Rh).cuda(),
                            'Th': torch.from_numpy(Th[None]).cuda(),
                            'cam_R': torch.from_numpy(RT[:3, :3]).cuda(),
                            'cam_T': torch.from_numpy(RT[:3, 3][None]).cuda()
                        }
            vis_map = get_vis_map(**vis_map_args)

            np.save(os.path.join(out_path, frame_name+'.npy'), vis_map.cpu().numpy())