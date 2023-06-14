import os
import sys

from shutil import copyfile

import pickle
import yaml
import numpy as np
from tqdm import tqdm

import cv2

from pathlib import Path

sys.path.append(str(Path(os.getcwd()).resolve().parents[1]))

from third_parties.smpl.smpl_numpy import SMPL
from core.utils.file_util import split_path
from core.utils.image_util import load_image, save_image, to_3ch_image

from core.utils.geometry_utils import rotation_matrix_to_angle_axis

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('cfg',
                    '387_eval.yaml',
                    'the path of config file')

MODEL_DIR = '../../third_parties/smpl/models'


def parse_config():
    config = None
    with open(FLAGS.cfg, 'r') as file:
        config = yaml.full_load(file)

    return config


def prepare_dir(output_path, name):
    out_dir = os.path.join(output_path, name)
    os.makedirs(out_dir, exist_ok=True)

    return out_dir


def get_mask(subject_dir, img_name):
    msk_path = os.path.join(subject_dir, 'mask',
                            img_name)[:-4] + '.png'
    msk = np.array(load_image(msk_path))[:, :, 0]
    msk = (msk != 0).astype(np.uint8)

    msk_path = os.path.join(subject_dir, 'mask_cihp',
                            img_name)[:-4] + '.png'
    msk_cihp = np.array(load_image(msk_path))[:, :, 0]
    msk_cihp = (msk_cihp != 0).astype(np.uint8)

    msk = (msk | msk_cihp).astype(np.uint8)
    msk[msk == 1] = 255

    return msk


def main(argv):
    del argv  # Unused.

    cfg = parse_config()
    subject = cfg['dataset']['subject']
    sex = cfg['dataset']['sex']
    max_frames = cfg['max_frames']

    dataset_dir = cfg['dataset']['genebody_path']
    subject_dir = os.path.join(dataset_dir, subject)
    smpl_params_dir = os.path.join(subject_dir, "smpl_param")

    select_view = []

    train_views = cfg['training_view']

    start_idx = cfg['start_idx']+100

    anno_path = os.path.join(subject_dir, 'annots.npy')
    annots = np.load(anno_path, allow_pickle=True).item()

    # load cameras
    cams = annots['cams']

    total_views = cams.keys()

    select_view = [view for view in total_views if int(view) not in train_views]

    cam_Ks = np.array([cams[view]['K'] for view in select_view]).astype('float32')
    cam_Rs = np.array([cams[view]['c2w_R'] for view in select_view]).astype('float32')
    cam_Ts = np.array([cams[view]['c2w_T'] for view in select_view]).astype('float32')
    cam_Ds = np.array([cams[view]['D'] for view in select_view]).astype('float32')

    cam_Rs = cam_Rs.transpose(0, 2, 1)
    cam_Ts = - np.matmul(cam_Rs, cam_Ts)

    K = cam_Ks  # view_num*3*3
    D = cam_Ds  # view_num*5
    E = np.zeros((cam_Ks.shape[0], 4, 4)).astype('float32')  # view_num*4*4
    cam_T = cam_Ts[..., 0]
    E[:, :3, :3] = cam_Rs
    E[:, :3, 3] = cam_T
    E[:, 3, 3] = 1.  # view_num*4*4

    ims_0 = sorted(os.listdir(os.path.join(subject_dir, 'image', '00')), key=lambda x: int(x[:4]))[100:100+max_frames]

    img_paths = [[os.path.join(view_i, im_i) for view_i in select_view] for im_i in ims_0]


    img_paths = np.stack(img_paths, 0)

    index_keep = np.array(range(0, len(img_paths), cfg['skip']))
    img_paths = img_paths[index_keep]
    
    output_path = os.path.join(cfg['output']['dir'],
                               subject if 'name' not in cfg['output'].keys() else cfg['output']['name'])
    os.makedirs(output_path, exist_ok=True)
    out_img_dir = prepare_dir(output_path, 'images')
    out_mask_dir = prepare_dir(output_path, 'masks')

    # copy config file
    copyfile(FLAGS.cfg, os.path.join(output_path, 'config.yaml'))

    smpl_model = SMPL(sex=sex, model_dir=MODEL_DIR)

    cameras = {}
    mesh_infos = {}
    all_betas = []
    [all_betas.append([]) for i in range(len(img_paths))]

    for idx_frame, path_frame in enumerate(tqdm(img_paths)):
        for idx_camera, path_camera in enumerate(path_frame):
            real_idx_frame = idx_frame * cfg['skip'] + start_idx
            real_idx_camera = select_view[idx_camera]
            out_name = 'frame_{:06d}_camera_{}'.format(real_idx_frame, real_idx_camera)

            img_path = os.path.join(subject_dir, 'image', path_camera)

            # load image
            img = np.array(load_image(img_path))
            
            if subject in ['313', '315']:
                smpl_idx = real_idx_frame + 1  # index begin with 1
            else:
                smpl_idx = real_idx_frame

            with open(os.path.join(smpl_params_dir, "{:04d}.pkl".format(smpl_idx)), 'rb') as f: 
                smpl_params = pickle.load(f)

            smpl_pose = smpl_params['body_pose'][0]
            smpl_shapes = smpl_params['betas'][0].numpy()

            smpl_pose = rotation_matrix_to_angle_axis(smpl_pose).reshape(-1)
            smpl_Rh = rotation_matrix_to_angle_axis(smpl_params['global_orient'][0])[0].numpy()

            smpl_pose_root = np.zeros(3)
            smpl_pose = np.concatenate([smpl_pose_root, smpl_pose])

            _, tpose_joints = smpl_model(np.zeros(72), smpl_shapes)
            pelvis_pos = tpose_joints[0].copy()
            smpl_Th = smpl_params['joints'][:, 0].detach().cpu().numpy() - cv2.Rodrigues(smpl_Rh)[0].dot(pelvis_pos)

            betas = smpl_shapes
            poses = smpl_pose
            Rh = smpl_Rh
            Th = smpl_Th[0]

            all_betas[idx_frame].append(betas)

            # write camera info
            cameras[out_name] = {
                'intrinsics': K[idx_camera],
                'extrinsics': E[idx_camera],
                'distortions': D[idx_camera]
            }

            # write mesh info
            _, tpose_joints = smpl_model(np.zeros_like(poses), betas)
            _, joints = smpl_model(poses, betas)
            mesh_infos[out_name] = {
                'Rh': Rh,
                'Th': Th,
                'poses': poses,
                'joints': joints,
                'tpose_joints': tpose_joints
            }

            # load and write mask
            mask_path_camera = path_camera.split('/')
            mask_path_camera[-1] = 'mask' + mask_path_camera[-1].split('.')[0] + '.png'
            mask_path_camera = '/'.join(mask_path_camera)

            mask = cv2.imread(os.path.join(subject_dir, 'mask', mask_path_camera))

            save_image(((mask>128)*255).astype(np.uint8),
                       os.path.join(out_mask_dir, out_name + '.png'))

            # write image
            out_image_path = os.path.join(out_img_dir, '{}.png'.format(out_name))
            save_image(img, out_image_path)

    # write camera infos
    with open(os.path.join(output_path, 'cameras.pkl'), 'wb') as f:
        pickle.dump(cameras, f)

    # write mesh infos
    with open(os.path.join(output_path, 'mesh_infos.pkl'), 'wb') as f:
        pickle.dump(mesh_infos, f)

    # write canonical joints
    # eliminate duplicate values ​​in all_betas
    all_betas = [all_betas[i][0] for i in range(len(all_betas))]
    avg_betas = np.mean(np.stack(all_betas, axis=0), axis=0)
    smpl_model = SMPL(sex, model_dir=MODEL_DIR)
    _, template_joints = smpl_model(np.zeros(72), avg_betas)
    with open(os.path.join(output_path, 'canonical_joints.pkl'), 'wb') as f:
        pickle.dump(
            {
                'joints': template_joints,
            }, f)


if __name__ == '__main__':
    app.run(main)
