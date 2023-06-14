import torch.utils.data as data
from lib.utils import base_utils
from PIL import Image
import numpy as np
import json
import os
import imageio
import cv2
from lib.config import cfg
from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils
from plyfile import PlyData
from lib.utils import render_utils
from lib.utils.file_utils import list_files, split_path
import pickle

from lib.third_parties.smpl.smpl_numpy import SMPL

MODEL_DIR = 'lib/third_parties/smpl/models'

SMPL_PARENT = np.array([4294967295,          0,          0,          0,          1,
                2,          3,          4,          5,          6,
                7,          8,          9,          9,          9,
               12,         13,         14,         16,         17,
               18,         19,         20,         21])


class Dataset(data.Dataset):
    def __init__(self, data_root, human, ann_file, split):
        super(Dataset, self).__init__()

        self.dataset_path = data_root
        self.image_dir = os.path.join(data_root, 'images')

        ## self.canonical_joints.shape = [24, 3]
        self.canonical_joints, self.canonical_bbox = \
            self.load_canonical_joints()

        self.cameras = self.load_train_cameras()
        self.mesh_infos = self.load_train_mesh_infos()

        self.ims = self.load_train_frames()[:cfg.num_train_frame]

        # self.ims = self.load_train_frames()

        N_inputs = 10
        skip = cfg.num_train_frame//N_inputs
        self.ref_ims = self.ims[::skip][:N_inputs]
        print(self.ref_ims)

        self.frame_interval = cfg.num_train_frame//20
        self.ims = self.ims[::self.frame_interval][:20]
        # self.ims = self.ims[cfg.num_train_frame:][:100]

        self.data_root = data_root
        self.human = human
        self.split = split

        self.smpl_model = SMPL(sex='neutral', model_dir=MODEL_DIR)

        self.joints = self.canonical_joints.astype(np.float32)
        self.parents = SMPL_PARENT
        self.weights = self.smpl_model.weights.astype(np.float32)

        self.big_A, self.big_poses = self.load_bigpose()

        self.nrays = cfg.N_rand
        self.num_cams = 1
        self.test_view = 0

        avg_betas = np.mean(np.stack([self.mesh_infos[frame_name]['betas'] for frame_name in self.mesh_infos], axis=0), axis=0)
        
        self.bigpose_vertices, _ = self.smpl_model(self.big_poses.reshape(-1), avg_betas)

        self.ref_infos = self.load_ref_frames()
        faces = self.smpl_model.faces

        self.ref_infos.update({'faces': faces})

    def load_ref_frames(self):
        ref_imgs = []
        ref_msks = []
        ref_RTs = []
        ref_Ks = []
        ref_As = []
        ref_Rhs = []
        ref_Ths = []
        ref_pvertices = []
        ref_vis_maps = []
        for frame_name in self.ref_ims:
            img_path = os.path.join(self.image_dir, '{}.png'.format(frame_name))
            img = imageio.imread(img_path).astype(np.float32) / 255.
            H, W = int(img.shape[0] * cfg.ratio), int(img.shape[1] * cfg.ratio)
            
            mask_path = os.path.join(self.dataset_path, 
                                'masks', 
                                '{}.png'.format(frame_name))
            msk = (imageio.imread(mask_path)[..., 0] > 0).astype(np.uint8)

            if not cfg.eval and cfg.erode_edge:
                border = 5
                kernel = np.ones((border, border), np.uint8)
                msk_erode = cv2.erode(msk.copy(), kernel)
                msk_dilate = cv2.dilate(msk.copy(), kernel)
                msk[(msk_dilate - msk_erode) == 1] = 100

            ### 这两句有问题！！！会导致worker>1的时候dataloader segmentation error
            ### 不清楚具体原因，也很难理解这是为啥…
            # msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
            # img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

            if cfg.mask_bkgd:
                img[msk == 0] = 0

            vis_map_path = os.path.join(self.dataset_path, 
                                'vis_maps', 
                                '{}.npy'.format(frame_name))

            vis_map = np.load(vis_map_path)

            wpts, pvertices, A, Rh, Th, poses, nearest_frame_index = self.prepare_input(frame_name)

            ref_camera = self.cameras[frame_name]

            ref_RT = ref_camera['extrinsics'].astype(np.float32).copy()
            ref_K = ref_camera['intrinsics'].astype(np.float32).copy()
            ref_K[:2] = ref_K[:2] * cfg.ratio

            Rh = cv2.Rodrigues(Rh)[0].astype(np.float32)

            ref_imgs.append(img)
            ref_msks.append(msk)
            ref_RTs.append(ref_RT)
            ref_Ks.append(ref_K)
            ref_As.append(A)
            ref_Rhs.append(Rh)
            ref_Ths.append(Th)
            ref_pvertices.append(pvertices)
            ref_vis_maps.append(vis_map)

        return {
                'ref_imgs': np.stack(ref_imgs, axis=0),
                'ref_msks': np.stack(ref_msks, axis=0),
                'ref_RTs': np.stack(ref_RTs, axis=0),
                'ref_Ks': np.stack(ref_Ks, axis=0),
                'ref_As': np.stack(ref_As, axis=0),
                'ref_Rhs': np.stack(ref_Rhs, axis=0),
                'ref_Ths': np.stack(ref_Ths, axis=0),
                'ref_pvertices': np.stack(ref_pvertices, axis=0),
                'ref_vis_maps': np.stack(ref_vis_maps, axis=0)
                }

    def load_bigpose(self):
        big_poses = np.zeros([len(self.joints), 3]).astype(np.float32).ravel()
        angle = 30
        big_poses[5] = np.deg2rad(angle)
        big_poses[8] = np.deg2rad(-angle)
        big_poses = big_poses.reshape(-1, 3)
        big_A = if_nerf_dutils.get_rigid_transformation(
            big_poses, self.joints, self.parents)
        big_A = big_A.astype(np.float32)
        return big_A, big_poses

    def load_train_mesh_infos(self):
        mesh_infos = None
        with open(os.path.join(self.dataset_path, 'mesh_infos.pkl'), 'rb') as f:   
            mesh_infos = pickle.load(f)

        for frame_name in mesh_infos.keys():
            bbox = self.skeleton_to_bbox(mesh_infos[frame_name]['joints'])
            mesh_infos[frame_name]['bbox'] = bbox

        return mesh_infos

    def load_canonical_joints(self):
        cl_joint_path = os.path.join(self.dataset_path, 'canonical_joints.pkl')
        with open(cl_joint_path, 'rb') as f:
            cl_joint_data = pickle.load(f)
        canonical_joints = cl_joint_data['joints'].astype('float32')
        canonical_bbox = self.skeleton_to_bbox(canonical_joints)

        return canonical_joints, canonical_bbox
    
    def skeleton_to_bbox(self, skeleton):
        min_xyz = np.min(skeleton, axis=0) - cfg.bbox_offset
        max_xyz = np.max(skeleton, axis=0) + cfg.bbox_offset

        return {
            'min_xyz': min_xyz,
            'max_xyz': max_xyz
        }

    def load_train_cameras(self):
        cameras = None
        with open(os.path.join(self.dataset_path, 'cameras.pkl'), 'rb') as f: 
            cameras = pickle.load(f)
        return cameras
    
    def load_train_frames(self):
        img_paths = list_files(os.path.join(self.dataset_path, 'images'),
                               exts=['.png'])
        return [split_path(ipath)[1] for ipath in img_paths]
    
    def query_dst_skeleton(self, frame_name):
        return {
            'poses': self.mesh_infos[frame_name]['poses'].astype('float32'), ## (72,)
            'dst_tpose_joints': \
                self.mesh_infos[frame_name]['tpose_joints'].astype('float32'), ## (24, 3)
            'bbox': self.mesh_infos[frame_name]['bbox'].copy(),
            'Rh': self.mesh_infos[frame_name]['Rh'].astype('float32'),
            'Th': self.mesh_infos[frame_name]['Th'].astype('float32'),
            'betas': self.mesh_infos[frame_name]['betas'].astype('float32'),
            'vertices': self.mesh_infos[frame_name]['vertices'].astype('float32')
        }

    def prepare_input(self, frame_name):
        dst_skel_info = self.query_dst_skeleton(frame_name)

        poses = dst_skel_info['poses']
        betas = dst_skel_info['betas']

        Rh = dst_skel_info['Rh']
        Th = dst_skel_info['Th']

        # prepare sp input of param pose
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)

        pxyz = dst_skel_info['vertices'].astype(np.float32)
        wxyz = np.dot(pxyz, R.T).astype(np.float32) + Th

        poses = poses.reshape(-1, 3)
        joints = self.joints
        parents = self.parents

        A, canonical_joints = if_nerf_dutils.get_rigid_transformation(
            poses, joints, parents, return_joints=True)

        posed_joints = np.dot(canonical_joints, R.T) + Th

        nearest_frame_index = 0

        poses = poses.ravel().astype(np.float32)

        return wxyz, pxyz, A, Rh, Th, poses, nearest_frame_index

    def __getitem__(self, index):
        latent_index = index
        frame_index = index * self.frame_interval
        frame_name = self.ims[index]


        # read v_shaped
        tvertices = self.bigpose_vertices.astype(np.float32)
        tbounds = if_nerf_dutils.get_bounds(tvertices)

        wpts, pvertices, A, Rh, Th, poses, nearest_frame_index = self.prepare_input(frame_name)

        pbounds = if_nerf_dutils.get_bounds(pvertices)
        wbounds = if_nerf_dutils.get_bounds(wpts)

        K = self.cameras[frame_name]['intrinsics'].astype(np.float32).copy()
        D = self.cameras[frame_name]['distortions']

        mask_path = os.path.join(self.dataset_path, 
                                'masks', 
                                '{}.png'.format(frame_name))
        msk = (imageio.imread(mask_path)[..., 0] > 0).astype(np.uint8)
        msk = cv2.undistort(msk, K, D)
        border = 5
        kernel = np.ones((border, border), np.uint8)
        msk = cv2.dilate(msk.copy(), kernel)

        msks = [msk]

        K[:2] = K[:2] * cfg.ratio

        img_path = os.path.join(self.image_dir, '{}.png'.format(frame_name))
        img = imageio.imread(img_path).astype(np.float32) / 255.
        H, W = img.shape[:2]
        H, W = int(H * cfg.ratio), int(W * cfg.ratio)
        msks = [
            cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
            for msk in msks
        ]
        msks = np.array(msks)

        E = self.cameras[frame_name]['extrinsics'].astype(np.float32)

        R = E[:3, :3]
        T = E[:3, 3][..., None]

        ray_o, ray_d, near, far, mask_at_box = if_nerf_dutils.get_rays_within_bounds(
            H, W, K, R, T, wbounds)

        ret = {
            'ray_o': ray_o,
            'ray_d': ray_d,
            'near': near,
            'far': far,
            'mask_at_box': mask_at_box
        }

        # blend weight
        meta = {
            'A': A,
            'big_A': self.big_A,
            'poses': poses,
            'weights': self.weights,
            'tvertices': tvertices,
            'pvertices': pvertices,
            'pbounds': pbounds,
            'wbounds': wbounds,
            'tbounds': tbounds
        }
        ret.update(meta)

        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        latent_index = min(latent_index, cfg.num_train_frame - 1)
        meta = {
            'R': R,
            'Th': Th,
            'latent_index': latent_index,
            'frame_index': frame_index,
            'view_index': self.test_view,
            'dataset_path': self.dataset_path
        }
        ret.update(meta)

        Ks = K[None, ...]
        RT = E[None, ...]

        meta = {'msks': msks, 'Ks': Ks, 'RT': RT, 'H': H, 'W': W}
        ret.update(meta)

        vis_map_path = os.path.join(self.dataset_path, 
                                'vis_maps', 
                                '{}.npy'.format(frame_name))

        vis_map = np.load(vis_map_path)

        ret.update({'vis_map': vis_map})

        ret.update(self.ref_infos)

        return ret

    def __len__(self):
        return len(self.ims)
