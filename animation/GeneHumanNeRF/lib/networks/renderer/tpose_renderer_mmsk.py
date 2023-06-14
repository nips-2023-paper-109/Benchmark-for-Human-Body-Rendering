import os
import numpy as np

import torch
import torch.nn.functional as F
from lib.config import cfg
from .nerf_net_utils import *
from .. import embedder
from lib.utils.blend_utils import *
from . import tpose_renderer

from lib.utils import sample_utils

from spconv.pytorch.conv import SparseConvTensor

class Renderer(tpose_renderer.Renderer):
    def __init__(self, net):
        super(Renderer, self).__init__(net)

        self.counter = 0

    def prepare_inside_pts(self, pts, batch):
        if 'Ks' not in batch:
            __import__('ipdb').set_trace()
            return raw

        sh = pts.shape
        pts = pts.view(sh[0], -1, sh[3])

        insides = []
        for nv in range(batch['Ks'].size(1)):
            # project pts to image space
            R = batch['RT'][:, nv, :3, :3]
            T = batch['RT'][:, nv, :3, 3]
            pts_ = torch.matmul(pts, R.transpose(2, 1)) + T[:, None]
            pts_ = torch.matmul(pts_, batch['Ks'][:, nv].transpose(2, 1))
            pts2d = pts_[..., :2] / pts_[..., 2:]

            # ensure that pts2d is inside the image
            pts2d = pts2d.round().long()
            H, W = batch['H'].item(), batch['W'].item()
            pts2d[..., 0] = torch.clamp(pts2d[..., 0], 0, W - 1)
            pts2d[..., 1] = torch.clamp(pts2d[..., 1], 0, H - 1)

            # remove the points outside the mask
            pts2d = pts2d[0]
            msk = batch['msks'][0, nv]
            inside = msk[pts2d[:, 1], pts2d[:, 0]][None].bool()
            insides.append(inside)

        inside = insides[0]
        for i in range(1, len(insides)):
            inside = inside * insides[i]

        return inside

    def get_vis_map(self, pvertices, faces, Rh, Th, cam_R, cam_T):
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

        # pts_ = torch.matmul(c_pvertices, batch['Ks'][0].transpose(2, 1))
        # pts2d = pts_[..., :2] / pts_[..., 2:]

        # img = np.zeros([batch['H'], batch['W'], 3])
        # img[pts2d[0][:, 1].cpu().numpy().astype(np.int16), pts2d[0][:, 0].cpu().numpy().astype(np.int16)] = 255

        # import cv2
        # cv2.imwrite('vis_full.png', img)

        ### put the vertices with the (-1, 1)
        c_bounds, _ = torch.abs(c_pvertices[0]).max(0)
        c_pvertices = c_pvertices[0]/(c_bounds[None, ...]+1e-6)

        (xy, z) = c_pvertices.split([2, 1], dim=1)
        visibility = sample_utils.get_visibility(xy, z, faces[:, [0, 2, 1]]).flatten()

        return visibility == 0

    def get_density_color(self, wpts, viewdir, z_vals, inside, raw_decoder):
        """
        wpts: n_batch, n_pixel, n_sample, 3
        viewdir: n_batch, n_pixel, 3
        z_vals: n_batch, n_pixel, n_sample
        """
        n_batch, n_pixel, n_sample = wpts.shape[:3]
        wpts = wpts.view(n_batch * n_pixel * n_sample, -1)
        viewdir = viewdir[:, :, None].repeat(1, 1, n_sample, 1).contiguous()
        viewdir = viewdir.view(n_batch * n_pixel * n_sample, -1)

        # calculate dists for the opacity computation
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, dists[..., -1:]], dim=2)
        dists = dists.view(n_batch * n_pixel * n_sample)

        full_raw = torch.zeros([n_batch * n_pixel * n_sample, 4]).to(wpts)
        # full_normal = torch.zeros([n_batch * n_pixel * n_sample, 3]).to(wpts)
        
        if inside.sum() == 0:
            ret = {'raw': full_raw}
            # ret = {'raw': full_raw, 'normal': full_normal}
            return ret

        inside = inside.view(n_batch * n_pixel * n_sample)

        ### KK ###
        ## 不做inside检测了
        # wpts = wpts[inside]
        # viewdir = viewdir[inside]
        # dists = dists[inside]
        # ret = raw_decoder(wpts, viewdir, dists)

        # full_raw[inside] = ret['raw']
        # full_normal[inside] = ret['normal']

        ret = raw_decoder(wpts, viewdir, dists)
        full_raw = ret['raw']
        # full_normal = ret['normal']
        ##########

        # ret = {'raw': full_raw, 'normal': full_normal}

        ret = {'raw': full_raw}
        
        return ret

    def get_pixel_value(self, ray_o, ray_d, near, far, batch):
        n_batch = ray_o.shape[0]

        # sampling points for nerf training
        wpts, z_vals = self.get_wsampling_points(ray_o, ray_d, near, far)

        inside = self.prepare_inside_pts(wpts, batch)

        # viewing direction, ray_d has been normalized in the dataset
        viewdir = ray_d

        raw_decoder = lambda wpts_val, viewdir_val, dists_val: self.net(
            wpts_val, viewdir_val, dists_val, batch)

        # compute the color and density
        ret = self.get_density_color(wpts, viewdir, z_vals, inside,
                                     raw_decoder)

        # reshape to [num_rays, num_samples along ray, 4]
        n_batch, n_pixel, n_sample = z_vals.shape


        raw = ret['raw'].reshape(-1, n_sample, 4)
        # normal = ret['normal'].reshape(-1, n_sample, 3)

        z_vals = z_vals.view(-1, n_sample)
        ray_d = ray_d.view(-1, 3)
        # rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
        #     raw, z_vals, cfg.white_bkgd)

        rgb_map, acc_map, _, depth_map = self._raw2outputs(raw, z_vals, ray_d, white_bkgd=False)

        # normal_map = torch.sum(weights[..., None] * normal, -2).view(n_batch, n_pixel, -1)

        rgb_map = rgb_map.view(n_batch, n_pixel, -1)
        acc_map = acc_map.view(n_batch, n_pixel)
        depth_map = depth_map.view(n_batch, n_pixel)        

        ret = {
            'rgb_map': rgb_map.detach().cpu(),
            'acc_map': acc_map.detach().cpu(),
            'depth_map': depth_map.detach().cpu(),
            # 'normal_map': normal_map.detach().cpu()
        }

        return ret
    
    def get_vol_feats(self, img_feats, batch):
        VOXEL_SIZE = 0.01

        xyz_min = torch.Tensor([-1.0, -1.2, -0.3]).to(img_feats)
        xyz_max = torch.Tensor([1.0, 0.8, 0.3]).to(img_feats)

        n_grid = (xyz_max-xyz_min)/VOXEL_SIZE

        tvertices = batch['tvertices'][0]
        # ref_vis_maps = batch['ref_vis_maps'][0]

        ref_pvertices = batch['ref_pvertices'][0]
        B, C, H, W = img_feats.shape

        features_per_vertice = torch.zeros([B, ref_pvertices[0].shape[0], C], dtype=torch.float, device=img_feats.device)

        for i in range(img_feats.shape[0]):
            ## no vis map
            # ref_vis_map = ref_vis_maps[i]
            # vis_pvertices = ref_pvertices[i][ref_vis_map.bool()]
            vis_pvertices = ref_pvertices[i]

            pose_pts = pose_points_to_world_points(vis_pvertices[None], batch['ref_Rhs'][0][i][None],
                                                batch['ref_Ths'][0][i][None])

            R = batch['ref_RTs'][:, i, :3, :3]
            T = batch['ref_RTs'][:, i, :3, 3]

            pose_pts = torch.matmul(pose_pts, R.transpose(2, 1)) + T[:, None]

            pts_ = torch.matmul(pose_pts, batch['ref_Ks'][0][i][None].transpose(2, 1))
            pts2d = pts_[..., :2] / pts_[..., 2:]

            # import cv2
            # img = np.zeros([batch['W'], batch['H'], 3])
            # img[pts2d[0][:, 1].cpu().numpy().astype(np.int16), pts2d[0][:, 0].cpu().numpy().astype(np.int16)] = 255
            # cv2.imwrite('img_proj_{}.png'.format(i), img)

            ## 需要把grid normalize到[-1, 1]之间
            grid = pts2d[0].float()
            grid[:, 0] = 2 * grid[:, 0] / (W * 4) - 1
            grid[:, 1] = 2 * grid[:, 1] / (H * 4) - 1
            
            pixel_features = F.grid_sample(img_feats[i][None, ...], 
                                     grid[None, None, ...], 
                                     align_corners=True, 
                                     mode='bilinear').squeeze().permute(1, 0)

            ## no vis map
            # features_per_vertice[i, ref_vis_map.bool()] = pixel_features
            features_per_vertice[i] = pixel_features

        ## no vis map
        # features_per_vertice = features_per_vertice.sum(0)/(ref_vis_maps[..., None].sum(0)+1e-6)
        features_per_vertice = features_per_vertice.mean(0)

        tvertices = ((tvertices-xyz_min)/(xyz_max-xyz_min) * n_grid).long()

        coord = torch.cat([torch.zeros(tvertices.shape[0], device=tvertices.device)[:, None], tvertices], dim=1)
        code = features_per_vertice

        xyzc = SparseConvTensor(code, coord.int(), n_grid, 1)

        vol_feature = self.net.spconv(xyzc)

        return vol_feature, xyz_min, xyz_max
        
    def render(self, batch):
        ray_o = batch['ray_o']
        ray_d = batch['ray_d']
        near = batch['near']
        far = batch['far']
        sh = ray_o.shape
        # volume rendering for each pixel
        n_batch, n_pixel = ray_o.shape[:2]
        chunk = 2048

        H, W = batch['H'], batch['W']
        ref_imgs = F.interpolate(batch['ref_imgs'][0].permute(0, 3, 1, 2), size=(H, W), mode='area')
        ref_msks = F.interpolate(batch['ref_msks'][0][:, None], size=(H, W), mode='nearest')
        
        img_feats = self.net.encoder(ref_imgs)
        
        '''
        vis_map_path = os.path.join(batch['dataset_path'][0], 'ref_vis_maps.npy')
        if os.path.exists(vis_map_path):
            ref_vis_maps = np.load(vis_map_path)
            ref_vis_maps = torch.from_numpy(ref_vis_maps).to(img_feats)
        else:
            ref_vis_maps = []
            for i in range(batch['ref_imgs'][0].shape[0]):
                vis_map_args = {
                    'pvertices': batch['ref_pvertices'][0][i][None],
                    'faces': batch['faces'].long()[0],
                    'Rh': batch['ref_Rhs'][0][i],
                    'Th': batch['ref_Ths'][0][i][None],
                    'cam_R': batch['ref_RTs'][0][i][:3, :3],
                    'cam_T': batch['ref_RTs'][0][i][:3, 3][None]
                }
                vis_map = self.get_vis_map(**vis_map_args)
                ref_vis_maps.append(vis_map)
            ref_vis_maps = torch.stack(ref_vis_maps, dim=0).to(img_feats)
            np.save(vis_map_path, ref_vis_maps.cpu().numpy())
        '''

        # vol_feats, xyz_min, xyz_max = self.get_vol_feats(img_feats, batch)

        vol_feats, xyz_min, xyz_max = self.net.get_vol_feats(img_feats, batch)

        batch['img_feats'] = img_feats
        batch['ref_imgs'] =  ref_imgs
        batch['ref_msks'] = ref_msks
        batch['vol_feats'] = vol_feats
        batch['xyz_min'] = xyz_min
        batch['xyz_max'] = xyz_max
        # batch['ref_vis_maps'] = ref_vis_maps

        ret_list = []
        for i in range(0, n_pixel, chunk):
            ray_o_chunk = ray_o[:, i:i + chunk]
            ray_d_chunk = ray_d[:, i:i + chunk]
            near_chunk = near[:, i:i + chunk]
            far_chunk = far[:, i:i + chunk]
            pixel_value = self.get_pixel_value(ray_o_chunk, ray_d_chunk,
                                               near_chunk, far_chunk, batch)
            ret_list.append(pixel_value)

        keys = ret_list[0].keys()
        ret = {k: torch.cat([r[k] for r in ret_list], dim=1) for k in keys}

        return ret
