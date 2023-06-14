import os
import numpy as np

import torch
import torch.nn.functional as F
from lib.config import cfg
from .nerf_net_utils import *
from .. import embedder
from lib.utils.blend_utils import *

from lib.utils import sample_utils

class Renderer:
    def __init__(self, net):
        self.net = net

    def _raw2outputs(self, raw, z_vals, rays_d, white_bkgd=False):
        def _raw2alpha(raw, dists, act_fn=F.relu):
            return 1.0 - torch.exp(-act_fn(raw)*dists)

        dists = z_vals[...,1:] - z_vals[...,:-1]

        infinity_dists = torch.Tensor([1e10])
        infinity_dists = infinity_dists.expand(dists[...,:1].shape).to(dists)
        dists = torch.cat([dists, infinity_dists], dim=-1) 
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

        rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
        alpha = _raw2alpha(raw[...,3], dists)  # [N_rays, N_samples]
        # alpha = alpha * raw_mask[:, :, 0]

        weights = alpha * torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1)).to(alpha), 
                       1.-alpha + 1e-10], dim=-1), dim=-1)[:, :-1]
        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        acc_map = torch.sum(weights, -1)

        # rgb_map = rgb_map + (1.-acc_map[...,None]) * bgcolor[None, :]/255.

        if white_bkgd:
            rgb_map = rgb_map + (1. - acc_map[..., None])

        return rgb_map, acc_map, weights, depth_map

    def get_wsampling_points(self, ray_o, ray_d, near, far):
        # calculate the steps for each ray
        t_vals = torch.linspace(0., 1., steps=cfg.N_samples).to(near)
        z_vals = near[..., None] * (1. - t_vals) + far[..., None] * t_vals

        if cfg.perturb > 0. and self.net.training:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(upper)
            z_vals = lower + (upper - lower) * t_rand

        pts = ray_o[:, :, None] + ray_d[:, :, None] * z_vals[..., None]

        return pts, z_vals

    def get_density_color(self, wpts, viewdir, z_vals, raw_decoder):
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

        ret = raw_decoder(wpts, viewdir, dists)

        return ret

    def get_pixel_value(self, ray_o, ray_d, near, far, occ, batch):
        n_batch = ray_o.shape[0]

        # sampling points for nerf training
        wpts, z_vals = self.get_wsampling_points(ray_o, ray_d, near, far)
        n_batch, n_pixel, n_sample = wpts.shape[:3]

        # viewing direction, ray_d has been normalized in the dataset
        viewdir = ray_d

        raw_decoder = lambda wpts_val, viewdir_val, dists_val: self.net(
            wpts_val, viewdir_val, dists_val, batch)

        # compute the color and density
        ret = self.get_density_color(wpts, viewdir, z_vals, raw_decoder)

        # reshape to [num_rays, num_samples along ray, 4]
        n_batch, n_pixel, n_sample = z_vals.shape
        raw = ret['raw'].reshape(-1, n_sample, 4)

        # normal = ret['normal'].reshape(-1, n_sample, 3)

        z_vals = z_vals.view(-1, n_sample)
        ray_d = ray_d.view(-1, 3)
        # rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
        #     raw, z_vals, cfg.white_bkgd)

        rgb_map, acc_map, _, depth_map = self._raw2outputs(raw, z_vals, ray_d, white_bkgd=False)

        rgb_map = rgb_map.view(n_batch, n_pixel, -1)
        acc_map = acc_map.view(n_batch, n_pixel)
        depth_map = depth_map.view(n_batch, n_pixel)

        # normal_map = torch.sum(weights[..., None] * normal, -2).view(n_batch, n_pixel, -1)

        ret.update({
            'rgb_map': rgb_map,
            'acc_map': acc_map,
            'depth_map': depth_map,
            'raw': raw.view(n_batch, -1, 4),
            # 'normal_map': normal_map
        })

        if 'pbw' in ret:
            pbw = ret['pbw'].view(n_batch, -1, 24)
            ret.update({'pbw': pbw})

        if 'tbw' in ret:
            tbw = ret['tbw'].view(n_batch, -1, 24)
            ret.update({'tbw': tbw})

        if 'sdf' in ret:
            # get pixels that outside the mask or no ray-geometry intersection
            sdf = ret['sdf'].view(n_batch, n_pixel, n_sample)
            min_sdf = sdf.min(dim=2)[0]

            free_sdf = min_sdf[occ == 0]

            free_label = torch.zeros_like(free_sdf)

            with torch.no_grad():
                intersection_mask, _ = get_intersection_mask(sdf, z_vals)
            ind = (intersection_mask == False) * (occ == 1)
            sdf = min_sdf[ind]
            label = torch.ones_like(sdf)

            sdf = torch.cat([sdf, free_sdf])
            label = torch.cat([label, free_label])

            if len(sdf) > 0:
                ret.update({
                    'msk_sdf': sdf.view(n_batch, -1),
                    'msk_label': label.view(n_batch, -1)
                })

        if not rgb_map.requires_grad:
            ret = {k: ret[k].detach().cpu() for k in ret.keys()}

        return ret

    def render(self, batch):
        ray_o = batch['ray_o']
        ray_d = batch['ray_d']
        near = batch['near']
        far = batch['far']
        occ = batch['occupancy']
        sh = ray_o.shape

        # volume rendering for each pixel
        n_batch, n_pixel = ray_o.shape[:2]
        chunk = 2400

        H, W = batch['H'], batch['W']
        ref_imgs = F.interpolate(batch['ref_imgs'][0].permute(0, 3, 1, 2), size=(H, W), mode='area')
        ref_msks = F.interpolate(batch['ref_msks'][0][:, None], size=(H, W), mode='nearest')
        
        img_feats = self.net.encoder(ref_imgs)

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
            occ_chunk = occ[:, i:i + chunk]
            pixel_value = self.get_pixel_value(ray_o_chunk, ray_d_chunk,
                                               near_chunk, far_chunk,
                                               occ_chunk, batch)
            ret_list.append(pixel_value)

        keys = ret_list[0].keys()
        ret = {k: torch.cat([r[k] for r in ret_list], dim=1) for k in keys}

        return ret
