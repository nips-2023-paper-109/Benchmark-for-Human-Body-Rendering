import torch.nn as nn
#import spconv
import torch.nn.functional as F
import torch
from lib.config import cfg
from lib.utils.blend_utils import *
from .. import embedder
from lib.utils import net_utils
import os
from lib.utils import sample_utils

import spconv.pytorch as spconv
from spconv.pytorch.conv import SparseConvTensor

import time

def fused_mean_variance(x):
    mean = x.mean(-2).unsqueeze(-2)
    var = torch.mean((x - mean)**2, dim=-2, keepdim=True)
    return mean, var

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        n_smpl=6890
        code_dim=32
        attn_n_heads=4
        in_feat_ch=32

        self.c = nn.Embedding(n_smpl, code_dim)

        self.xyzc_attn = MultiHeadAttention(attn_n_heads, code_dim, code_dim//attn_n_heads,
                                            code_dim//attn_n_heads, kv_dim=in_feat_ch, sum=False)

        self.encoder = FeatureNet()

        self.spconv = SparseConvNet()

        self.color_density_decoder = feat_vol_decoder(input_ch_feat=352, input_ch_pixel_feat=32+3)

        self.actvn = nn.ReLU()

        if cfg.get('init_sdf', False):
            init_path = os.path.join('data/trained_model', cfg.task,
                                     cfg.init_sdf)
            net_utils.load_network(self,
                                   init_path,
                                   only=None)


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
        # features_per_vertice = features_per_vertice.mean(0)
        latent = self.c(torch.arange(0, 6890).cuda()).unsqueeze(1)
        smpl_feat_sampled = features_per_vertice.permute(1,0,2)

        xyzc_fuse = self.xyzc_attn(
            latent, smpl_feat_sampled, smpl_feat_sampled
        )[0].squeeze(1)

        tvertices = ((tvertices-xyz_min)/(xyz_max-xyz_min) * n_grid).long()

        coord = torch.cat([torch.zeros(tvertices.shape[0], device=tvertices.device)[:, None], tvertices], dim=1)
        # code = features_per_vertice

        code = xyzc_fuse

        xyzc = SparseConvTensor(code, coord.int(), n_grid, 1)

        vol_feature = self.spconv(xyzc)

        return vol_feature, xyz_min, xyz_max


    def pose_points_to_tpose_points(self, pose_pts, pose_dirs, batch):
        """
        pose_pts: n_batch, n_point, 3
        """
        # initial blend weights of points at i
        pbw, _ = sample_utils.sample_blend_closest_points(pose_pts, batch['pvertices'], batch['weights'])
        pbw = pbw.permute(0, 2, 1)


        # transform points from i to i_0
        init_tpose = pose_points_to_tpose_points(pose_pts, pbw,
                                                 batch['A'])
        init_bigpose = tpose_points_to_pose_points(init_tpose, pbw,
                                                   batch['big_A'])

        ### KK ###
        ## 先不考虑residual deformation
        # resd = self.calculate_residual_deformation(init_bigpose, batch)
        # tpose = init_bigpose + resd
        resd = None
        tpose = init_bigpose
        ###########

        if cfg.tpose_viewdir and pose_dirs is not None:
            init_tdirs = pose_dirs_to_tpose_dirs(pose_dirs, pbw,
                                                 batch['A'])
            tpose_dirs = tpose_dirs_to_pose_dirs(init_tdirs, pbw,
                                                 batch['big_A'])
        else:
            tpose_dirs = None

        return tpose, tpose_dirs, init_bigpose, resd

    def tpose_dirs_to_pose_dirs(self, tpose_pts, tpose_dirs, batch):
        ### KK ###
        ## transform normals in tpose to pose space

        # initial blend weights of points at i
        tbw, _ = sample_utils.sample_blend_closest_points(tpose_pts, batch['tvertices'], batch['weights'])
        tbw = tbw.permute(0, 2, 1)

        init_tpose_dirs = pose_dirs_to_tpose_dirs(tpose_dirs, tbw, batch['big_A'])

        pose_dirs = tpose_dirs_to_pose_dirs(init_tpose_dirs, tbw, batch['A'])

        return pose_dirs

    
    def query_ref_color(self, tpts, bw, batch):
        

        ### big_A --> A
        tpose_pts = pose_points_to_tpose_points(tpts, bw, batch['big_A'])

        ## tpose_pts = batch['tvertices']
        ## bw = batch['weights'].permute(0,2,1)

        ref_color = []
        # ref_vis = []
        ref_feats = []

        vis_sim, vis_sim_ind = torch.sort((batch['vis_map'] * batch['ref_vis_maps'][0]).sum(-1), descending=True)

        for i in range(batch['ref_As'][0].shape[0]):
            ### A --> pose
            pose_pts = tpose_points_to_pose_points(tpose_pts, bw, batch['ref_As'][0][i][None])
            
            # visibility, pnorm = sample_utils.sample_blend_closest_points(pose_pts, batch['ref_pvertices'][:, i], batch['ref_vis_maps'][0][i][None, :, None].float().cuda())

            ### canonical --> world
            pose_pts = pose_points_to_world_points(pose_pts, batch['ref_Rhs'][0][i][None],
                                                batch['ref_Ths'][0][i][None])

            R = batch['ref_RTs'][:, i, :3, :3]
            T = batch['ref_RTs'][:, i, :3, 3]

            pose_pts = torch.matmul(pose_pts, R.transpose(2, 1)) + T[:, None]

            pts_ = torch.matmul(pose_pts, batch['ref_Ks'][0][i][None].transpose(2, 1))
            pts2d = pts_[..., :2] / pts_[..., 2:]

            grid = pts2d[0].clone().float()
            grid[:, 0] = 2 * grid[:, 0] / batch['W'] - 1
            grid[:, 1] = 2 * grid[:, 1] / batch['H'] - 1
            
            ref_img = batch['ref_imgs'][i]
            ref_msk = batch['ref_msks'][i]
            ref_img_feats = batch['img_feats'][i]

            ref_msk_img = torch.cat([ref_msk, ref_img], dim=0)

            queried_c = F.grid_sample(ref_msk_img[None, ...], 
                                     grid[None, None, ...], 
                                     align_corners=True, 
                                     mode='bilinear')[0, :, 0, :].permute(1, 0)

            queried_feats = F.grid_sample(ref_img_feats[None, ...], 
                                     grid[None, None, ...], 
                                     align_corners=True, 
                                     mode='bilinear')[0, :, 0, :].permute(1, 0)
                                     
            ref_color.append(queried_c)
            # ref_vis.append(visibility)
            ref_feats.append(queried_feats)

            # import cv2
            # img = np.zeros([batch['W'], batch['H'], 3])
            # img[pts2d[0][:, 1].cpu().numpy().astype(np.int16), pts2d[0][:, 0].cpu().numpy().astype(np.int16)] = 255
            # cv2.imwrite('img_sampled_{}.png'.format(i), img)

        # ref_img = (batch['ref_imgs'][0][i].cpu().numpy() * 255).astype(np.uint8)
        # img = np.zeros_like(ref_img)
        # img[pts2d[0][:, 1].cpu().numpy().astype(np.int16), pts2d[0][:, 0].cpu().numpy().astype(np.int16)] = 255
        # cv2.imwrite('img.png', img)

        N_selected = 3

        ref_color = torch.stack(ref_color, dim=0)[vis_sim_ind[:N_selected]]
        ref_feats = torch.stack(ref_feats, dim=0)[vis_sim_ind[:N_selected]]

        # ref_vis = torch.cat(ref_vis, dim=0)[vis_sim_ind[:N_selected]]

        batch.update({'selected_ref_ind': vis_sim_ind[:N_selected]})
        
        color = torch.cat([ref_color[:, :, 1:], ref_feats], dim=-1)
        msk = ref_color[:, :, 0]

        # vis_thr = 0.2
        ## original: 0.5
        vis_thr = 0.5
        
        # ref_color = (color * (ref_vis>=vis_thr)*(msk[..., None]>0)).sum(0)/(((ref_vis>=vis_thr)*(msk[..., None]>0)).sum(0)+1e-6)
        # ref_color = color.mean(0)

        ref_color = color

        return ref_color

    def interpolate_features(self, grid_coords, feature_volume):
        features = []
        for volume in feature_volume:
            feature = F.grid_sample(volume,
                                    grid_coords,
                                    padding_mode='zeros',
                                    align_corners=True)
            features.append(feature)
        features = torch.cat(features, dim=1)
        features = features.view(features.size(0), -1, features.size(4))
        return features

    def forward(self, wpts, viewdir, dists, batch):
        # transform points from the world space to the pose space
        wpts = wpts[None]

        pose_pts = world_points_to_pose_points(wpts, batch['R'], batch['Th'])
        
        viewdir = viewdir[None]
        pose_dirs = world_dirs_to_pose_dirs(viewdir, batch['R'])

        ### KK ###
        ## batch['pvertices'].shape = torch.Size([1, 6890, 3])
        ## batch['weights'].shape = torch.Size([1, 6890, 24])
        
        with torch.no_grad():
            
            pbw, pnorm = sample_utils.sample_blend_closest_points(pose_pts, batch['pvertices'], batch['weights'])

            pnorm = pnorm[..., 0]
            norm_th = 0.1

            pind = pnorm < norm_th
            
            pind[torch.arange(len(pnorm)), pnorm.argmin(dim=1)] = True
            pose_pts = pose_pts[pind][None]
            viewdir = viewdir[pind][None]
            pose_dirs = pose_dirs[pind][None]

        # transform points from the pose space to the tpose space
        tpose, tpose_dirs, init_bigpose, resd = self.pose_points_to_tpose_points(
            pose_pts, pose_dirs, batch)
        tpose = tpose[0]

        ### KK ###
        ## cfg.tpose_viewdir = True
        if cfg.tpose_viewdir:
            viewdir = tpose_dirs[0]
        else:
            viewdir = viewdir[0]

        grid = 2 * (tpose-batch['xyz_min']) / (batch['xyz_max']-batch['xyz_min']) - 1

        queried_vol_feats = self.interpolate_features(grid[None,None,None,...].float(), batch['vol_feats'])[0].transpose(1, 0)

        rgb_feats = self.query_ref_color(tpose[None], pbw[pind][None].permute(0, 2, 1), batch)

        # ret = self.tpose_human(tpose, viewdir, dists, queried_vol_feats, color_feats, batch)

        ret = self.color_density_decoder(queried_vol_feats, rgb_feats)

        ### KK ###
        ## 替换为直接的颜色
        # ret['raw'][:, :3] = color_feats[:, :3]
        ##########

        ### KK ###
        ## 把invisible points的颜色直接置为0
        ## 这种方式用于移除不必要的点非常有效，但是太暴力了，需要一个更合理的方式填补invisible points
        # ret['raw'][color_feats.sum(-1) == 0, :3] = 0
        ##########
        
        tbounds = batch['tbounds'][0]
        tbounds[0] -= 0.05
        tbounds[1] += 0.05
        inside = tpose > tbounds[:1]
        inside = inside * (tpose < tbounds[1:])
        outside = torch.sum(inside, dim=1) != 3

        ret['raw'][outside] = 0

        n_batch, n_point = wpts.shape[:2]
        raw = torch.zeros([n_batch, n_point, 4]).to(wpts)
        raw[pind] = ret['raw']
        # sdf = 10 * torch.ones([n_batch, n_point, 1]).to(wpts)
        # sdf[pind] = ret['sdf']
        # ret.update({'raw': raw, 'sdf': sdf, 'resd': resd})
        # ret.update({'raw': raw, 'sdf': sdf})

        ret.update({'raw': raw})

        # ret.update({'gradients': ret['gradients'][None]})

        ### KK ###
        # tpose_grads = ret['gradients']
        # posed_grads = self.tpose_dirs_to_pose_dirs(tpose[None], tpose_grads, batch)
        # normal = torch.zeros([n_batch, n_point, 3]).to(wpts)
        # normal[pind] = posed_grads
        # ret.update({'normal': normal})
        ##########
        
        return ret


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)

class feat_vol_decoder(nn.Module):
    def __init__(self, D=8, W=256, input_ch_feat=352, output_ch=4, input_ch_pixel_feat=32):
        """
        """
        super(feat_vol_decoder, self).__init__()
        self.D = D
        self.W = W
        self.input_ch_feat, self.input_ch_pixel_feat = input_ch_feat, input_ch_pixel_feat

        self.out_geometry_fc0 = nn.Sequential(nn.Linear(input_ch_feat, 64),
                                             nn.ELU(inplace=True),)

        self.out_geometry_fc1 = nn.Sequential(
                                             nn.Linear(64+(32+3)*2, 64),
                                             nn.ELU(inplace=True),
                                             nn.Linear(64, 32),
                                             nn.ELU(inplace=True),
                                             nn.Linear(32, 16),
                                             nn.ELU(inplace=True),
                                             nn.Linear(16, 1),
                                             nn.ReLU())

        self.base_fc = nn.Sequential(
                                     nn.Linear((32+3)*3, 64),
                                     nn.ELU(inplace=True),
                                     nn.Linear(64, 32),
                                     nn.ELU(inplace=True)
                                    )
        self.vis_fc = nn.Sequential(nn.Linear(32, 32),
                                    nn.ELU(inplace=True),
                                    nn.Linear(32, 32),
                                    nn.ELU(inplace=True)
                                    )
        self.rgb_fc = nn.Sequential(nn.Linear(96, 32),
                                    nn.ELU(inplace=True),
                                    nn.Linear(32, 16),
                                    nn.ELU(inplace=True),
                                    nn.Linear(16, 3))

        self.apply(weights_init)


    def forward(self, vol_feats, rgb_feat):
        ## vol_feats.shape = torch.Size([23489, 352])
        ## rgb_feat.shape = torch.Size([3, 23489, 35])

        num_views = rgb_feat.shape[0]

        rgb_feat = rgb_feat.permute(1,0,2)

        sigma_feat = self.out_geometry_fc0(vol_feats)

        mean, var = fused_mean_variance(rgb_feat)

        globalfeat = torch.cat([mean, var], dim=-1)

        sigma = self.out_geometry_fc1(torch.cat([sigma_feat, globalfeat[:, 0, :]], dim=-1))

        x = torch.cat([globalfeat.expand(-1, num_views, -1), rgb_feat], dim=-1)

        x = self.base_fc(x)
        x_vis = self.vis_fc(x * 1.0 / num_views)
        x = x + x_vis

        rgb_out = self.rgb_fc(x.flatten(1, 2))

        ret = {'raw': torch.cat([rgb_out, sigma], dim=1)}

        return ret


InPlaceABN=None

class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act=InPlaceABN):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        # self.bn = norm_act(out_channels)
        self.bn = nn.ReLU()


    def forward(self, x):
        return self.bn(self.conv(x))


class FeatureNet(nn.Module):
    """
    output 3 levels of features using a FPN structure
    """
    def __init__(self, norm_act=InPlaceABN):
        super(FeatureNet, self).__init__()

        self.conv0 = nn.Sequential(
                        ConvBnReLU(3, 8, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(8, 8, 3, 1, 1, norm_act=norm_act))

        self.conv1 = nn.Sequential(
                        ConvBnReLU(8, 16, 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act))

        self.conv2 = nn.Sequential(
                        ConvBnReLU(16, 32, 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act))

        self.toplayer = nn.Conv2d(32, 32, 1)

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.conv0(x) # (B, 8, H, W)
        x = self.conv1(x) # (B, 16, H//2, W//2)
        x = self.conv2(x) # (B, 32, H//4, W//4)
        x = self.toplayer(x) # (B, 32, H//4, W//4)

        return x


class SparseConvNet(nn.Module):
    def __init__(self):
        super(SparseConvNet, self).__init__()

        self.conv0 = double_conv(32, 32, 'subm0')
        self.down0 = stride_conv(32, 32, 'down0')

        self.conv1 = double_conv(32, 32, 'subm1')
        self.down1 = stride_conv(32, 64, 'down1')

        self.conv2 = triple_conv(64, 64, 'subm2')
        self.down2 = stride_conv(64, 128, 'down2')

        self.conv3 = triple_conv(128, 128, 'subm3')
        self.down3 = stride_conv(128, 128, 'down3')

        self.conv4 = triple_conv(128, 128, 'subm4')

    def forward(self, x):
        net = self.conv0(x)
        net = self.down0(net)

        net = self.conv1(net)
        net1 = net.dense()
        net = self.down1(net)

        net = self.conv2(net)
        net2 = net.dense()
        net = self.down2(net)

        net = self.conv3(net)
        net3 = net.dense()
        net = self.down3(net)

        net = self.conv4(net)
        net4 = net.dense()

        volumes = [net1, net2, net3, net4]

        return volumes


def single_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          1,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def double_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def triple_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def stride_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SparseConv3d(in_channels,
                            out_channels,
                            3,
                            2,
                            padding=1,
                            bias=False,
                            indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01), nn.ReLU())


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, kv_dim=None, sum=True):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.sum_flag = sum

        if kv_dim is None:
            kv_dim = d_model
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(kv_dim, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(kv_dim, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k**0.5)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, qv_sz_b, len_k, len_v = (
            q.size(0),
            q.size(1),
            k.size(0),
            k.size(1),
            v.size(1),
        )

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(qv_sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(qv_sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        # q = self.dropout(self.fc(q))
        q = self.fc(q)

        if self.sum_flag is True:
            q += residual
            q = self.layer_norm(q)

        return q, attn