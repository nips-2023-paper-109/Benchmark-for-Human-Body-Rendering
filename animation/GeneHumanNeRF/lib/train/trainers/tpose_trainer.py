import torch.nn as nn
from lib.config import cfg
import torch
from lib.networks.renderer import make_renderer
from lib.networks.renderer import tpose_renderer
from lib.train import make_optimizer
from lib.utils.if_nerf import if_nerf_net_utils
from . import crit

from lib.third_parties.lpips import LPIPS


def _unpack_imgs(rgbs, patch_masks, bgcolor, targets, div_indices):
    N_patch = len(div_indices) - 1
    assert patch_masks.shape[0] == N_patch
    assert targets.shape[0] == N_patch

    patch_imgs = bgcolor.expand(targets.shape).clone() # (N_patch, H, W, 3)
    for i in range(N_patch):
        patch_imgs[i, patch_masks[i]] = rgbs[div_indices[i]:div_indices[i+1]]

    return patch_imgs


def scale_for_lpips(image_tensor):
    return image_tensor * 2. - 1.


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net
        self.renderer = make_renderer(cfg, self.net)

        self.bw_crit = torch.nn.functional.smooth_l1_loss
        self.img2mse = lambda x, y: torch.mean((x - y)**2)
        self.lpips = LPIPS(net='vgg')
        set_requires_grad(self.lpips, requires_grad=False)
        self.lpips = self.lpips.cuda()

    def forward(self, batch):
        ret = self.renderer.render(batch)
        scalar_stats = {}
        loss = 0

        ### KK ###
        ## anisdf_pdf
        ## ret.keys() = dict_keys(['raw', 'sdf', 'gradients', 'observed_gradients', 
        #                          'resd', 'rgb_map', 'acc_map', 'depth_map', 'msk_sdf', 'msk_label'])

        if 'resd' in ret:
            offset_loss = torch.norm(ret['resd'], dim=2).mean()
            scalar_stats.update({'offset_loss': offset_loss})
            loss += 0.01 * offset_loss

        # if 'gradients' in ret:
        #     gradients = ret['gradients']
        #     grad_loss = (torch.norm(gradients, dim=2) - 1.0)**2
        #     grad_loss = grad_loss.mean()
        #     scalar_stats.update({'grad_loss': grad_loss})
        #     loss += 0.01 * grad_loss

        if 'observed_gradients' in ret:
            ogradients = ret['observed_gradients']
            ograd_loss = (torch.norm(ogradients, dim=2) - 1.0)**2
            ograd_loss = ograd_loss.mean()
            scalar_stats.update({'ograd_loss': ograd_loss})
            loss += 0.01 * ograd_loss

        if 'pred_pbw' in ret:
            bw_loss = (ret['pred_pbw'] - ret['smpl_tbw']).pow(2).mean()
            scalar_stats.update({'tbw_loss': bw_loss})
            loss += bw_loss

        if 'pbw' in ret:
            bw_loss = self.bw_crit(ret['pbw'], ret['tbw'])
            scalar_stats.update({'bw_loss': bw_loss})
            loss += bw_loss

        # if 'msk_sdf' in ret:
        #     mask_loss = crit.sdf_mask_crit(ret, batch)
        #     scalar_stats.update({'mask_loss': mask_loss})
        #     loss += mask_loss

        patch_loss_weight = 1.0
        rgb_loss_weight = 1.0

        if patch_loss_weight > 0 and 'target_patches' in batch.keys():
            bgcolor = torch.Tensor([0.0, 0.0, 0.0]).float().cuda()
            rgb = _unpack_imgs(ret['rgb_map'][0], batch['patch_masks'][0], bgcolor,
                                     batch['target_patches'][0], batch['patch_div_indices'][0])
            target = batch['target_patches'][0]
            
            lpips_loss = self.lpips(scale_for_lpips(rgb.permute(0, 3, 1, 2)), 
                                    scale_for_lpips(target.permute(0, 3, 1, 2)))

            scalar_stats.update({'lpips_loss': torch.mean(lpips_loss)})
            loss += patch_loss_weight * torch.mean(lpips_loss)

            rgb_loss_weight= 0.2

        mask = batch['mask_at_box']
        img_loss = self.img2mse(ret['rgb_map'][mask], batch['rgb'][mask])
        scalar_stats.update({'img_loss': img_loss})
        loss += rgb_loss_weight * img_loss

        if 'rgb0' in ret:
            img_loss0 = self.img2mse(ret['rgb0'], batch['rgb'])
            scalar_stats.update({'img_loss0': img_loss0})
            loss += img_loss0

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return ret, loss, scalar_stats, image_stats
