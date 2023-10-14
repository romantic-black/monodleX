import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.helpers.decode_helper import _transpose_and_gather_feat
from lib.losses.focal_loss import focal_loss_cornernet
from lib.losses.uncertainty_loss import laplacian_aleatoric_uncertainty_loss
from lib.losses.dim_aware_loss import dim_aware_l1_loss

eps = 1e-6


def compute_centernet3d_loss(input, target):
    stats_dict = {}

    edge_fusion = False
    if 'edge_len' in target.keys():
        edge_fusion = True

    seg_loss = compute_segmentation_loss(input, target)
    # offset2d_loss = compute_offset2d_loss(input, target, edge_fusion=edge_fusion)
    # size2d_loss = compute_size2d_loss(input, target)
    # offset3d_loss = compute_offset3d_loss(input, target, edge_fusion=edge_fusion)
    # depth_loss = compute_depth_loss(input, target)
    size3d_loss = compute_size3d_loss(input, target)
    heading_loss = compute_heading_loss(input, target)
    location_loss_0 = compute_location_loss(input, target, 0)
    location_loss_1 = compute_location_loss(input, target, 1)
    margin_loss = compute_margin_loss(input, target)

    # statistics
    stats_dict['seg'] = seg_loss.item()
    # stats_dict['offset2d'] = offset2d_loss.item()
    # stats_dict['size2d'] = size2d_loss.item()
    # stats_dict['offset3d'] = offset3d_loss.item()
    # stats_dict['depth'] = depth_loss.item()
    stats_dict['size3d'] = size3d_loss.item()
    stats_dict['heading'] = heading_loss.item()
    stats_dict['location_0'] = location_loss_0.item()
    stats_dict['location_1'] = location_loss_1.item()
    stats_dict['margin_loss'] = margin_loss.item()
    # seg_loss > depth_loss > size2d_loss > heading_loss > size3d_loss
    total_loss = seg_loss + size3d_loss + heading_loss + location_loss_0 + location_loss_1 + margin_loss
    return total_loss, stats_dict


def compute_segmentation_loss(input, target):
    input['heatmap'] = torch.clamp(input['heatmap'].sigmoid_(), min=1e-4, max=1 - 1e-4)
    loss = focal_loss_cornernet(input['heatmap'], target['heatmap'])
    return loss


def compute_size2d_loss(input, target):
    # compute size2d loss
    size2d_input = extract_input_from_tensor(input['size_2d'], target['indices'], target['mask_2d'])
    size2d_target = extract_target_from_tensor(target['size_2d'], target['mask_2d'])
    size2d_loss = F.l1_loss(size2d_input, size2d_target, reduction='mean')
    if torch.any(torch.isnan(size2d_loss)):
        size2d_loss = torch.tensor([0.0]).to(size2d_input.device)
    return size2d_loss


def compute_offset2d_loss(input, target, edge_fusion=False):
    # compute offset2d loss
    offset2d_input = extract_input_from_tensor(input['offset_2d'], target['indices'], target['mask_2d'])
    offset2d_target = extract_target_from_tensor(target['offset_2d'], target['mask_2d'])
    if edge_fusion:  # False
        trunc_mask = extract_target_from_tensor(target['trunc_mask'], target['mask_2d']).bool()
        offset2d_loss = F.l1_loss(offset2d_input, offset2d_target, reduction='none').sum(dim=1)
        # use different loss functions for inside and outside objects
        trunc_offset_loss = torch.log(1 + offset2d_loss[trunc_mask]).sum() / torch.clamp(trunc_mask.sum() + eps, min=1)
        offset2d_loss = offset2d_loss[~trunc_mask].mean()
        return trunc_offset_loss + offset2d_loss
    elif (target['mask_2d'].sum() > 0):  # True
        offset2d_loss = F.l1_loss(offset2d_input, offset2d_target, reduction='mean')
        return offset2d_loss
    else:
        offset2d_loss = torch.tensor([0.0]).to(offset2d_input.device)
        return offset2d_loss


def compute_location_loss(input, target, index):
    offset_3d = extract_input_from_tensor(input['offset_3d'], target['indices'], target['mask_3d'])
    offset_3d = offset_3d[:, 0:7] if index == 0 else offset_3d[:, 7:14]

    offset_3d, offset_center, log_variance, depth = \
        offset_3d[:, 0:2], offset_3d[:, 2:5], offset_3d[:, 5:6], offset_3d[:, 6:7]
    depth = 1. / (depth.sigmoid() + 1e-6) - 1.
    location_target = extract_target_from_tensor(target['location'], target['mask_3d'])
    calib = extract_target_from_tensor(target['cu_cv_fu_fv_tx_ty'], target['mask_3d'])
    indices = extract_target_from_tensor(target['indices'], target['mask_3d'])
    ratio = extract_target_from_tensor(target['ratio'], target['mask_3d'])
    u, v = (indices % 320).unsqueeze(1), (indices // 320).unsqueeze(1)
    g_points = (torch.cat((u, v), dim=-1) + offset_3d) * ratio
    proj = img_to_rect(g_points, depth, calib)
    location = proj + offset_center
    if target['mask_3d'].sum() > 0:
        location_loss = laplacian_aleatoric_uncertainty_loss(location, location_target,
                                                             log_variance, reduction='mean')
    else:
        location_loss = torch.tensor([0.0]).to(g_points.device)
    return location_loss


def compute_margin_loss(input, target):
    offset_3d = extract_input_from_tensor(input['offset_3d'], target['indices'], target['mask_3d'])
    offset_3d_0, offset_3d_1 = offset_3d[:, 0:2], offset_3d[:, 7:9]
    size_2d = extract_target_from_tensor(target['size_2d'], target['mask_3d'])
    threshold = torch.norm(size_2d, dim=1) / 4 / 2      # 4 for down sample, 2 for g point num
    if target['mask_3d'].sum() > 0:
        margin_loss = margin_distanse_loss(offset_3d_0, offset_3d_1,
                                           threshold, reduction='mean')
    else:
        margin_loss = torch.tensor([0.0]).to(offset_3d.device)
    return margin_loss


def compute_depth_loss(input, target):
    # depth: [4,2,96,328], indices: [4, 50], mask_3d: [4, 50]
    depth_input = extract_input_from_tensor(input['depth'], target['indices'], target['mask_3d'])
    depth_input, depth_log_variance = depth_input[:, 0:1], depth_input[:, 1:2]
    # 1e-6 保证输出接近 0 时不受干扰，x = 0.5 时，输出约为 1，因此 -1 进行去中心化
    depth_input = 1. / (depth_input.sigmoid() + 1e-6) - 1.
    depth_target = extract_target_from_tensor(target['depth'], target['mask_3d'])
    if target['mask_3d'].sum() > 0:  # 来自 monopair
        depth_loss = laplacian_aleatoric_uncertainty_loss(depth_input, depth_target, depth_log_variance)
    else:
        depth_loss = torch.tensor([0.0]).to(depth_input.device)
    return depth_loss


def compute_offset3d_loss(input, target, edge_fusion=False):
    offset3d_input = extract_input_from_tensor(input['offset_3d'], target['indices'], target['mask_3d'])
    offset3d_target = extract_target_from_tensor(target['offset_3d'], target['mask_3d'])
    if target['mask_3d'].sum() > 0:
        offset3d_loss = F.l1_loss(offset3d_input, offset3d_target, reduction='mean')
    else:
        offset3d_loss = torch.tensor([0.0]).to(offset3d_input.device)
    if edge_fusion:
        sum_target_trunc_mask = target['trunc_mask'].sum()
        if sum_target_trunc_mask > 0:
            trunc_offset3d_input = extract_input_from_tensor(input['offset_3d'], target['indices'],
                                                             target['trunc_mask'])
            trunc_offset3d_target = extract_target_from_tensor(target['offset_3d'], target['trunc_mask'])
            trunc_offset3d_loss = torch.log(1 + F.l1_loss(trunc_offset3d_input,
                                                          trunc_offset3d_target, reduction='none').sum() / torch.clamp(
                sum_target_trunc_mask, min=1))

        else:
            trunc_offset3d_loss = torch.tensor([0.0]).to(offset3d_input.device)
        return offset3d_loss + trunc_offset3d_loss
    return offset3d_loss


def compute_size3d_loss(input, target):
    size3d_input = extract_input_from_tensor(input['size_3d'], target['indices'], target['mask_3d'])
    size3d_target = extract_target_from_tensor(target['size_3d'], target['mask_3d'])
    # target['dimension'] is size3d_target
    dimension_target = extract_target_from_tensor(target['dimension'], target['mask_3d'])
    if target['mask_3d'].sum() > 0:
        size3d_loss = dim_aware_l1_loss(size3d_input, size3d_target, dimension_target)
    else:
        size3d_loss = torch.tensor([0.0]).to(size3d_input.device)
    return size3d_loss


def compute_heading_loss(input, target):
    heading_input = _transpose_and_gather_feat(input['heading'], target['indices'])  # B * C * H * W ---> B * K * C
    heading_input = heading_input.view(-1, 24)
    heading_target_cls = target['heading_bin'].view(-1)
    heading_target_res = target['heading_res'].view(-1)
    mask = target['mask_2d'].view(-1)

    # classification loss
    heading_input_cls = heading_input[:, 0:12]
    # heading_input_cls, heading_target_cls = heading_input_cls[mask], heading_target_cls[mask]
    heading_input_cls, heading_target_cls = heading_input_cls[mask > 0], heading_target_cls[mask > 0]
    if mask.sum() > 0:
        cls_loss = F.cross_entropy(heading_input_cls, heading_target_cls, reduction='mean')
    else:
        cls_loss = torch.tensor([0.0]).to(heading_input_cls.device)

    # regression loss
    heading_input_res = heading_input[:, 12:24]
    heading_input_res, heading_target_res = heading_input_res[mask > 0], heading_target_res[mask > 0]

    cls_onehot = torch.zeros(heading_target_cls.shape[0], 12).cuda().scatter_(dim=1,
                                                                              index=heading_target_cls.view(-1, 1),
                                                                              value=1)
    heading_input_res = torch.sum(heading_input_res * cls_onehot, 1)
    reg_loss = F.l1_loss(heading_input_res, heading_target_res, reduction='mean')
    if torch.any(torch.isnan(reg_loss)):
        reg_loss = torch.tensor([0.0]).to(heading_input_res.device)
    return cls_loss + reg_loss


######################  auxiliary functions #########################

def margin_distanse_loss(input, target, threshold, reduction='mean'):
    dis = torch.norm(input - target, dim=1)
    loss = torch.pow(torch.clamp(threshold - dis, min=0), 2)
    return loss.mean() if reduction == 'mean' else loss.sum()


def extract_input_from_tensor(input, ind, mask):
    input = _transpose_and_gather_feat(input, ind)  # B*C*H*W --> B*K*C
    return input[mask > 0]  # B*K*C --> M * C


def extract_target_from_tensor(target, mask):
    return target[mask > 0]


def image_point_to_road(road, p2_inv, points):  # 输出位于 rect 坐标系
    # Step 1: Convert the point to homogeneous coordinates
    uv_hom = torch.cat((points, torch.ones((points.shape[0], 1)).cuda()), dim=-1)

    # Step 2: Compute the inverse projection
    ray = torch.bmm(p2_inv, uv_hom.unsqueeze(2)).squeeze(2)

    # Since the ray is lambda * [X, Y, Z, 1]^T, its direction is only the first 3 elements.
    dir = ray[:, :3]

    # Step 3: Compute the intersection of the ray with the ground plane
    # ax + by + cz + d = 0 => aX + bY + cZ + d = 0 (for homogeneous [X, Y, Z, 1]^T)
    # lambda is the scaling factor
    lambda_val = -road[:, 3] / (road[:, 0] * dir[:, 0] + road[:, 1] * dir[:, 1] + road[:, 2] * dir[:, 2])
    lambda_val = lambda_val.unsqueeze(1)
    # Step 4: Compute the ground point
    ground_point = dir * lambda_val
    return ground_point


def img_to_rect(points, depth, calib):
    x = ((points[:, 0:1] - calib[:, 0:1]) * depth) / calib[:, 2:3] + calib[:, 4:5]
    y = ((points[:, 1:2] - calib[:, 1:2]) * depth) / calib[:, 3:4] + calib[:, 5:6]
    return torch.cat((x, y, depth), dim=-1)


if __name__ == '__main__':
    input_cls = torch.zeros(2, 50, 12)  # B * 50 * 24
    input_reg = torch.zeros(2, 50, 12)  # B * 50 * 24
    target_cls = torch.zeros(2, 50, 1, dtype=torch.int64)
    target_reg = torch.zeros(2, 50, 1)

    input_cls, target_cls = input_cls.view(-1, 12), target_cls.view(-1)
    cls_loss = F.cross_entropy(input_cls, target_cls, reduction='mean')

    a = torch.zeros(2, 24, 10, 10)
    b = torch.zeros(2, 10).long()
    c = torch.ones(2, 10).long()
    d = torch.zeros(2, 10, 1).long()
    e = torch.zeros(2, 10, 1)
    print(compute_heading_loss(a, b, c, d, e))
