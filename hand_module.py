# Copyright (c) Facebook, Inc. and its affiliates.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# dct is the abbr. of Human Model recovery with Densepose supervision
import numpy as np
import torch


def extract_hand_output(output, hand_type, hand_info, top_finger_joints_type='ave', use_cuda=True):
    assert hand_type in ['left', 'right']

    if hand_type == 'left':
        wrist_idx, hand_start_idx, middle_finger_idx = 20, 25, 28
    else:
        wrist_idx, hand_start_idx, middle_finger_idx = 21, 40, 43

    vertices = output.vertices
    joints = output.joints
    vertices_shift = vertices - joints[:, hand_start_idx:hand_start_idx + 1, :]

    hand_verts_idx = torch.Tensor(hand_info[f'{hand_type}_hand_verts_idx']).long()
    if use_cuda:
        hand_verts_idx = hand_verts_idx.cuda()

    hand_verts = vertices[:, hand_verts_idx, :]
    hand_verts_shift = hand_verts - joints[:, hand_start_idx:hand_start_idx + 1, :]

    hand_joints = torch.cat((joints[:, wrist_idx:wrist_idx + 1, :],
                             joints[:, hand_start_idx:hand_start_idx + 15, :]), dim=1)

    # add top hand joints
    if len(top_finger_joints_type) > 0:
        if top_finger_joints_type in ['long', 'manual']:
            key = f'{hand_type}_top_finger_{top_finger_joints_type}_vert_idx'
            top_joint_vert_idx = hand_info[key]
            hand_joints = torch.cat((hand_joints, vertices[:, top_joint_vert_idx, :]), dim=1)
        else:
            assert top_finger_joints_type == 'ave'
            key1 = f'{hand_type}_top_finger_{top_finger_joints_type}_vert_idx'
            key2 = f'{hand_type}_top_finger_{top_finger_joints_type}_vert_weight'
            top_joint_vert_idxs = hand_info[key1]
            top_joint_vert_weight = hand_info[key2]
            bs = vertices.size(0)

            for top_joint_id, selected_verts in enumerate(top_joint_vert_idxs):
                top_finger_vert_idx = hand_verts_idx[np.array(selected_verts)]
                top_finger_verts = vertices[:, top_finger_vert_idx]
                # weights = torch.from_numpy(np.repeat(top_joint_vert_weight[top_joint_id]).reshape(1, -1, 1)
                weights = top_joint_vert_weight[top_joint_id].reshape(1, -1, 1)
                weights = np.repeat(weights, bs, axis=0)
                weights = torch.from_numpy(weights)
                if use_cuda:
                    weights = weights.cuda()
                top_joint = torch.sum((weights * top_finger_verts), dim=1).view(bs, 1, 3)
                hand_joints = torch.cat((hand_joints, top_joint), dim=1)

    hand_joints_shift = hand_joints - joints[:, hand_start_idx:hand_start_idx + 1, :]

    output = dict(
        wrist_idx=wrist_idx,
        hand_start_idx=hand_start_idx,
        middle_finger_idx=middle_finger_idx,
        vertices_shift=vertices_shift,
        hand_vertices=hand_verts,
        hand_vertices_shift=hand_verts_shift,
        hand_joints=hand_joints,
        hand_joints_shift=hand_joints_shift
    )
    return output
