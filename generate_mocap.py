import os
import os.path as osp
import random
from collections import OrderedDict
from datetime import datetime
from shutil import copy, rmtree

import cv2
import numpy as np
import torch

import utils as gnu
import visualization_utils as demo_utils
from arguments import ArgumentOptions
from conversion_utils import convert_smpl_to_bbox, convert_bbox_to_oriIm
from hand_module import extract_hand_output
from models import SMPLX

random.seed(datetime.now())


def __get_smpl_model():
    smplx_model_path = './extra_data/smpl/SMPLX_NEUTRAL.pkl'

    smpl = SMPLX(
        smplx_model_path,
        batch_size=1,
        num_betas=10,
        use_pca=False,
        create_transl=False)
    return smpl


def __calc_hand_mesh(hand_type, pose_params, betas, smplx_model):
    hand_rotation = pose_params[:, :3]
    hand_pose = pose_params[:, 3:]
    body_pose = torch.zeros((1, 63)).float()

    assert hand_type in ['left_hand', 'right_hand']
    if hand_type == 'right_hand':
        body_pose[:, 60:] = hand_rotation  # set right hand rotation
        right_hand_pose = hand_pose
        left_hand_pose = torch.zeros((1, 45), dtype=torch.float32)
    else:
        body_pose[:, 57:60] = hand_rotation  # set right hand rotation
        left_hand_pose = hand_pose
        right_hand_pose = torch.zeros((1, 45), dtype=torch.float32)

    output = smplx_model(
        global_orient=torch.zeros((1, 3)),
        body_pose=body_pose,
        betas=betas,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        return_verts=True)

    hand_info_file = "extra_data/hand_module/SMPLX_HAND_INFO.pkl"
    hand_info = gnu.load_pkl(hand_info_file)
    hand_output = extract_hand_output(
        output,
        hand_type=hand_type.split("_")[0],
        hand_info=hand_info,
        top_finger_joints_type='ave',
        use_cuda=False)

    pred_verts = hand_output['hand_vertices_shift'].detach().numpy()
    faces = hand_info[f'{hand_type}_faces_local']
    return pred_verts[0], faces


def _calc_body_mesh(smpl_model, body_pose, betas,
                    left_hand_pose, right_hand_pose):
    smpl_output = smpl_model(
        global_orient=body_pose[:, :3],
        body_pose=body_pose[:, 3:],
        betas=betas,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
    )

    vertices = smpl_output.vertices.detach().cpu().numpy()[0]
    faces = smpl_model.faces
    return vertices, faces


def __calc_mesh(smpl_model, pred_output_list):
    pred_output = pred_output_list[0]
    pred_output['pred_body_pose'][0][0] = 0
    pose_params = torch.from_numpy(pred_output['pred_body_pose'])
    betas = torch.from_numpy(pred_output['pred_betas'])
    if 'pred_right_hand_pose' in pred_output:
        pred_right_hand_pose = torch.from_numpy(pred_output['pred_right_hand_pose'])
    else:
        pred_right_hand_pose = torch.zeros((1, 45), dtype=torch.float32)
    if 'pred_left_hand_pose' in pred_output:
        pred_left_hand_pose = torch.from_numpy(pred_output['pred_left_hand_pose'])
    else:
        pred_left_hand_pose = torch.zeros((1, 45), dtype=torch.float32)
    pred_verts, faces = _calc_body_mesh(
        smpl_model, pose_params, betas, pred_left_hand_pose, pred_right_hand_pose)

    pred_output['pred_vertices_smpl'] = pred_verts
    pred_output['faces'] = faces


def calculate_body_posture(pred_output_list):
    pred_output = pred_output_list[0]
    cam_scale = pred_output['pred_camera'][0]
    cam_trans = pred_output['pred_camera'][1:]
    vert_bboxcoord = convert_smpl_to_bbox(pred_output['pred_vertices_smpl'], cam_scale, cam_trans,
                                          bAppTransFirst=False)  # SMPL space -> bbox space
    bbox_scale_ratio = pred_output['bbox_scale_ratio']
    bbox_top_left = pred_output['bbox_top_left']
    vert_imgcoord = convert_bbox_to_oriIm(vert_bboxcoord, bbox_scale_ratio, bbox_top_left)
    pred_output['pred_vertices_img'] = vert_imgcoord


def generate_random_shape(out_dir, i):
    # https://khanhha.github.io/posts/SMPL-model-introduction/
    saved_data = OrderedDict()
    saved_data['betas'] = np.array((np.random.rand(1, 10) - 0.5) * 2.5, dtype="float32")

    pkl_name = "shape{0}.pkl".format(i)
    pkl_path = osp.join(out_dir, 'shapes', pkl_name)
    gnu.make_subdir(pkl_path)
    gnu.save_pkl(pkl_path, saved_data)
    print(f"Shape saved: {pkl_path}")


def read_shape(out_dir, nr):
    pkl_name = "shape{0}.pkl".format(nr)
    pkl_path = osp.join(out_dir, 'shapes', pkl_name)
    saved_data = gnu.load_pkl(pkl_path)
    return saved_data['betas']


def generate_prediction(args, smpl_model, pkl_files, idx_shape):
    shape = read_shape(args.out_dir, idx_shape)
    for pkl_file in pkl_files:
        # load data
        saved_data = gnu.load_pkl(pkl_file)

        print("--------------------------------------")

        pred_output_list = saved_data['pred_output_list']

        pred_output_list[0]['pred_betas'] = shape

        __calc_mesh(smpl_model, pred_output_list)
        calculate_body_posture(pred_output_list)
        pred_mesh_list = demo_utils.extract_mesh_from_output(pred_output_list)

        if args.out_dir is not None:
            demo_utils.save_obj_file(args.out_dir, pkl_file, pred_mesh_list[0]['vertices'],
                                     pred_mesh_list[0]['faces'], idx_shape)


def main():
    nr_data_shapes = 3000
    nr_data_poses = 100
    args = ArgumentOptions().parse()

    # load pkl files
    pkl_files = gnu.get_all_files(args.pkl_dir, ".pkl", "full")

    # get smpl model
    smpl_model = __get_smpl_model()

    # load smpl model
    for i in range(0, nr_data_shapes):
        generate_random_shape(args.out_dir, i)

    # select poses
    # random_list = random.sample(range(0, len(pkl_files) - 1), nr_data_poses)
    # random_list.sort()
    # selected_pkl_files = [pkl_files[i] for i in random_list]
    selected_pkl_files = pkl_files

    # save file in poses folder
    print("Copying selected poses...")
    pose_folder = os.path.join(args.out_dir, 'poses')
    if os.path.exists(pose_folder) and os.path.isdir(pose_folder):
        rmtree(pose_folder)
    os.makedirs(pose_folder, exist_ok=True)
    for pkl_file in selected_pkl_files:
        copy(pkl_file,
             os.path.join(pose_folder, os.path.basename(pkl_file.replace('_prediction_result_prediction_result', ''))))
    print("Copied selected poses.")

    # generate 3d models
    for i in range(0, nr_data_shapes):
        generate_prediction(args, smpl_model, selected_pkl_files, i)


if __name__ == '__main__':
    main()
