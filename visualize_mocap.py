# Copyright (c) Facebook, Inc. and its affiliates.

import cv2
import numpy as np
import torch

import utils as gnu
import visualization_utils as demo_utils
from arguments import ArgumentOptions
from conversion_utils import convert_smpl_to_bbox, convert_bbox_to_oriIm
from hand_module import extract_hand_output
from models import SMPLX
from parameter_gui import ParametersWindow
from renderer.glViewer import backwardsDirection as goBackwards
from renderer.visualizer import Visualizer

display_time = 100000


def continue_frames():
    global display_time
    display_time = 1


def stop_frames():
    global display_time
    display_time = 100000


paramsWin = ParametersWindow(np.zeros(72), np.zeros(45), np.zeros(45), continue_frames, stop_frames)


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
    pred_verts, faces = _calc_body_mesh(smpl_model, pose_params, betas, pred_left_hand_pose,
                                        pred_right_hand_pose)

    pred_output['pred_vertices_smpl'] = pred_verts
    pred_output['faces'] = faces


def extract_average_shape(pkl_files):
    pred_shape = np.empty((0, 10))

    for pkl_file in pkl_files:
        saved_data = gnu.load_pkl(pkl_file)
        pred_output_list = saved_data['pred_output_list']
        assert len(pred_output_list) > 0
        pred_output = pred_output_list[0]
        pred_shape = np.mean(np.vstack((pred_shape, pred_output['pred_betas'].reshape(1, -1))), axis=0, dtype="float32")
    return pred_shape.reshape(1, -1)


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


def fix_body_posture(pred_output_list, cam_scale, cam_trans, bbox_scale_ratio, bbox_top_left):
    pred_output = pred_output_list[0]
    vert_bboxcoord = convert_smpl_to_bbox(pred_output['pred_vertices_smpl'], cam_scale, cam_trans,
                                          bAppTransFirst=False)  # SMPL space -> bbox space
    vert_imgcoord = convert_bbox_to_oriIm(vert_bboxcoord, bbox_scale_ratio, bbox_top_left)
    pred_output['pred_vertices_img'] = vert_imgcoord


def average_camera(all_pred_output_posture):
    camera_scales = all_pred_output_posture['pred_camera_scale']
    camera_transf = all_pred_output_posture['pred_camera_trans']
    avg_scale = np.mean(camera_scales)
    avg_transf = np.mean(camera_transf, axis=0)
    return avg_scale, avg_transf


def average_bbox(all_pred_output_posture):
    bbox_scale = all_pred_output_posture['bbox_scale_ratio']
    bbox_top_left = all_pred_output_posture['bbox_top_left']
    avg_scale = np.mean(bbox_scale)
    avg_top_left = np.mean(bbox_top_left, axis=0)
    return avg_scale, avg_top_left


def extract_general_pose(pkl_files):
    all_pred_output_posture = {
        'pred_camera_scale': np.array([]),
        'pred_camera_trans': np.empty((0, 2)),
        'bbox_scale_ratio': np.array([]),
        'bbox_top_left': np.empty((0, 2)),
    }
    for pkl_file in pkl_files:
        saved_data = gnu.load_pkl(pkl_file)
        pred_output_list = saved_data['pred_output_list']
        assert len(pred_output_list) > 0
        pred_output = pred_output_list[0]
        all_pred_output_posture['pred_camera_scale'] = np.hstack(
            (all_pred_output_posture['pred_camera_scale'], pred_output['pred_camera'][0]))
        all_pred_output_posture['pred_camera_trans'] = np.vstack(
            (all_pred_output_posture['pred_camera_trans'], pred_output['pred_camera'][1:]))
        all_pred_output_posture['bbox_scale_ratio'] = np.hstack(
            (all_pred_output_posture['bbox_scale_ratio'], pred_output['bbox_scale_ratio']))
        all_pred_output_posture['bbox_top_left'] = np.vstack(
            (all_pred_output_posture['bbox_top_left'], pred_output['bbox_top_left']))
    avg_camera_scale, avg_camera_trans = average_camera(all_pred_output_posture)
    avg_bbox_scale, avg_bbox_top_left = average_bbox(all_pred_output_posture)
    return avg_camera_scale, avg_camera_trans, avg_bbox_scale, avg_bbox_top_left


def visualize_prediction(args, smpl_model, pkl_files, visualizer):
    print('Calculating average camera parameters...')
    avg_camera_scale, avg_camera_trans, avg_bbox_scale, avg_bbox_top_left = extract_general_pose(pkl_files)
    print('Average camera parameters done.')

    print('Calculating average shape parameters...')
    pred_shape = extract_average_shape(pkl_files)
    print('Average shape parameters done.')

    global display_time

    length = len(pkl_files)
    i = 0
    while i < length:
        # load data
        pkl_file = pkl_files[i]
        saved_data = gnu.load_pkl(pkl_file)

        image_path = saved_data['image_path']

        image_path = image_path.replace('frankmocap/mocap_output_gen/frames',
                                        'masters-thesis/frankmocap_output/rendered')

        img_original_bgr = cv2.imread(image_path)
        if img_original_bgr is None:
            print(f"{image_path} does not exists, skip")

        img_original_bgr = img_original_bgr[:, :1280]

        print("--------------------------------------")

        hand_bbox_list = saved_data['hand_bbox_list']
        body_bbox_list = saved_data['body_bbox_list']
        pred_output_list = saved_data['pred_output_list']

        global paramsWin
        body_pose = pred_output_list[0]['pred_body_pose']
        paramsWin.set_body_params(body_pose)

        left_hand_pose = pred_output_list[0]['pred_left_hand_pose']
        paramsWin.set_left_hand_params(left_hand_pose)

        right_hand_pose = pred_output_list[0]['pred_right_hand_pose']
        paramsWin.set_right_hand_params(right_hand_pose)

        parameters = paramsWin.get_body_params()
        pred_output_list[0]['pred_body_pose'] = parameters

        parameters = paramsWin.get_left_param_params()
        pred_output_list[0]['pred_left_hand_pose'] = parameters

        parameters = paramsWin.get_right_param_params()
        pred_output_list[0]['pred_right_hand_pose'] = parameters

        pred_output_list[0]['pred_betas'] = pred_shape

        __calc_mesh(smpl_model, pred_output_list)
        fix_body_posture(pred_output_list, avg_camera_scale, avg_camera_trans, avg_bbox_scale, avg_bbox_top_left)
        pred_mesh_list = demo_utils.extract_mesh_from_output(pred_output_list)

        def refresh_mesh():
            parameters = paramsWin.get_body_params()
            pred_output_list[0]['pred_body_pose'] = parameters

            parameters = paramsWin.get_left_param_params()
            pred_output_list[0]['pred_left_hand_pose'] = parameters

            parameters = paramsWin.get_right_param_params()
            pred_output_list[0]['pred_right_hand_pose'] = parameters

            __calc_mesh(smpl_model, pred_output_list)
            fix_body_posture(pred_output_list, avg_camera_scale, avg_camera_trans, avg_bbox_scale, avg_bbox_top_left)
            pred_mesh_list_update = demo_utils.extract_mesh_from_output(pred_output_list)
            visualizer.update_mesh(pred_mesh_list_update, img_original_bgr)

        def continue_frame():
            return display_time == 1

        # visualization
        res_image = visualizer.visualize(
            img_original_bgr,
            pred_mesh_list=pred_mesh_list,
            body_bbox_list=body_bbox_list,
            hand_bbox_list=hand_bbox_list,
            refresh_mesh=refresh_mesh,
            display_time=display_time,
            continue_frame=continue_frame)

        if goBackwards():
            if i > 0:
                i -= 1
        else:
            i += 1

        # save predictions to pkl
        if args.save_pred_pkl:
            args.use_smplx = True
            demo_utils.save_pred_to_pkl(
                args, pkl_file, body_bbox_list, hand_bbox_list, pred_output_list)

            # save the obtained video frames
        if args.out_dir is not None:
            demo_utils.save_res_img(args.out_dir, pkl_file, res_image)

    demo_utils.gen_video_out(args.out_dir, 'result_video')


def main():
    args = ArgumentOptions().parse()

    # load pkl files
    pkl_files = gnu.get_all_files(args.pkl_dir, ".pkl", "full")

    # get smpl model
    smpl_model = __get_smpl_model()

    # Set Visualizer
    visualizer = Visualizer('opengl_gui')

    # load smpl model
    visualize_prediction(args, smpl_model, pkl_files, visualizer)


if __name__ == '__main__':
    main()
