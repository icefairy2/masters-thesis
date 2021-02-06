# Copyright (c) Facebook, Inc. and its affiliates.

import os
import os.path as osp
from collections import OrderedDict

import numpy as np

import utils as gnu


def extract_mesh_from_output(pred_output_list):
    pred_mesh_list = list()
    pred_output = pred_output_list[0]
    vertices = pred_output['pred_vertices_img']
    faces = pred_output['faces'].astype(np.int32)
    pred_mesh_list.append(dict(
        vertices=vertices,
        faces=faces
    ))
    return pred_mesh_list


def save_pred_to_pkl(args, image_path, body_bbox_list, hand_bbox_list, pred_output_list):
    assert len(hand_bbox_list) == len(body_bbox_list)
    assert len(body_bbox_list) == len(pred_output_list)

    # demo type / smpl type / image / bbox
    saved_data = OrderedDict()
    saved_data['image_path'] = osp.abspath(image_path)
    saved_data['body_bbox_list'] = body_bbox_list
    saved_data['hand_bbox_list'] = hand_bbox_list

    saved_data['pred_output_list'] = list()
    num_subject = len(hand_bbox_list)
    for s_id in range(num_subject):
        # predict params
        pred_output = pred_output_list[s_id]
        if pred_output is None:
            saved_pred_output = None
        else:
            saved_pred_output = dict()

            for pred_key in pred_output:
                if pred_key in ['pred_camera', 'bbox_scale_ratio', 'bbox_top_left', 'pred_betas', 'pred_body_pose',
                                'pred_right_hand_pose', 'pred_left_hand_pose', 'faces']:
                    saved_pred_output[pred_key] = pred_output[pred_key]

        saved_data['pred_output_list'].append(saved_pred_output)

    # write data to pkl
    img_name = osp.basename(image_path)
    record = img_name.split('.')
    pkl_name = f"{'.'.join(record[:-1])}_prediction_result.pkl"
    pkl_path = osp.join(args.out_dir, 'mocap', pkl_name)
    gnu.make_subdir(pkl_path)
    gnu.save_pkl(pkl_path, saved_data)
    print(f"Prediction saved: {pkl_path}")


def save_res_img(out_dir, image_path, res_img):
    out_dir = osp.join(out_dir, "frames")
    img_name = osp.basename(image_path)
    img_name = img_name[:-4] + '.png'
    res_img_path = osp.join(out_dir, img_name)
    gnu.make_subdir(res_img_path)
    # cv2.imwrite(res_img_path, res_img)
    res_img.save(res_img_path, 'PNG')
    print(f"Visualization saved: {res_img_path}")


def gen_video_out(out_dir, seq_name):
    outVideo_fileName = osp.join(out_dir, seq_name + '.mp4')
    print(f">> Generating video in {outVideo_fileName}")

    in_dir = osp.abspath(osp.join(out_dir, "frames"))
    out_path = osp.abspath(osp.join(out_dir, seq_name + '.mp4'))
    ffmpeg_cmd = f'ffmpeg -y -f image2 -framerate 25 -pattern_type glob -i "{in_dir}/*.png"  -pix_fmt yuv420p -c:v libx264 -x264opts keyint=25:min-keyint=25:scenecut=-1 -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" {out_path}'
    os.system(ffmpeg_cmd)
    # print(ffmpeg_cmd.split())
    # sp.run(ffmpeg_cmd.split())
    # sp.Popen(ffmpeg_cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE)


def save_obj_file(out_dir, file_path, verts, faces, idx_shape=''):
    obj_dir = "3d_objects"
    out_dir = osp.join(out_dir, obj_dir)
    obj_name = osp.basename(file_path)
    obj_name = 'pose' + obj_name[:-4] + 'shape' + str(idx_shape) + '.obj'
    res_obj_path = osp.join(out_dir, obj_name)

    if not os.path.exists(os.path.dirname(res_obj_path)):
        os.makedirs(os.path.dirname(res_obj_path))

    with open(res_obj_path, 'w') as fp:
        for v in verts:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in faces + 1:
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    print(f"Object saved: {res_obj_path}")
