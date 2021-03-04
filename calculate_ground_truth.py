import csv
import json
import os
import re
import subprocess

import numpy as np
import open3d as o3d

from constants import DISTANCE_INDEX_TO_HAND_MARK_PAIR, HAND_MARK_TO_VERTEX, CIRCUMFERENCE_MEASUREMENT_LABELS

POINT_CLOUD = False
SAVE_POINT_CLOUD = True
VISUALIZE = False

BASE_FOLDER = "base_poses"

OBJ_FILES_FOLDER = os.path.join(BASE_FOLDER, "obj_files")
PLY_FILES_FOLDER = os.path.join(BASE_FOLDER, "ply_files")

CIRC_SOFTWARE_PATH = os.path.join("circumference_software", "LandmarkPointsCmd.exe")
CIRC_MEASUREMENTS_PATH = os.path.join(BASE_FOLDER, "hand_circumferences.json")
CIRC_OUTPUT_FILE_JSON = "measurements.json"
CIRC_OUTPUT_FILE_CSV = "measurements.csv"

OUTPUT_FILE = os.path.join(BASE_FOLDER, "ground_truth.csv")

# the lengths are just identified by their numeric id as in https://link.springer.com/article/10.1007/s12652-020-02354-8
# 1-63  (range is to 64 because it goes to one less)
COLUMN_NAMES = ["file_path", "pose_id", "shape_id"] + list(map(lambda x: 'L' + str(x), range(1, 64))) + list(
    map(lambda x: 'R' + str(x), range(1, 64)))


def visualize_obj(csv_writer):
    for file in os.listdir(OBJ_FILES_FOLDER):
        mesh_file = os.path.join(OBJ_FILES_FOLDER, file)
        mesh = o3d.io.read_triangle_mesh(mesh_file)

        pose_shape = re.split('[a-zA-Z_.]+', file)
        pose_shape = list(filter(None, pose_shape))
        pose_id = int(pose_shape[0])
        shape_id = int(pose_shape[1])

        metadata = {
            "file_path": mesh_file,
            "pose_id": pose_id,
            "shape_id": shape_id
        }

        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices

        # print("Vertices: ")
        # print(np.asarray(mesh.vertices))
        # print("Triangles: ")
        # print(np.asarray(mesh.triangles))

        # distances
        for dist_idx, [distA, distB] in DISTANCE_INDEX_TO_HAND_MARK_PAIR.items():
            # left hand distance
            p1 = pcd.points[HAND_MARK_TO_VERTEX[distA][0]]
            p2 = pcd.points[HAND_MARK_TO_VERTEX[distB][0]]
            # distL = np.sqrt(np.sum((p1-p2)**2, axis=0))
            distL = np.linalg.norm(p1 - p2)
            metadata['L' + str(dist_idx)] = distL

            # right hand distance
            p1 = pcd.points[HAND_MARK_TO_VERTEX[distA][1]]
            p2 = pcd.points[HAND_MARK_TO_VERTEX[distB][1]]
            # distR = np.sqrt(np.sum((p1-p2)**2, axis=0))
            distR = np.linalg.norm(p1 - p2)
            metadata['R' + str(dist_idx)] = distR

        # circumferences
        subprocess.call([CIRC_SOFTWARE_PATH, CIRC_MEASUREMENTS_PATH, mesh_file])
        with open(CIRC_OUTPUT_FILE_JSON) as f:
            json_data = json.load(f)
            circ_measurements = json_data["Measurements"]
            for measurement in circ_measurements:
                hand_side, circ_label = measurement["label"].split("_", 1)
                msr_idx = CIRCUMFERENCE_MEASUREMENT_LABELS[circ_label]
                # for some reason the circumference measurements have a different scale
                if hand_side == "left":
                    metadata['L' + str(msr_idx)] = float(measurement["value"]) / 100
                elif hand_side == "right":
                    metadata['R' + str(msr_idx)] = float(measurement["value"]) / 100

        for key in COLUMN_NAMES:
            if key not in metadata:
                metadata[key] = 0

        csv_writer.writerow(metadata)

        if VISUALIZE:
            if POINT_CLOUD:
                o3d.visualization.draw_geometries_with_editing([pcd],
                                                               window_name='Open3D', width=1280, height=960)
            else:
                mesh.compute_vertex_normals()
                o3d.visualization.draw_geometries_with_vertex_selection([mesh],
                                                                        window_name='Open3D', width=1280, height=960)

        if SAVE_POINT_CLOUD:
            file_name = os.path.join(PLY_FILES_FOLDER, file[:-4] + ".ply")
            o3d.io.write_point_cloud(file_name, pcd)

        print("Processed file {0}".format(file))


if __name__ == "__main__":
    os.makedirs(PLY_FILES_FOLDER, exist_ok=True)

    csv_file = open(OUTPUT_FILE, mode='w', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=COLUMN_NAMES)
    writer.writeheader()

    visualize_obj(writer)

    csv_file.close()

    os.remove(CIRC_OUTPUT_FILE_JSON)
    os.remove(CIRC_OUTPUT_FILE_CSV)
