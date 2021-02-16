import csv
import os
import re

import numpy as np
import open3d as o3d

POINT_CLOUD = False
SAVE_POINT_CLOUD = False
VISUALIZE = True

BASE_FOLDER = "base_poses"

OBJ_FILES_FOLDER = os.path.join(BASE_FOLDER, "obj_files")
PLY_FILES_FOLDER = os.path.join(BASE_FOLDER, "ply_files")

OUTPUT_FILE = os.path.join(BASE_FOLDER, "ground_truth.csv")

# the lengths are just identified by their numeric id as in https://link.springer.com/article/10.1007/s12652-020-02354-8
# 1-63  (range is to 64 because it goes to one less)
COLUMN_NAMES = ["file_path", "pose_id", "shape_id"] + list(map(str, range(1, 64)))


def visualize_obj():
    for file in os.listdir(OBJ_FILES_FOLDER):
        mesh = o3d.io.read_triangle_mesh(os.path.join(OBJ_FILES_FOLDER, file))

        pose_shape = re.split('[a-zA-Z_.]+', file)
        pose_shape = list(filter(None, pose_shape))
        pose_id = int(pose_shape[0])
        shape_id = int(pose_shape[1])

        metadata = {
            "file_path": os.path.join(OBJ_FILES_FOLDER, file),
            "pose_id": pose_id,
            "shape_id": shape_id
        }

        writer.writerow(metadata)

        mesh.compute_vertex_normals()
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices

        # print("Vertices: ")
        # print(np.asarray(mesh.vertices))
        # print("Triangles: ")
        # print(np.asarray(mesh.triangles))

        if VISUALIZE:
            if POINT_CLOUD:
                o3d.visualization.draw_geometries_with_editing([pcd],
                                                               window_name='Open3D', width=1280, height=960)
            else:
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

    visualize_obj()

    csv_file.close()
