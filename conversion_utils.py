"""
This file contains functions that are used to perform data augmentation.
"""

import numpy as np


# For converting coordinate between SMPL 3D coord <-> 2D bbox <-> original 2D image
# data3D: (N,3), where N is number of 3D points in "smpl"'s 3D coordinate (vertex or skeleton)

def convert_smpl_to_bbox(data3D, scale, trans, bAppTransFirst=False):
    data3D = data3D.copy()
    resnet_input_size_half = 224 * 0.5
    if bAppTransFirst:  # Hand model
        data3D[:, 0:2] += trans
        data3D *= scale  # apply scaling
    else:
        data3D *= scale  # apply scaling
        data3D[:, 0:2] += trans

    data3D *= resnet_input_size_half  # 112 is originated from hrm's input size (224,24)
    # data3D[:,:2]*= resnet_input_size_half # 112 is originated from hrm's input size (224,24)
    return data3D


def convert_bbox_to_oriIm(data3D, boxScale_o2n, bboxTopLeft):
    data3D = data3D.copy()
    resnet_input_size_half = 224 * 0.5

    data3D /= boxScale_o2n

    if not isinstance(bboxTopLeft, np.ndarray):
        assert isinstance(bboxTopLeft, tuple)
        assert len(bboxTopLeft) == 2
        bboxTopLeft = np.array(bboxTopLeft)

    data3D[:, :2] += (bboxTopLeft + resnet_input_size_half / boxScale_o2n)

    return data3D
