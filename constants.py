"""
We create a superset of joints containing the OpenPose joints together with the ones that each dataset provides.
We keep a superset of 24 joints such that we include all joints from every dataset.
If a dataset doesn't provide annotations for a specific joint, we simply ignore it.
The joints used here are the following:
"""
JOINT_NAMES = [
    'Right Ankle', 'Right Knee', 'Right Hip',  # 0,1,2
    'Left Hip', 'Left Knee', 'Left Ankle',  # 3, 4, 5
    'Right Wrist', 'Right Elbow', 'Right Shoulder',  # 6
    'Left Shoulder', 'Left Elbow', 'Left Wrist',  # 9
    'Neck (LSP)', 'Top of Head (LSP)',  # 12, 13
    'Pelvis (MPII)', 'Thorax (MPII)',  # 14, 15
    'Spine (H36M)', 'Jaw (H36M)',  # 16, 17
    'Head (H36M)', 'Nose', 'Left Eye',  # 18, 19, 20
    'Right Eye', 'Left Ear', 'Right Ear'  # 21,22,23 (Total 24 joints)
]

BODY_JOINT_NAMES_DISPLAY = [
    'pelvis',  # 0
    'left_hip',  # 1
    'right_hip',  # 2
    'spine1',  # 3
    'left_knee',  # 4
    'right_knee',  # 5
    'spine2',  # 6
    'left_ankle',  # 7
    'right_ankle',  # 8
    'spine3',  # 9
    'left_foot',  # 10
    'right_foot',  # 11
    'neck',  # 12
    'left_collar',  # 13
    'right_collar',  # 14
    'head',  # 15
    'left_shoulder',  # 16
    'right_shoulder',  # 17
    'left_elbow',  # 18
    'right_elbow',  # 19
    'left_wrist',  # 20
    'right_wrist',  # 21
    'left_hand',  # 22
    'right_hand',  # 23
]

LEFT_HAND_JOINT_NAMES_DISPLAY = [
    'left_index1',
    'left_index2',
    'left_index3',
    'left_middle1',
    'left_middle2',
    'left_middle3',
    'left_pinky1',
    'left_pinky2',
    'left_pinky3',
    'left_ring1',
    'left_ring2',
    'left_ring3',
    'left_thumb1',
    'left_thumb2',
    'left_thumb3',
]

RIGHT_HAND_JOINT_NAMES_DISPLAY = [
    'right_index1',
    'right_index2',
    'right_index3',
    'right_middle1',
    'right_middle2',
    'right_middle3',
    'right_pinky1',
    'right_pinky2',
    'right_pinky3',
    'right_ring1',
    'right_ring2',
    'right_ring3',
    'right_thumb1',
    'right_thumb2',
    'right_thumb3',
]

JOINTS_TEMP_SMOOTH = {
    'pelvis': False,
    'left_hip': False,
    'right_hip': False,
    'spine1': False,
    'left_knee': False,
    'right_knee': False,
    'spine2': False,
    'left_ankle': False,
    'right_ankle': False,
    'spine3': False,
    'left_foot': False,
    'right_foot': False,
    'neck': True,
    'left_collar': True,
    'right_collar': True,
    'head': True,
    'left_shoulder': True,
    'right_shoulder': True,
    'left_elbow': True,
    'right_elbow': True,
    'left_wrist': True,
    'right_wrist': True,
    'left_hand': True,
    'right_hand': True,
}

# Dict containing the joints in numerical order
JOINT_IDS = {JOINT_NAMES[i]: i for i in range(len(JOINT_NAMES))}
BODY_JOINT_IDS_DISPLAY = {BODY_JOINT_NAMES_DISPLAY[i]: i for i in range(len(BODY_JOINT_NAMES_DISPLAY))}
LEFT_HAND_JOINT_IDS_DISPLAY = {LEFT_HAND_JOINT_NAMES_DISPLAY[i]: i for i in range(len(LEFT_HAND_JOINT_NAMES_DISPLAY))}
RIGHT_HAND_JOINT_IDS_DISPLAY = {RIGHT_HAND_JOINT_NAMES_DISPLAY[i]: i for i in
                                range(len(RIGHT_HAND_JOINT_NAMES_DISPLAY))}

# Map joints to SMPL joints
JOINT_MAP = {
    'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45,
    'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7,
    'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17,
    'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20,
    'Neck (LSP)': 47, 'Top of Head (LSP)': 48,
    'Pelvis (MPII)': 49, 'Thorax (MPII)': 50,
    'Spine (H36M)': 51, 'Jaw (H36M)': 52,
    'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26,
    'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27
}

# Joint selectors
# Indices to get the 14 LSP joints from the 17 H36M joints
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]
# Indices to get the 14 LSP joints from the ground truth joints
J24_TO_J17 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 14, 16, 17]
J24_TO_J14 = J24_TO_J17[:14]

# Permutation of SMPL pose parameters when flipping the shape
SMPL_JOINTS_FLIP_PERM = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]
SMPL_POSE_FLIP_PERM = []
for i in SMPL_JOINTS_FLIP_PERM:
    SMPL_POSE_FLIP_PERM.append(3 * i)
    SMPL_POSE_FLIP_PERM.append(3 * i + 1)
    SMPL_POSE_FLIP_PERM.append(3 * i + 2)
# Permutation indices for the 24 ground truth joints
J24_FLIP_PERM = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18, 19, 21, 20, 23, 22]
# Permutation indices for the full set of 49 joints
J49_FLIP_PERM = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21] \
                + [25 + i for i in J24_FLIP_PERM]
