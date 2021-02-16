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
    "Global",
    "L_Hip",
    "R_Hip",
    "Spine_01",
    "L_Knee",
    "R_Knee",
    "Spine_02",
    "L_Ankle",
    "R_Ankle",
    "Spine_03",
    "L_Toe",
    "R_Toe",
    "Middle_Shoulder",
    "L_Clavice",
    "R_Clavice",
    "Nose",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Palm(Invalid for SMPL - X)",
    "R_Palm(Invalid for SMPL - X)"
]

LEFT_HAND_JOINT_NAMES_DISPLAY = [
    "Index_00",
    "Index_01",
    "Index_02",
    "Middle_00",
    "Middle_01",
    "Middle_02",
    "Little_00",
    "Little_01",
    "Little_02",
    "Ring_00",
    "Ring_01",
    "Ring_02",
    "Thumb_00",
    "Thumb_01",
    "Thumb_02"
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

# Mapping from hand marker points to corresponding vertex indices on [left, right] hand
# Palm is default, back markers are marked with 'back'
# DP - joint between Distal and Proximal phalanges
# PM - joint between Proximal phalanx and Metacarpal
# DI - joint between Distal and Intermediate phalanges
# IP - joint between Intermediate and Proximal phalanges
HAND_MARK_TO_VERTEX = {
    'thumb_point': [10891, 22109],
    'thumb_DP1': [10882, 22095],
    'thumb_DP2': [10914, 22146],
    'thumb_DP3': [14340, 22162],
    'thumb_PM': [10333, 20495],
    'index_point': [11162, 20757],
    'index_DI1': [10265, 20721],
    'index_DI2': [13026, 20826],
    'index_DI3': [15885, 20729],
    'index_IP1': [10540, 20226],
    'index_IP2': [10168, 20076],
    'index_IP3': [10085, 20574],
    'index_PM': [10107, 20251],
    'middle_point': [10409, 21152],
    'middle_DI1': [10400, 21147],
    'middle_DI2': [11125, 21177],
    'middle_DI3': [10416, 21077],
    'middle_IP1': [10370, 20955],
    'middle_IP2': [10378, 20940],
    'middle_IP3': [12374, 21033],
    'middle_PM': [11044, 20635],
    'ring_point': [10628, 21506],
    'ring_DI1': [10602, 21528],
    'ring_DI2': [10606, 21531],
    'ring_DI3': [14319, 21431],
    'ring_IP1': [12026, 21312],
    'ring_IP2': [10496, 21302],
    'ring_IP3': [10484, 21381],
    'ring_PM': [10862, 20208],
    'little_point': [10725, 21852],
    'little_DI1': [10703, 21876],
    'little_DI2': [10791, 21879],
    'little_DI3': [10706, 21779],
    'little_IP1': [10679, 21657],
    'little_IP2': [10700, 21642],
    'little_IP3': [10648, 21660],
    'little_PM': [10782, 21664],
    'wrist_thumb': [10093, 20120],
    'wrist_middle': [12397, 20162],
    'wrist_outer': [10002, 20149],
    'palm_inner': [9964, 20047],
    'palm_outer': [10017, 20221],
    'thumb_DP_back': [10818, 22020],
    'thumb_PM_back': [9988, 20116],
    'index_DI_back': [10227, 20712],
    'index_IP_back': [10064, 20331],
    'index_PM_back': [10183, 20211],
    'middle_DI_back': [11033, 21105],
    'middle_IP_back': [10560, 20982],
    'middle_PM_back': [10472, 20353],
    'ring_DI_back': [10709, 21414],
    'ring_IP_back': [10478, 21336],
    'ring_PM_back': [10527, 20264],
    'little_DI_back': [12178, 21762],
    'little_IP_back': [10646, 21696],
    'little_PM_back': [11102, 20556],
    'wrist_middle_back': [10125, 20322]
}

# Hand measurement distances as defined in https://link.springer.com/article/10.1007/s12652-020-02354-8
DISTANCE_INDEX_TO_HAND_MARK_PAIR = {
    1: ['middle_point', 'wrist_middle'],
    2: ['middle_PM', 'wrist_middle'],
    3: ['thumb_PM', 'thumb_DP2'],
    4: ['thumb_DP2', 'thumb_point'],
    5: ['thumb_PM', 'thumb_point'],
    6: ['index_PM', 'index_IP2'],
    7: ['index_IP2', 'index_DI2'],
    8: ['index_DI2', 'index_point'],
    9: ['index_PM', 'index_point'],
    10: ['middle_PM', 'middle_IP2'],
    11: ['middle_IP2', 'middle_DI2'],
    12: ['middle_DI2', 'middle_point'],
    13: ['middle_PM', 'middle_point'],
    14: ['ring_PM', 'ring_IP2'],
    15: ['ring_IP2', 'ring_DI2'],
    16: ['ring_DI2', 'ring_point'],
    17: ['ring_PM', 'ring_point'],
    18: ['little_PM', 'little_IP2'],
    19: ['little_IP2', 'little_DI2'],
    20: ['little_DI2', 'little_point'],
    21: ['little_PM', 'little_point'],
    22: ['thumb_point', 'thumb_DP_back'],
    23: ['thumb_DP_back', 'thumb_PM_back'],
    24: ['thumb_point', 'thumb_PM_back'],
    25: ['index_PM_back', 'index_IP_back'],
    26: ['index_IP_back', 'index_DI_back'],
    27: ['index_DI_back', 'index_point'],
    28: ['index_PM_back', 'index_point'],
    29: ['middle_PM_back', 'middle_IP_back'],
    30: ['middle_IP_back', 'middle_DI_back'],
    31: ['middle_DI_back', 'middle_point'],
    32: ['middle_PM_back', 'middle_point'],
    33: ['ring_PM_back', 'ring_IP_back'],
    34: ['ring_IP_back', 'ring_DI_back'],
    35: ['ring_DI_back', 'ring_point'],
    36: ['ring_PM_back', 'ring_point'],
    37: ['little_PM_back', 'little_IP_back'],
    38: ['little_IP_back', 'little_DI_back'],
    39: ['little_DI_back', 'little_point'],
    40: ['little_PM_back', 'little_point'],
    41: ['thumb_DP1', 'thumb_DP3'],
    42: ['index_IP1', 'index_IP3'],
    43: ['index_DI1', 'index_DI3'],
    44: ['middle_IP1', 'middle_IP3'],
    45: ['middle_DI1', 'middle_DI3'],
    46: ['ring_IP1', 'ring_IP3'],
    47: ['ring_DI1', 'ring_DI3'],
    48: ['little_IP1', 'little_IP3'],
    49: ['little_DI1', 'little_DI3'],
    50: ['wrist_thumb', 'wrist_outer'],
    51: ['palm_inner', 'palm_outer'],
    # circumferences
    # 52: ['', ''],
    # 53: ['', ''],
    # 54: ['', ''],
    # 55: ['', ''],
    # 56: ['', ''],
    # 57: ['', ''],
    # 58: ['', ''],
    # 59: ['', ''],
    # 60: ['', ''],
    # 61: ['', ''],
    62: ['index_PM_back', 'index_PM'],
    63: ['wrist_middle_back', 'wrist_middle'],
}
