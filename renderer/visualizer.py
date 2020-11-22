# Copyright (c) Facebook, Inc. and its affiliates.

"""
Visualizing 3D humans via Opengl
- Options:
    GUI mode: a screen is required

"""

from renderer import glViewer
from renderer import meshRenderer  # glRenderer
from renderer import viewer2D  # , glViewer, glRenderer
from renderer.image_utils import draw_raw_bbox, draw_hand_bbox, draw_body_bbox, draw_arm_pose


class Visualizer(object):
    """
   Visualizer to visualize SMPL reconstruction output from HMR family (HMR, SPIN, EFT)

    Args:
        reconstruction output
        rawImg, bbox, 
        smpl_params (shape, pose, cams )
    """

    def __init__(
            self,
            rendererType='opengl_gui'  # nongui or gui
    ):
        self.rendererType = rendererType
        if rendererType != "opengl_gui" and rendererType != "opengl":
            print("Wrong rendererType: {rendererType}")
            assert False

        self.cam_all = []
        self.vert_all = []
        self.bboxXYXY_all = []

        self.bg_image = None

        # Output rendering
        self.renderout = None

    def visualize(self,
                  input_img,
                  hand_bbox_list=None,
                  body_bbox_list=None,
                  body_pose_list=None,
                  raw_hand_bboxes=None,
                  pred_mesh_list=None,
                  vis_raw_hand_bbox=True,
                  vis_body_pose=True,
                  ):
        # init
        res_img = input_img.copy()

        # draw raw hand bboxes
        if raw_hand_bboxes is not None and vis_raw_hand_bbox:
            res_img = draw_raw_bbox(input_img, raw_hand_bboxes)
            # res_img = np.concatenate((res_img, raw_bbox_img), axis=1)

        # draw 2D Pose
        if body_pose_list is not None and vis_body_pose:
            res_img = draw_arm_pose(res_img, body_pose_list)

        # draw body bbox
        if body_bbox_list is not None:
            body_bbox_img = draw_body_bbox(input_img, body_bbox_list)
            res_img = body_bbox_img

        # draw hand bbox
        if hand_bbox_list is not None:
            res_img = draw_hand_bbox(res_img, hand_bbox_list)

        # render predicted meshes
        if pred_mesh_list is not None:
            self.__render_pred_verts(input_img, pred_mesh_list)

        return res_img

    def __render_pred_verts(self, img_original, pred_mesh_list):

        res_img = img_original.copy()

        pred_mesh_list_offset = []
        for mesh in pred_mesh_list:
            # Mesh vertices have in image coordinate (left, top origin)
            # Move the X-Y origin in image center
            mesh_offset = mesh['vertices'].copy()
            mesh_offset[:, 0] -= img_original.shape[1] * 0.5
            mesh_offset[:, 1] -= img_original.shape[0] * 0.5
            pred_mesh_list_offset.append({'ver': mesh_offset, 'f': mesh['faces']})  # verts = mesh['vertices']
            # faces = mesh['faces']
        if self.rendererType == "opengl_gui":
            self._visualize_gui_naive(pred_mesh_list_offset, img_original=res_img)

    def _visualize_gui_naive(self, meshList, skelList=None, body_bbox_list=None, img_original=None,
                             normal_compute=True):
        """
            args:
                meshList: list of {'ver': pred_vertices, 'f': smpl.faces}
                skelList: list of [JointNum*3, 1]       (where 1 means num. of frames in glviewer)
                bbr_list: list of [x,y,w,h]
        """
        if body_bbox_list is not None:
            for bbr in body_bbox_list:
                viewer2D.Vis_Bbox(img_original, bbr)
        # viewer2D.ImShow(img_original)

        glViewer.setWindowSize(img_original.shape[1], img_original.shape[0])
        # glViewer.setRenderOutputSize(inputImg.shape[1],inputImg.shape[0])
        glViewer.setBackgroundTexture(img_original)
        glViewer.SetOrthoCamera(True)
        glViewer.setMeshData(meshList,
                             bComputeNormal=normal_compute)  # meshes = {'ver': pred_vertices, 'f': smplWrapper.f}

        if skelList is not None:
            glViewer.setSkeleton(skelList)

        # glViewer.setSaveFolderName(overlaidImageFolder)
        glViewer.setNearPlane(50)
        glViewer.setWindowSize(img_original.shape[1], img_original.shape[0])
        # glViewer.show_SMPL(bSaveToFile = True, bResetSaveImgCnt = False, countImg = False, mode = 'camera')
        glViewer.show(100000)
