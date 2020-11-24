# Copyright (c) Facebook, Inc. and its affiliates.

'''
This code has opengl visualization of 3D skeletons, including floor visualization and mouse+keyboard control
See the main function for demo codes.

showSkeleton: visualize 3D human body skeletons. This can handle holden's formation, 3.6m formation, and domeDB
setSpeech: to set speechAnnotation. Should be called before showSkeleton

renderscene: main function to render scenes using openGL

Note: this visualizer is assuming CM metric (joint, mesh).

Hanbyul Joo (jhugestar@gmail.com)
'''

import math
import pickle
import time

import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from PIL import Image, ImageOps

from renderer.render_utils import ComputeNormal

# -----------
# Global Variables
# -----------
# g_camView_fileName = '/ssd/camInfo/camInfo.pkl'
g_camView_fileName = './camInfo/camInfo.pkl'

g_fViewDistance = 50.
# g_Width = 1280
# g_Height = 720
g_Width = 1920
g_Height = 1080

g_nearPlane = 0.01  # original
# g_nearPlane = 1000          #cameramode
g_farPlane = 9000.

g_action = ""
g_xMousePtStart = g_yMousePtStart = 0.

g_xTrans = 0.
g_yTrans = 0.
g_zTrans = 0.
# g_zoom = 378
g_zoom = 600.
g_xRotate = 59.
g_yRotate = -41.
g_zRotate = 0.
g_xrot = 0.0
g_yrot = 0.0
# the usual screen dimension variables and lighting

# Generic Lighting values and coordinates
g_ambientLight = (0.35, 0.35, 0.35, 1.0)
g_diffuseLight = (0.75, 0.75, 0.75, 0.7)
g_specular = (0.2, 0.2, 0.2, 1.0)
g_specref = (0.5, 0.5, 0.5, 1.0)

# To visualize in Dome View point
# g_viewMode = 'camView'#free' #'camView'
# g_viewMode = 'camView' #'camView'
g_viewMode = 'free'  # 'camView'
g_bOrthoCam = False  # If true, draw by ortho camera mode
from collections import deque

# g_camid = deque('00',maxlen=2)
g_camid = deque('27', maxlen=2)

g_onlyDrawHumanIdx = -1

# To save rendered scene into file
g_stopMainLoop = False
g_winID = None

g_bSaveToFile = False
g_bSaveToFile_done = False  # If yes, file has been saved
g_saveImageName_last = None  # Remember the rendered image

# g_savedImg = None       # Keep the save image as a global variable, if ones want to obtain this
# g_haggling_render = False       #Automatically Load next seqeucne, when frames are done
g_bSaveOnlyMode = False  # If true, load camera turn on save mode and exit after farmes are done
g_saveFolderName = None
g_saveImageName = None

"Visualization Options"
g_bApplyRootOffset = False
ROOT_OFFSET_DIST = 160
# ROOT_OFFSET_DIST = 30

""" Mesh Drawing Option """

# A tutorial for VAO and VBO: https://www.khronos.org/opengl/wiki/Tutorial2:_VAOs,_VBOs,_Vertex_and_Fragment_Shaders_(C_/_SDL)
g_vao = None
g_vertex_buffer = None
g_normal_buffer = None
g_tangent_buffer = None
g_index_buffer = None

g_hagglingseq_name = None
# global vts_num

# global SMPL_vts
# global face_num

# global SMPLModel
g_saveFrameIdx = 0
# global MoshParam


g_bGlInitDone = False  # To initialize opengl only once

BACKGROUND_IMAGE_PLANE_DEPTH = 500

######################################################
#  MTC Camera view
g_camView_K = None
g_camView_K_list = None  # if  each mesh uses diff K.... very ugly
g_bShowBackground = True

g_backgroundTextureID = None
g_textureData = None
# g_textureImgOriginal = None  #Original Size

g_renderOutputSize = None  # (width, height) If not None, crop the output to this size
######################################################
#  Visualization Option

g_bShowFloor = False
g_bShowWiredMesh = False

g_bRotateView = False  # Rotate view 360 degrees
g_rotateView_counter = 0
g_rotateInterval = 2

g_bShowSkeleton = True
g_bShowMesh = True

g_backwardsDirection = False

########################################################
# 3D keypoints Visualization Setting
g_skeletons = None  # list of np.array. skeNum x  (skelDim, skelFrames)
# g_skeletons_GT = None
g_trajectory = None  # list of np.array. skeNum x  (trajDim:3, skelFrames)
HOLDEN_DATA_SCALING = 5

g_faces = None  # (faceNum, skelDim, skelFrames)
g_faceNormals = None
g_bodyNormals = None
g_hands = None  # (handNum, skelDim, skelFrames)
g_hands_left = None
g_hands_right = None

g_posOnly = None

g_meshes = None

g_cameraPoses = None
g_cameraRots = None
g_ptCloud = None
g_ptCloudColor = None
g_ptSize = 2

g_frameLimit = -1
g_frameIdx = 0
g_lastframetime = g_currenttime = time.time()
g_fps = 0

g_speech = None  # list of np.array. humanNum x  speechUnit, where elmenet is a dict with 'indicator', 'word', 'root' with a size of (1, skelFrames)
g_speechGT = None  # list of np.array. humanNum x  speechUnit, , where elmenet is a dict with 'indicator', 'word', 'root' with a size of (1, skelFrames)

# Original
# g_colors = [ (0, 255, 127), (209, 206, 0), (0, 0, 128), (153, 50, 204), (60, 20, 220),
#     (0, 128, 0), (180, 130, 70), (147, 20, 255), (128, 128, 240), (154, 250, 0), (128, 0, 0),
#     (30, 105, 210), (0, 165, 255), (170, 178, 32), (238, 104, 123)]

# g_colors = [ (0, 255, 127), (170, 170, 0), (0, 0, 128), (153, 50, 204), (60, 20, 220),
#     (0, 128, 0), (180, 130, 70), (147, 20, 255), (128, 128, 240), (154, 250, 0), (128, 0, 0),
#     (30, 105, 210), (0, 165, 255), (170, 178, 32), (238, 104, 123)]

# g_colors = [ (250,0,0), (255,0, 0 ), (0, 255, 127), (209, 206, 0), (0, 0, 128), (153, 50, 204), (60, 20, 220),
#     (0, 128, 0), (180, 130, 70), (147, 20, 255), (128, 128, 240), (154, 250, 0), (128, 0, 0),
#     (30, 105, 210), (0, 165, 255), (170, 178, 32), (238, 104, 123)]


# GT visualization (by red)
g_colors = [(255, 0, 0), (0, 255, 127), (170, 170, 0), (0, 0, 128), (153, 50, 204), (60, 20, 220),
            (0, 128, 0), (180, 130, 70), (147, 20, 255), (128, 128, 240), (154, 250, 0), (128, 0, 0),
            (30, 105, 210), (0, 165, 255), (170, 178, 32), (238, 104, 123)]

# Custom (non brl) originalPanoptic data
g_colors = [(0, 255, 127), (255, 0, 0), (170, 170, 0), (0, 0, 128), (153, 50, 204), (60, 20, 220),
            (0, 128, 0), (180, 130, 70), (147, 20, 255), (128, 128, 240), (154, 250, 0), (128, 0, 0),
            (30, 105, 210), (0, 165, 255), (170, 178, 32), (238, 104, 123)]

# RGB order!!
g_colors = [(0, 0, 255), (255, 0, 0), (0, 255, 127), (170, 170, 0), (0, 0, 128), (153, 50, 204), (60, 20, 220),
            (0, 128, 0), (180, 130, 70), (147, 20, 255), (128, 128, 240), (154, 250, 0), (128, 0, 0),
            (30, 105, 210), (0, 165, 255), (170, 178, 32), (238, 104, 123)]

g_colorSet = {}  # RGB
g_colorSet['eft'] = (int(0.53 * 255), int(0.53 * 255), int(0.8 * 255))  # bluish color
g_colorSet['spin'] = (int(0.7 * 255), int(0.5 * 255), int(0.5 * 255))  # SPIN color
g_colorSet['hand'] = (30, 178, 166)  # Hand + Body (cyan like)

# g_meshColor = (0.4, 0.4, 0.7) #Blue Default (R,G,B)
# g_meshColor = (0.53, 0.53, 0.8)   #prediction: blue
# g_meshColor = (30/255.0, 178/255.0, 166/255.0)   #hand+body

# g_meshColor = (0.4, 0.4, 0.7) #Blue Default (R,G,B)
g_meshColor = (0.53, 0.53, 0.8)  # prediction: blue
# g_meshColor = (30/255.0, 178/255.0, 166/255.0)   #hand+body

# g_colors = [(0, 255, 127), (255,0,0), (170, 170, 0)  , (0, 0, 128), (153, 50, 204), (60, 20, 220),
#      (0, 128, 0), (180, 130, 70), (147, 20, 255), (128, 128, 240), (154, 250, 0), (128, 0, 0),
#      (30, 105, 210), (0, 165, 255), (170, 178, 32), (238, 104, 123)]

import timeit

#######################################v
# Parametric Mesh Models
g_faceModel = None


########################################################3
# Opengl Setting
def init():
    # global width
    # global height

    glClearColor(1.0, 1.0, 1.0, 1.0)
    # Enable depth testing
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)
    # glShadeModel(GL_FLAT)
    glShadeModel(GL_SMOOTH)
    glEnable(GL_LIGHTING)
    glLightfv(GL_LIGHT0, GL_AMBIENT, g_ambientLight)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, g_diffuseLight)
    glLightfv(GL_LIGHT0, GL_SPECULAR, g_specular)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)
    glMaterialfv(GL_FRONT, GL_SPECULAR, g_specref)
    glMateriali(GL_FRONT, GL_SHININESS, 128)

    # # #Mesh Rendering
    global g_vao
    global g_vertex_buffer
    global g_normal_buffer
    global g_tangent_buffer
    global g_index_buffer

    g_vao = glGenVertexArrays(1)
    # glBindVertexArray(g_vao)

    g_vertex_buffer = glGenBuffers(40)
    # #glBindBuffer(GL_ARRAY_BUFFER, g_vertex_buffer)

    g_normal_buffer = glGenBuffers(40)
    # #glBindBuffer(GL_ARRAY_BUFFER, g_normal_buffer)

    g_tangent_buffer = glGenBuffers(40)
    # #glBindBuffer(GL_ARRAY_BUFFER, g_tangent_buffer)

    g_index_buffer = glGenBuffers(40)
    # # #glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_index_buffer)

    # Create Background Texture
    global g_backgroundTextureID
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)  # So that texture doesnt have to be power of 2
    g_backgroundTextureID = glGenTextures(1)

    glBindTexture(GL_TEXTURE_2D, g_backgroundTextureID)
    # glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, 1920, 1080, 0, GL_RGB, GL_FLOAT, 0)
    # glTexImage2D(GL_TEXTURE_2D, 0,GL_RGB, width, height, 0, GL_BGR, GL_UNSIGNED_BYTE, data);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)


def init_minimum():
    # global width
    # global height

    glClearColor(1.0, 1.0, 1.0, 1.0)
    # Enable depth testing
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glLightfv(GL_LIGHT0, GL_AMBIENT, g_ambientLight)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, g_diffuseLight)
    glLightfv(GL_LIGHT0, GL_SPECULAR, g_specular)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)
    glMaterialfv(GL_FRONT, GL_SPECULAR, g_specref)
    glMateriali(GL_FRONT, GL_SHININESS, 128)


def RenderDomeFloor():
    glPolygonMode(GL_FRONT, GL_FILL)
    glPolygonMode(GL_BACK, GL_FILL)
    # glPolygonMode(GL_FRONT, GL_FILL)
    gridNum = 50
    width = 500
    halfWidth = width / 2
    # g_floorCenter = np.array([0,0.5,0])

    # g_floorCenter = np.array([0,500,0])
    g_floorCenter = np.array([0, 1500, 0])
    g_floorAxis1 = np.array([1, 0, 0])
    g_floorAxis2 = np.array([0, 0, 1])

    origin = g_floorCenter - g_floorAxis1 * (width * gridNum / 2) - g_floorAxis2 * (width * gridNum / 2)
    axis1 = g_floorAxis1 * width
    axis2 = g_floorAxis2 * width
    for y in range(gridNum + 1):
        for x in range(gridNum + 1):

            if (x + y) % 2 == 0:
                glColor(1.0, 1.0, 1.0, 1.0)  # white
            else:
                # glColor(0.95,0.95,0.95,0.3) #grey
                glColor(0.3, 0.3, 0.3, 0.5)  # grey

            p1 = origin + axis1 * x + axis2 * y
            p2 = p1 + axis1
            p3 = p1 + axis2
            p4 = p1 + axis1 + axis2

            glBegin(GL_QUADS)
            glVertex3f(p1[0], p1[1], p1[2])
            glVertex3f(p2[0], p2[1], p2[2])
            glVertex3f(p4[0], p4[1], p4[2])
            glVertex3f(p3[0], p3[1], p3[2])
            glEnd()


def setFree3DView():
    glTranslatef(0, 0, g_zoom)

    glRotatef(-g_yRotate, 1.0, 0.0, 0.0)
    glRotatef(-g_xRotate, 0.0, 1.0, 0.0)

    glRotatef(g_zRotate, 0.0, 0.0, 1.0)
    glTranslatef(g_xTrans, 0.0, 0.0)
    glTranslatef(0.0, g_yTrans, 0.0)
    glTranslatef(0.0, 0, g_zTrans)


# g_hdCams = None
# def load_panoptic_cameras():
#     global g_hdCams
#     with open('/media/posefs3b/Users/xiu/domedb/171204_pose3/calibration_171204_pose3.json') as f:
#         rawCalibs = json.load(f)
#     cameras = rawCalibs['cameras']
#     allPanel = map(lambda x:x['panel'],cameras)
#     hdCamIndices = [i for i,x in enumerate(allPanel) if x==0]
#     g_hdCams = [cameras[i] for i in hdCamIndices]

# def setPanopticCameraView(camid):
#     if g_hdCams==None:
#         load_cameras()

#     if camid>=len(g_hdCams):
#         camid = 0
#     cam = g_hdCams[camid]
#     invR = np.array(cam['R'])
#     invT = np.array(cam['t'])
#     camMatrix = np.hstack((invR, invT))
#     # denotes camera matrix, [R|t]
#     camMatrix = np.vstack((camMatrix, [0, 0, 0, 1]))
#     #camMatrix = numpy.linalg.inv(camMatrix)
#     K = np.array(cam['K'])
#     #K = K.flatten()
#     glMatrixMode(GL_PROJECTION)
#     glLoadIdentity()
#     Kscale = 1920.0/g_Width
#     K = K/Kscale
#     ProjM = np.zeros((4,4))
#     ProjM[0,0] = 2*K[0,0]/g_Width
#     ProjM[0,2] = (g_Width - 2*K[0,2])/g_Width
#     ProjM[1,1] = 2*K[1,1]/g_Height
#     ProjM[1,2] = (-g_Height+2*K[1,2])/g_Height

#     ProjM[2,2] = (-g_farPlane-g_nearPlane)/(g_farPlane-g_nearPlane)
#     ProjM[2,3] = -2*g_farPlane*g_nearPlane/(g_farPlane-g_nearPlane)
#     ProjM[3,2] = -1

#     glLoadMatrixd(ProjM.T)
#     glMatrixMode(GL_MODELVIEW)
#     glLoadIdentity()
#     gluLookAt(0, 0, 0, 0, 0, 1, 0, -1, 0)
#     glMultMatrixd(camMatrix.T)


# def load_MTC_default_camera():
#     global g_Width,  g_Height

#     camRender_width = 1920
#     camRender_height = 1080

#     g_Width = camRender_width
#     g_Height = camRender_height


# 3x3 intrinsic camera matrix
def setCamView_K(K):
    global g_camView_K
    g_camView_K = K


# 3x3 intrinsic camera matrix
# Set a default camera matrix used for MTC
def setCamView_K_DefaultForMTC():
    global g_camView_K

    K = np.array([[2000, 0, 960], [0, 2000, 540], [0, 0, 1]])  # MTC default camera. for 1920 x 1080 input image
    g_camView_K = K


# Show the world in a camera cooridnate (defined by K)
def setCameraView():
    # camRender_width = 1920
    # camRender_height = 1080

    # global g_Width,  g_Height
    # if camRender_width != g_Width or  camRender_height!=g_Height:
    #     g_Width = camRender_width
    #     g_Height = camRender_height
    #     #reshape(g_Width, g_Height)
    #     glutReshapeWindow(g_Width,g_Height)

    # invR = np.array(cam['R'])
    # invT = np.array(cam['t'])
    invR = np.eye(3)
    invT = np.zeros((3, 1))
    # invT[2] = 400
    camMatrix = np.hstack((invR, invT))
    # denotes camera matrix, [R|t]
    camMatrix = np.vstack((camMatrix, [0, 0, 0, 1]))
    # camMatrix = numpy.linalg.inv(camMatrix)
    # K = np.array(cam['K'])
    # K = np.array(cam['K'])
    # K = np.array([[2000, 0, 960],[0, 2000, 540],[0,0,1]])       #MTC default camera
    # global g_camView_K
    # g_camView_K = K
    # K = K.flatten()

    if g_camView_K is None:
        print("## Warning: no K is set, so I use a default cam parameter defined for MTC")
        setCamView_K_DefaultForMTC()
    K = g_camView_K.copy()

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    # Kscale = 1920.0/g_Width
    Kscale = 1.0  # 1920.0/g_Width        :: why do we need this?
    K = K / Kscale
    ProjM = np.zeros((4, 4))
    ProjM[0, 0] = 2 * K[0, 0] / g_Width
    ProjM[0, 2] = (g_Width - 2 * K[0, 2]) / g_Width
    ProjM[1, 1] = 2 * K[1, 1] / g_Height
    ProjM[1, 2] = (-g_Height + 2 * K[1, 2]) / g_Height

    ProjM[2, 2] = (-g_farPlane - g_nearPlane) / (g_farPlane - g_nearPlane)
    ProjM[2, 3] = -2 * g_farPlane * g_nearPlane / (g_farPlane - g_nearPlane)
    ProjM[3, 2] = -1

    glLoadMatrixd(ProjM.T)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(0, 0, 0, 0, 0, 1, 0, -1, 0)
    glMultMatrixd(camMatrix.T)


def SetOrthoCamera(bOrtho=True):
    global g_bOrthoCam
    g_bOrthoCam = bOrtho


# Show the world in a camera cooridnate (defined by K)
def setCameraViewOrth():
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    texHeight, texWidth = g_textureData.shape[:2]
    # texHeight,texWidth =   1024, 1024
    texHeight *= 0.5
    texWidth *= 0.5
    # texHeight *=BACKGROUND_IMAGE_PLANE_DEPTH
    # texWidth *=BACKGROUND_IMAGE_PLANE_DEPTH

    glOrtho(-texWidth, texWidth, -texHeight, texHeight, -1500, 1500)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(0, 0, 0, 0, 0, 1, 0, -1, 0)
    # glMultMatrixd(camMatrix.T)


def setRenderOutputSize(imWidth, imHeight):
    global g_renderOutputSize
    g_renderOutputSize = (imWidth, imHeight)


def setWindowSize(new_width, new_height):
    global g_Width, g_Height

    if new_height > 1600:  # Max height of screen
        new_width = int(new_width * 0.7)
        new_height = int(new_height * 0.7)

    if new_width != g_Width or new_height != g_Height:
        g_Width = new_width
        g_Height = new_height
        # reshape(g_Width, g_Height)

        if g_bGlInitDone:
            glutReshapeWindow(g_Width, g_Height)


def reshape(width, height):
    # lightPos = (-50.0, 50.0, 100.0, 1.0)
    nRange = 250.0
    global g_Width, g_Height
    g_Width = width
    g_Height = height
    glViewport(0, 0, g_Width, g_Height)

    # # Set perspective (also zoom)
    # glMatrixMode(GL_PROJECTION)
    # glLoadIdentity()
    # #gluPerspective(zoom, float(g_Width)/float(g_Height), g_nearPlane, g_farPlane)
    # gluPerspective(65, float(g_Width)/float(g_Height), g_nearPlane, g_farPlane)
    # print("here: {}".format(float(g_Width)/float(g_Height)))


def SaveScenesToFile():
    global g_saveImageName_last
    # global g_Width, g_Height, g_bSaveToFile, g_fameIdx, g_hagglingseq_name
    # global g_bSaveToFile
    # global g_saveFrameIdx

    glReadBuffer(GL_FRONT)
    img = glReadPixels(0, 0, g_Width, g_Height, GL_RGBA, GL_UNSIGNED_BYTE, outputType=None)
    img = Image.frombytes("RGBA", (g_Width, g_Height), img)
    img = ImageOps.flip(img)

    if False:
        pix = np.array(img)
        from renderer import viewer2D
        viewer2D.ImShow(pix)

    # Crop and keep the original size
    # if g_textureImgOriginal is not None and g_viewMode=='camView':
    #     width, height = img.size
    #     img = img.crop( (0,0,g_textureImgOriginal.shape[1], g_textureImgOriginal.shape[0]))     #top, left, bottom, right where origin seems top left

    if g_renderOutputSize is not None and g_viewMode == 'camView':
        width, height = g_renderOutputSize
        img = img.crop((0, 0, width, height))  # top, left, bottom, right where origin seems top left

    # img.save('/ssd/render_general/frame_{}.png'.format(g_frameIdx), 'PNG')
    if g_saveFolderName is not None:
        folderPath = g_saveFolderName
    else:  # g_haggling_render == False:
        # folderPath = '/home/hjoo//temp/render_general/'
        folderPath = '/home/hjoo//temp/render_general/'
    # else:
    #     if g_meshes == None and g_hagglingFileList == None: #Initial loading
    #         LoadHagglingData_Caller() #Load New Scene
    #     folderPath = '/hjoo/home/temp/render_mesh/' + g_hagglingseq_name
    #     if os.path.exists(folderPath) == False:
    #         os.mkdir(folderPath)

    if os.path.exists(folderPath) == False:
        os.mkdir(folderPath)
    # img.save('{0}/scene_{1:08d}.png'.format(folderPath,g_saveFrameIdx), 'PNG')
    img = img.convert("RGB")

    if g_saveImageName is not None:
        g_saveImageName_last = '{0}/{1}.jpg'.format(folderPath, g_saveImageName)
    else:
        g_saveImageName_last = '{0}/scene_{1:08d}.jpg'.format(folderPath, g_saveFrameIdx)

    img.save(g_saveImageName_last, "JPEG")

    print(f"Render image to:{g_saveImageName_last}")

    # print(img.size)
    # g_saveFrameIdx +=1
    # Done saving
    global g_bSaveToFile_done
    g_bSaveToFile_done = True

    # global g_savedImg = img     #save rendered one into gobal variable

    # if g_frameIdx+1 >= g_frameLimit:
    #     if g_haggling_render == False:
    #         g_bSaveToFile = False
    #     else:
    #         bLoaded = LoadHagglingData_Caller() #Load New Scene
    #         if bLoaded ==False:
    #             g_bSaveToFile = False

    # cur_ind += 1
    # print(cur_ind)

    # glutPostRedisplay()


def SaveCamViewInfo():
    global g_Width, g_Height
    global g_nearPlane, g_farPlane
    global g_zoom, g_yRotate, g_xRotate, g_zRotate, g_xTrans, g_yTrans, g_zTrans

    fileName = '/ssd/camInfo/camInfo.pkl'
    if os.path.exists('/ssd/camInfo/') == False:
        os.mkdir('/ssd/camInfo/')

    if os.path.exists('/ssd/camInfo/archive/') == False:
        os.mkdir('/ssd/camInfo/archive/')

    if os.path.exists(fileName):
        # resave it
        for i in range(1000):
            newName = '/ssd/camInfo/archive/camInfo_old{}.pkl'.format(i)
            if os.path.exists(newName) == False:
                import shutil
                shutil.copy2(fileName, newName)
                break

    camInfo = dict()
    camInfo['g_Width'] = g_Width
    camInfo['g_Height'] = g_Height
    camInfo['g_nearPlane'] = g_nearPlane
    camInfo['g_farPlane'] = g_farPlane
    camInfo['g_zoom'] = g_zoom
    camInfo['g_yRotate'] = g_yRotate
    camInfo['g_xRotate'] = g_xRotate
    camInfo['g_zRotate'] = g_zRotate
    camInfo['g_xTrans'] = g_xTrans
    camInfo['g_yTrans'] = g_yTrans

    pickle.dump(camInfo, open(fileName, "wb"))

    print('camInfo')


def LoadCamViewInfo():
    global g_Width, g_Height
    global g_nearPlane, g_farPlane
    global g_zoom, g_yRotate, g_xRotate, g_zRotate, g_xTrans, g_yTrans
    global g_camView_fileName
    fileName = g_camView_fileName
    if not os.path.exists(fileName):
        print("No cam info: {}".format(fileName))
        return

    camInfo = pickle.load(open(fileName, "rb"), encoding='latin1')
    g_yTrans = camInfo['g_Width']
    g_Height = camInfo['g_Height']
    g_nearPlane = camInfo['g_nearPlane']
    g_farPlane = camInfo['g_farPlane']
    g_zoom = camInfo['g_zoom']
    g_yRotate = camInfo['g_yRotate']
    g_xRotate = camInfo['g_xRotate']
    g_zRotate = camInfo['g_zRotate']
    g_xTrans = camInfo['g_xTrans']
    g_yTrans = camInfo['g_yTrans']

    reshape(g_Width, g_Height)


def PuttingObjectCenter():
    global g_zoom, g_xTrans, g_yTrans, g_zTrans
    global g_xRotate, g_yRotate, g_zRotate
    if (g_skeletons is not None) and len(g_skeletons) > 0:

        g_xRotate = 0
        g_yRotate = 0
        g_zRotate = 0

        # if g_skeletonType == 'smplcoco':
        #     g_xTrans = -(g_skeletons[0][9,0] + g_skeletons[0][12,0] )*0.5
        #     g_yTrans = -(g_skeletons[0][10,0] + g_skeletons[0][13,0] )*0.5
        #     g_zTrans = -(g_skeletons[0][11,0] + g_skeletons[0][14,0] )*0.5
        #     g_zoom = 300
        # else: #Adam
        g_xTrans = -g_skeletons[0]['skeleton'][3 * 39]
        g_yTrans = -g_skeletons[0]['skeleton'][3 * 39 + 1]  # 0#100
        g_zTrans = -g_skeletons[0]['skeleton'][3 * 39 + 2]
        g_zoom = 300

    elif g_meshes is not None and len(g_meshes) > 0:

        g_xRotate = 0
        g_yRotate = 0
        g_zRotate = 0

        g_xTrans = -g_meshes[0]['ver'][0, 1767, 0]
        g_yTrans = -g_meshes[0]['ver'][0, 1767, 1]
        g_zTrans = -g_meshes[0]['ver'][0, 1767, 2]

        # print("{} {} {}".format(g_xTrans, g_yTrans, g_zTrans))
        g_zoom = 300


def keyboard(key, x, y):
    global g_stopMainLoop, g_frameIdx
    global g_ptSize
    global g_xRotate, g_yRotate, g_zRotate

    if isinstance(key, bytes):
        key = key.decode()  # Python3: b'X' -> 'X' (bytes -> str)
    if key == chr(27) or key == 'q':
        # sys.exit()
        g_stopMainLoop = True
        g_frameIdx = 0
        # glutIdleFunc(0); # Turn off Idle function if used.
        # glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,GLUT_ACTION_CONTINUE_EXECUTION)
        # glutLeaveMainLoop()
        # glutDestroyWindow(g_winID) # Close open windows
    elif key == 'p':
        # global width, height
        glReadBuffer(GL_FRONT)
        img = glReadPixels(0, 0, g_Width, g_Height, GL_RGBA, GL_UNSIGNED_BYTE, outputType=None)
        img = Image.frombytes("RGBA", (g_Width, g_Height), img)
        img = ImageOps.flip(img)
        img.save('temp.jpg', 'JPG')

    elif key == 's':
        g_frameIdx = 0


    elif key == 't':

        g_xRotate = 0
        g_yRotate = -90
        g_zRotate = 0
        print('showTopView')

    elif key == 'r':
        global g_bSaveToFile
        g_frameIdx = 0
        g_bSaveToFile = not g_bSaveToFile

    elif key == 'w':
        global g_bShowWiredMesh
        g_bShowWiredMesh = not g_bShowWiredMesh

    elif key == 'o':
        global g_bApplyRootOffset
        g_bApplyRootOffset = not g_bApplyRootOffset

    elif key == 'f':
        global g_bShowFloor
        g_bShowFloor = not g_bShowFloor

    elif key == 'V':
        SaveCamViewInfo()

    elif key == 'v':
        LoadCamViewInfo()

    # elif key =='c':
    #     g_xRotate =0
    #     g_yRotate =0
    #     g_zRotate =180

    elif key == 'c':  # put the target human in the center
        PuttingObjectCenter()
    elif key == '+':
        g_ptSize += 1
    elif key == '-':
        # global g_ptSize
        if g_ptSize >= 2:
            g_ptSize -= 1
    elif key == 'R':  # rotate cameras
        global g_bRotateView, g_rotateView_counter

        g_bRotateView = not g_bRotateView
        g_rotateView_counter = 0

    elif key == 'j':  # Toggle joint
        global g_bShowSkeleton
        g_bShowSkeleton = not g_bShowSkeleton
    elif key == 'm':
        global g_bShowMesh
        g_bShowMesh = not g_bShowMesh
    elif key == 'b':
        global g_bShowBackground
        g_bShowBackground = not g_bShowBackground

    # elif key =='C':
    #     g_xRotate =0
    #     g_yRotate =0
    #     g_zRotate =0
    elif key == 'C':
        print('Toggle camview / freeview')
        global g_viewMode, g_nearPlane
        if g_viewMode == 'free':
            g_viewMode = 'camView'
            g_nearPlane = 500  # cameramode
        else:
            g_viewMode = 'free'
            g_nearPlane = 0.01  # original


    elif key == 'S':
        global g_zoom, g_xTrans, g_yTrans
        g_xTrans = -4.126092433929443
        g_yTrans = 12
        # g_zoom= 190
        g_zoom = 1028
        # g_xRotate = 90
        g_xRotate = -58
        # g_yRotate= 0
        g_yRotate = 9
        g_zRotate = 0.0
        g_viewMode = 'free'

    # elif key>='0' and key<='9':
    #     g_camid.popleft()
    #     g_camid.append(key)
    #     print('camView: CamID:{}'.format(g_camid))

    elif key == '0':
        glutReshapeWindow(1920, 720)

    glutPostRedisplay()


def mouse(button, state, x, y):
    global g_action, g_xMousePtStart, g_yMousePtStart
    if (button == GLUT_LEFT_BUTTON):
        if (glutGetModifiers() == GLUT_ACTIVE_SHIFT):
            g_action = "TRANS"
        else:
            g_action = "MOVE_EYE"
    # elif (button==GLUT_MIDDLE_BUTTON):
    #    action = "TRANS"
    elif (button == GLUT_RIGHT_BUTTON):
        g_action = "ZOOM"
    g_xMousePtStart = x
    g_yMousePtStart = y


def motion(x, y):
    global g_zoom, g_xMousePtStart, g_yMousePtStart, g_xRotate, g_yRotate, g_zRotate, g_xTrans, g_zTrans
    if (g_action == "MOVE_EYE"):
        g_xRotate += x - g_xMousePtStart
        g_yRotate -= y - g_yMousePtStart
    elif (g_action == "MOVE_EYE_2"):
        g_zRotate += y - g_yMousePtStart
    elif (g_action == "TRANS"):
        g_xTrans += x - g_xMousePtStart
        g_zTrans += y - g_yMousePtStart
    elif (g_action == "ZOOM"):
        g_zoom -= y - g_yMousePtStart
        # print(g_zoom)
    else:
        print("unknown action\n", g_action)
    g_xMousePtStart = x
    g_yMousePtStart = y

    # print ('xTrans {},  yTrans {}, zoom {} xRotate{} yRotate {} zRotate {}'.format(g_xTrans,  g_yTrans,  g_zoom,  g_xRotate,  g_yRotate,  g_zRotate))
    glutPostRedisplay()


def setBackgroundTexture(img):
    global g_textureData  # , g_textureImgOriginal
    g_textureData = img

    # In MTC, the background should be always 1920x1080
    # g_textureData = np.ones( (1080, 1920, 3), dtype=img.dtype)*0     #dtype==np.unit8
    # g_textureData[:img.shape[0],:img.shape[1] ] = img
    # g_textureImgOriginal = img  #keep the original image

    # import cv2
    # cv2.imshow('here??',img)
    # cv2.waitKey(0)


def SetCameraPoses(camRots, camPoses):
    global g_cameraPoses, g_cameraRots

    # g_cameraPoses = camPoses
    # g_cameraRots = camRots

    g_cameraPoses = camPoses  # for cam Vis
    g_cameraRots = []  # for cam Vis

    for r in camRots:
        cam_extR_4x4 = np.eye(4, dtype=r.dtype)
        # cam_extR_4x4[:3,:3] = cam_R_rot.transpose()
        cam_extR_4x4[:3, :3] = r.transpose()  # For visualizing cameras, R should be inversed
        cam_extR_4x4[3, 3] = 1.0
        g_cameraRots.append(cam_extR_4x4)


def SetPtCloud(ptCloud, ptCloudColor=None):
    global g_ptCloud, g_ptCloudColor
    g_ptCloud = ptCloud
    g_ptCloudColor = ptCloudColor


def DrawBackgroundOrth():
    if g_textureData is None:
        return

    glDisable(GL_CULL_FACE)
    glDisable(GL_DEPTH_TEST)
    glDisable(GL_LIGHTING)
    glEnable(GL_TEXTURE_2D)

    # glUseProgram(0)

    glBindTexture(GL_TEXTURE_2D, g_backgroundTextureID)
    # glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1920, 1080, 0, GL_RGB, GL_UNSIGNED_BYTE, g_textureData)
    # glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 480, 640, 0, GL_RGB, GL_UNSIGNED_BYTE, g_textureData.data)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, g_textureData.shape[1], g_textureData.shape[0], 0, GL_BGR, GL_UNSIGNED_BYTE,
                 g_textureData.data)
    texHeight, texWidth = g_textureData.shape[:2]
    texHeight *= 0.5
    texWidth *= 0.5

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    glBegin(GL_QUADS)
    glColor3f(1.0, 1.0, 1.0)
    # d = BACKGROUND_IMAGE_PLANE_DEPTH
    d = 10

    glTexCoord2f(0, 0)
    # #Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> K(m_options.m_pK);
    P = np.array([-texWidth, -texHeight, d])
    # P = np.matmul(K_inv,P)
    # P = P / P[2]
    glVertex3f(P[0], P[1], P[2]);  # K^{-1} [0, 0, 1]^T

    glTexCoord2f(1, 0)
    # P = [1920, 0, 1]
    P = [texWidth, -texHeight, d]
    glVertex3f(P[0], P[1], P[2]);  # K^{-1} [0, 0, 1]^T

    glTexCoord2f(1, 1)
    # P = [1920, 1080, 1]
    P = [texWidth, texHeight, d]
    # P = np.matmul(K_inv,P)
    # P = P / P[2]
    # glVertex3f(P[0] * d, P[1] * d, P[2] * d)
    glVertex3f(P[0], P[1], P[2]);  # K^{-1} [0, 0, 1]^T

    glTexCoord2f(0, 1)
    # P = [0, 1080, 1]
    P = [-texWidth, texHeight, d]
    # glVertex3f(P[0] * d, P[1] * d, P[2] * d)
    glVertex3f(P[0], P[1], P[2]);  # K^{-1} [0, 0, 1]^T
    glEnd()

    glEnable(GL_LIGHTING)
    glEnable(GL_CULL_FACE)
    glEnable(GL_DEPTH_TEST)
    glDisable(GL_TEXTURE_2D)


def DrawBackground():
    if g_camView_K is None or g_textureData is None:
        return

    K_inv = np.linalg.inv(g_camView_K)

    glDisable(GL_CULL_FACE)
    glDisable(GL_DEPTH_TEST)
    glDisable(GL_LIGHTING)
    glEnable(GL_TEXTURE_2D)

    # glUseProgram(0)

    glBindTexture(GL_TEXTURE_2D, g_backgroundTextureID)
    # glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1920, 1080, 0, GL_RGB, GL_UNSIGNED_BYTE, g_textureData)
    # glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 480, 640, 0, GL_RGB, GL_UNSIGNED_BYTE, g_textureData.data)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, g_textureData.shape[1], g_textureData.shape[0], 0, GL_BGR, GL_UNSIGNED_BYTE,
                 g_textureData.data)
    texHeight, texWidth = g_textureData.shape[:2]

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    glBegin(GL_QUADS)
    glColor3f(1.0, 1.0, 1.0)
    d = BACKGROUND_IMAGE_PLANE_DEPTH

    glTexCoord2f(0, 0)
    # #Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> K(m_options.m_pK);
    P = np.array([0, 0, 1])
    P = np.matmul(K_inv, P)
    P = P / P[2]
    glVertex3f(P[0] * d, P[1] * d, P[2] * d);  # K^{-1} [0, 0, 1]^T

    glTexCoord2f(1, 0)
    # P = [1920, 0, 1]
    P = [texWidth, 0, 1]

    P = np.matmul(K_inv, P)
    P = P / P[2]
    glVertex3f(P[0] * d, P[1] * d, P[2] * d);  # K^{-1} [0, 0, 1]^T

    glTexCoord2f(1, 1)
    # P = [1920, 1080, 1]
    P = [texWidth, texHeight, 1]
    P = np.matmul(K_inv, P)
    P = P / P[2]
    glVertex3f(P[0] * d, P[1] * d, P[2] * d)

    glTexCoord2f(0, 1)
    # P = [0, 1080, 1]
    P = [0, texHeight, 1]
    P = np.matmul(K_inv, P)
    P = P / P[2]
    glVertex3f(P[0] * d, P[1] * d, P[2] * d)
    glEnd()

    glEnable(GL_LIGHTING)
    glEnable(GL_CULL_FACE)
    glEnable(GL_DEPTH_TEST)
    glDisable(GL_TEXTURE_2D)


def specialkeys(key, x, y):
    global g_xrot
    global g_yrot
    global g_cur_ind
    global g_stopMainLoop
    global g_backwardsDirection
    if key == GLUT_KEY_UP:
        g_xrot -= 2.0
    if key == GLUT_KEY_DOWN:
        g_xrot += 2.0
    if key == GLUT_KEY_RIGHT:
        g_stopMainLoop = True
        g_backwardsDirection = False
    if key == GLUT_KEY_LEFT:
        g_stopMainLoop = True
        g_backwardsDirection = True
    glutPostRedisplay()


def backwardsDirection():
    return g_backwardsDirection


def init_gl_util():
    global g_bGlInitDone, g_lastframetime, g_currenttime, g_fps

    g_lastframetime = g_currenttime
    g_currenttime = time.time()
    refresh_fps = 0.15
    g_fps = (1 - refresh_fps) * g_fps + refresh_fps * 1 / (g_currenttime - g_lastframetime)

    if g_bGlInitDone == False:
        glutInit()
        glutInitDisplayMode(
            GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_MULTISAMPLE)  # GLUT_MULTISAMPLE is required for anti-aliasing
        # glutInitDisplayMode(GLUT_RGB |GLUT_DOUBLE|GLUT_DEPTH)
        glutInitWindowPosition(100, 100)
        glutInitWindowSize(g_Width, g_Height)

        global g_winID
        g_winID = glutCreateWindow("3D View")
        init()
        # init_minimum()
        glutReshapeFunc(reshape)
        glutDisplayFunc(renderscene)
        glutKeyboardFunc(keyboard)
        glutMouseFunc(mouse)
        glutMotionFunc(motion)
        glutSpecialFunc(specialkeys)
        # glutIdleFunc(idlefunc)
        glutIdleFunc(renderscene)

        # Ver 1: Infinite loop (termination is not possible)
        # glutMainLoop()

        # Ver 2: for better termination (by pressing 'q')
        g_bGlInitDone = True
    else:
        glutReshapeWindow(g_Width, g_Height)  # Just doing resize
        # glutReshapeWindow(int(g_Width*0.5), int(g_Height*0.5))     #Just doing resize


def init_gl(maxIter=-10):
    # Init_Haggling()
    # global width
    # global height
    # Setup for double-buffered display and depth testing
    init_gl_util()

    global g_stopMainLoop
    g_stopMainLoop = False
    while True:
        glutPostRedisplay()
        if bool(glutMainLoopEvent) == False:
            continue
        glutMainLoopEvent()
        if g_stopMainLoop:
            break
        if maxIter > 0:
            maxIter -= 1
            if maxIter <= 0:
                g_stopMainLoop = True

    # print("Escaped glut loop")


# g_faces should be: #(faceNum, faceDim, faceFrames)
def DrawFaces():
    # global g_colors
    # global g_faces, g_faceNormals, g_frameIdx#, g_normals

    if g_faces is None:
        return

    # g_frameLimit = g_faces.shape[2]

    for humanIdx in range(len(g_faces)):

        if (g_frameIdx >= g_faces[humanIdx].shape[1]):
            continue

        if g_onlyDrawHumanIdx >= 0 and humanIdx != g_onlyDrawHumanIdx:
            continue

        face3D = g_faces[humanIdx][:, g_frameIdx]  # 210x1
        drawface_70(face3D, g_colors[humanIdx % len(g_colors)])

        if g_faceNormals is not None and len(g_faceNormals) > humanIdx:

            if g_faceNormals[humanIdx].shape[1] <= g_frameIdx:
                print("Warning: g_faceNormals[humanIdx].shape[2]<=g_frameId")
                continue

            normal3D = g_faceNormals[humanIdx][:, g_frameIdx]  # 3x1
            # drawfaceNormal_70(normal3D, face3D, g_colors[humanIdx % len(g_colors)])
            # drawfaceNormal_70(normal3D, face3D, [0, 255, 255])
            eyeCenterPoint = 0.5 * (face3D[(45 * 3):(45 * 3 + 3)] + face3D[(36 * 3):(36 * 3 + 3)])
            # drawNormal(normal3D, eyeCenterPoint, [0, 255, 0],normalLength=25)
            drawNormal(normal3D, eyeCenterPoint, [0, 255, 255], normalLength=25)

        # drawbody_joint_ptOnly(face3D, g_colors[humanIdx])
    # g_frameIdx +=1

    # if g_frameIdx>=g_frameLimit:
    #    g_frameIdx =0


# g_faces should be: #(faceNum, faceDim, faceFrames)
def DrawHands():
    # global g_colors
    # global g_hands_left,g_hands_right, g_frameIdx#, g_normals

    # g_frameLimit = g_faces.shape[2]

    if g_hands_left is not None:
        for humanIdx in range(len(g_hands_left)):
            if g_onlyDrawHumanIdx >= 0 and humanIdx != g_onlyDrawHumanIdx:
                continue

            if (g_frameIdx >= g_hands_left[humanIdx].shape[1]):
                continue

            hand3D = g_hands_left[humanIdx][:, g_frameIdx]  # 210x1
            drawhand_21(hand3D, g_colors[humanIdx % len(g_colors)])
            # drawbody_joint_ptOnly(face3D, g_colors[humanIdx])

    if g_hands_right is not None:
        for humanIdx in range(len(g_hands_right)):
            if g_onlyDrawHumanIdx >= 0 and humanIdx != g_onlyDrawHumanIdx:
                continue

            if (g_frameIdx >= g_hands_right[humanIdx].shape[1]):
                continue

            hand3D = g_hands_right[humanIdx][:, g_frameIdx]  # 210x1
            drawhand_21(hand3D, g_colors[humanIdx % len(g_colors)])
            # drawbody_joint_ptOnly(face3D, g_colors[humanIdx])

    # g_frameIdx +=1

    # if g_frameIdx>=g_frameLimit:
    #    g_frameIdx =0


# Face keypoint orders follow Openpose keypoint output
# https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md
# Face outline points (0-16) are unstable
face_edges = np.array([
    # [0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[11,12],[12,13],[14,15],[15,16], #outline (ignored)
    [17, 18], [18, 19], [19, 20], [20, 21],  # right eyebrow
    [22, 23], [23, 24], [24, 25], [25, 26],  # left eyebrow
    [27, 28], [28, 29], [29, 30],  # nose upper part
    [31, 32], [32, 33], [33, 34], [34, 35],  # nose lower part
    [36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [41, 36],  # right eye
    [42, 43], [43, 44], [44, 45], [45, 46], [46, 47], [47, 42],  # left eye
    [48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54], [54, 55], [55, 56], [56, 57], [57, 58], [58, 59],
    [59, 48],
    # Lip outline
    [60, 61], [61, 62], [62, 63], [63, 64], [64, 65], [65, 66], [66, 67], [67, 60]  # Lip inner line
])


# joints70: 3x70 =210 dim
def drawface_70(joints, color):
    glLineWidth(2.0)
    # Visualize Joints
    glColor3ub(color[0], color[1], color[2])
    # for i in range(len(joints)/3):
    #    glPushMatrix()
    #    glTranslate(joints[3*i], joints[3*i+1], joints[3*i+2])
    #    glutSolidSphere(0.5, 10, 10)
    #    glPopMatrix()

    # connMat_coco19 = g_connMat_smc19
    # Visualize Bones
    for conn in face_edges:
        # x0, y0, z0 is the coordinate of the base point
        x0 = joints[3 * conn[0]]
        y0 = joints[3 * conn[0] + 1]
        z0 = joints[3 * conn[0] + 2]

        x1 = joints[3 * conn[1]]
        y1 = joints[3 * conn[1] + 1]
        z1 = joints[3 * conn[1] + 2]

        glBegin(GL_LINES)
        glVertex3f(x0, y0, z0)
        glVertex3f(x1, y1, z1)
        glEnd()


def SetMeshColor(colorName='blue'):
    global g_meshColor
    if colorName == 'blue':
        # g_meshColor = (0.4, 0.4, 0.)   #prediction: blue
        g_meshColor = (0.53, 0.53, 0.8)  # prediction: blue
        # glColor3f(0.53, 0.53, 0.8)
    elif colorName == 'red':
        g_meshColor = (0.7, 0.5, 0.5)  # targer: red
    else:
        assert False


g_firsttime = True
""" With normal"""


def DrawMeshes():
    global g_firsttime
    global g_meshes, g_frameIdx  # , g_normals

    global g_vao
    global g_vertex_buffer
    global g_normal_buffer
    global g_tangent_buffer
    global g_index_buffer

    # MESH_SCALING = 100.0  #from meter (model def) to CM (panoptic def)
    # MESH_SCALING = 1.0  #from meter (model def) to CM (panoptic def)

    if g_meshes is None:
        return

    # g_meshes[humanIdx]['ver']: frames x 6890 x 3
    for humanIdx in range(len(g_meshes)):

        if False:  # humanIdx ==1:        #Second one red
            glColor3f(0.7, 0.5, 0.5)  # targer: spin color
            # glColor3f(0.53, 0.53, 0.8)   #SPIN: red
            # glColor3f(0.4, 0.4, 0.7)   #prediction: blue
        elif 'color' in g_meshes[humanIdx].keys():
            glColor3f(g_meshes[humanIdx]['color'][0] / 255.0, g_meshes[humanIdx]['color'][1] / 255.0,
                      g_meshes[humanIdx]['color'][2] / 255.0)
        else:
            #     glColor3f(0.4, 0.4, 0.7)   #prediction: blue
            glColor3f(g_meshColor[0], g_meshColor[1], g_meshColor[2])
            # glColor3f(0.53, 0.53, 0.8)
        # glColor3f(g_meshColor[0],g_meshColor[1],g_meshColor[2])

        if (g_frameIdx >= g_meshes[humanIdx]['ver'].shape[0]):
            continue

        # if(humanIdx==0 or humanIdx==1):       #Debug
        #     continue

        SMPL_vts = g_meshes[humanIdx]['ver'][g_frameIdx, :, :]  # 6890x3
        SMPL_inds = g_meshes[humanIdx]['f']
        SMPL_vts = SMPL_vts.flatten()  # * MESH_SCALING
        SMPL_vts = SMPL_vts.astype(np.float32)
        SMPL_inds = SMPL_inds.flatten()
        vts_num = int(len(SMPL_vts) / 3)
        face_num = int(len(SMPL_inds) / 3)

        if 'normal' in g_meshes[humanIdx].keys() and g_meshes[humanIdx]['normal'] is None:
            here = 0
        if 'normal' in g_meshes[humanIdx].keys() and len(g_meshes[humanIdx]['normal']) > 0:
            SMPL_normals = g_meshes[humanIdx]['normal'][g_frameIdx, :, :]
            SMPL_normals = SMPL_normals.flatten().astype(np.float32)
            tangent = np.zeros(SMPL_normals.shape)
            tangent.astype(np.float32)
            normal_num = len(SMPL_normals) / 3
        else:
            SMPL_normals = None

        glBindVertexArray(g_vao)

        glEnableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, g_vertex_buffer[humanIdx])
        glBufferData(GL_ARRAY_BUFFER, len(SMPL_vts) * sizeof(ctypes.c_float),
                     (ctypes.c_float * len(SMPL_vts))(*SMPL_vts), GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

        if SMPL_normals is not None:
            # this is not needed.. .but normal should be the third attribute...
            if g_firsttime:
                glEnableVertexAttribArray(1)
                glBindBuffer(GL_ARRAY_BUFFER, g_tangent_buffer[humanIdx])
                glBufferData(GL_ARRAY_BUFFER, len(tangent) * sizeof(ctypes.c_float),
                             (ctypes.c_float * len(tangent))(*tangent), GL_STATIC_DRAW)
                glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)

            glEnableVertexAttribArray(2)
            glBindBuffer(GL_ARRAY_BUFFER, g_normal_buffer[humanIdx])
            glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, None)
            # glBufferData(GL_ARRAY_BUFFER, len(SMPL_normals) * sizeof(ctypes.c_float), (ctypes.c_float * len(SMPL_normals))(*SMPL_normals), GL_STATIC_DRAW)
            # glBufferData(GL_ARRAY_BUFFER, len(SMPL_normals) * sizeof(ctypes.c_float), (ctypes.c_float * len(SMPL_normals))(*SMPL_normals), GL_STATIC_DRAW)
            glBufferData(GL_ARRAY_BUFFER,
                         len(SMPL_normals) * sizeof(ctypes.c_float),
                         SMPL_normals,
                         GL_STATIC_DRAW)

        if True:  # g_firsttime:
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_index_buffer[humanIdx])
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(ctypes.c_uint) * len(SMPL_inds),
                         (ctypes.c_uint * len(SMPL_inds))(*SMPL_inds), GL_STATIC_DRAW)

        g_firsttime = False
        # set the dimensions of the position attribute, so it consumes 2 floats at a time (default is 4)

        # Draw by vertex array
        glPushMatrix()

        # if humanIdx ==0:
        #     #glColor3f(0.5, 0.2, 0.2)   #targer: red
        #     glColor3f(0.4, 0.4, 0.7)   #prediction: blue
        # elif humanIdx ==1:
        #     glColor3f(0.5, 0.2, 0.2)   #targer: red
        #     #glColor3f(0.0, 0.8, 0.3)   #buyer
        # elif humanIdx ==2:
        #     glColor3f(0.0, 0.8, 0.3)   #buyer
        #     #glColor3f(0.5, 0.5, 0)   #another: yellow
        # else:
        # #elif humanIdx ==2:
        #     glColor3f(0.5, 0.5, 0)   #another: yellow
        # glColor3f(0.6, 0.6, 0.4)   #another: yellow

        # glColor3f(0.8, 0.8, 0.8)
        glLineWidth(.5)

        if SMPL_normals is not None and g_bShowWiredMesh == False:
            glPolygonMode(GL_FRONT, GL_FILL)
            glPolygonMode(GL_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT, GL_LINE)
            glPolygonMode(GL_BACK, GL_LINE)
        # for vao_object in g_vao_object_list:
        #     glBindVertexArray(vao_object)
        #     glDrawElements(GL_TRIANGLES, face_num * 3, GL_UNSIGNED_INT, None)
        if g_bApplyRootOffset:
            # glTranslatef(40*humanIdx,0,0)
            glTranslatef(ROOT_OFFSET_DIST * humanIdx, 0, 0)

        glDrawElements(GL_TRIANGLES, face_num * 3, GL_UNSIGNED_INT, None)
        # glPolygonMode(GL_FRONT, GL_FILL)
        # glPolygonMode(GL_BACK, GL_FILL)
        glPopMatrix()

        glDisableVertexAttribArray(0)

        if SMPL_normals is not None:
            glDisableVertexAttribArray(1)
            glDisableVertexAttribArray(2)

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)  # Unbind


# #normal3D: 3 dim
# #joints70: 3x70 =210 dim
# def drawfaceNormal_70(normal3D, joints,  color):

#     glLineWidth(2.0)
#     #Visualize Joints
#     glColor3ub(color[0], color[1], color[2])

#     #faceEye
#     eyeCenterPoint = 0.5 *(joints[(45*3):(45*3+3)] + joints[(36*3):(36*3+3)])
#     normalEndPoint = eyeCenterPoint+ normal3D * 20

#     glBegin(GL_LINES)
#     glVertex3f(eyeCenterPoint[0], eyeCenterPoint[1], eyeCenterPoint[2])
#     glVertex3f(normalEndPoint[0], normalEndPoint[1], normalEndPoint[2])
#     glEnd()

#     glPushMatrix()
#     glTranslate(normalEndPoint[0], normalEndPoint[1], normalEndPoint[2])
#     glutSolidSphere(1, 10, 10)
#     glPopMatrix()


# #normal3D: 3 dim
# #joints: 3x21
# def drawbodyNormal(normal3D, rootPts,  color):

#     glLineWidth(2.0)
#     #Visualize Joints
#     glColor3ub(color[0], color[1], color[2])

#     #neckPoint = joints[(0*3):(0*3+3)]
#     normalEndPoint = rootPts+ normal3D * 40

#     glBegin(GL_LINES)
#     glVertex3f(rootPts[0], rootPts[1], rootPts[2])
#     glVertex3f(normalEndPoint[0], normalEndPoint[1], normalEndPoint[2])
#     glEnd()

#     glPushMatrix()
#     glTranslate(normalEndPoint[0], normalEndPoint[1], normalEndPoint[2])
#     glutSolidSphere(1, 10, 10)
#     glPopMatrix()


# normal3D: 3 dim
# rootPt: 3x1
def drawNormal(normal3D, rootPt, color, normalLength=40):
    glLineWidth(2.0)
    # Visualize Joints
    glColor3ub(color[0], color[1], color[2])

    # neckPoint = joints[(0*3):(0*3+3)]
    normalEndPoint = rootPt + normal3D * normalLength

    glBegin(GL_LINES)
    glVertex3f(rootPt[0], rootPt[1], rootPt[2])
    glVertex3f(normalEndPoint[0], normalEndPoint[1], normalEndPoint[2])
    glEnd()

    glPushMatrix()
    glTranslate(normalEndPoint[0], normalEndPoint[1], normalEndPoint[2])
    glutSolidSphere(1, 10, 10)
    glPopMatrix()


g_connMat_hand21 = [[0, 1], [1, 2], [2, 3], [3, 4],
                    [0, 5], [5, 6], [6, 7], [7, 8],
                    [0, 9], [9, 10], [10, 11], [11, 12],
                    [0, 13], [13, 14], [14, 15], [15, 16],
                    [0, 17], [17, 18], [18, 19], [19, 20]]
g_connMat_hand21 = np.array(g_connMat_hand21, dtype=int)


# joints70: 3x70 =210 dim
def drawhand_21(joints, color, normal=None):
    # #Visualize Joints
    glColor3ub(color[0], color[1], color[2])
    # for i in range(len(joints)/3):

    #     glPushMatrix()
    #     glTranslate(joints[3*i], joints[3*i+1], joints[3*i+2])
    #     glutSolidSphere(1, 10, 10)
    #     glPopMatrix()

    connMat_coco19 = g_connMat_hand21
    # Visualize Bones
    for conn in connMat_coco19:
        # x0, y0, z0 is the coordinate of the base point
        x0 = joints[3 * conn[0]]
        y0 = joints[3 * conn[0] + 1]
        z0 = joints[3 * conn[0] + 2]
        # x1, y1, z1 is the vector points from the base to the target
        x1 = joints[3 * conn[1]]
        y1 = joints[3 * conn[1] + 1]
        z1 = joints[3 * conn[1] + 2]

        if (x0 == 0) or (x1 == 0):
            continue

        x1 -= x0
        y1 -= y0
        z1 -= z0

        if abs(x1) > 20 or abs(y1) > 20 or abs(z1) > 20:
            continue

        if g_bSimpleHead and conn[0] == 0 and conn[1] == 1:
            x1 = x1 * 0.5
            y1 = y1 * 0.5
            z1 = z1 * 0.5

        length = math.sqrt(x1 * x1 + y1 * y1 + z1 * z1)
        theta = math.degrees(math.acos(z1 / length))
        phi = math.degrees(math.atan2(y1, x1))

        glPushMatrix()
        glTranslate(x0, y0, z0)
        glRotatef(phi, 0, 0, 1)
        glRotatef(theta, 0, 1, 0)
        glutSolidCone(0.5, length, 10, 10)
        glPopMatrix()


# draw human location only
def DrawPosOnly():
    global g_colors
    global g_frameIdx  # , g_normals
    global g_speech, g_speechGT
    global g_bApplyRootOffset

    global g_posOnly
    global g_bodyNormals, g_faceNormals

    # print(g_frameIdx)
    if g_posOnly is None:
        return

    for humanIdx in range(len(g_posOnly)):

        if (g_frameIdx >= g_posOnly[humanIdx].shape[1]):
            continue

        # if(humanIdx==1):
        #     continue

        skel = g_posOnly[humanIdx][:, g_frameIdx]
        # normal = g_normals[skelIdx, :, g_idx]
        color = g_colors[humanIdx % len(g_colors)]
        glColor3ub(color[0], color[1], color[2])

        glPushMatrix()
        glTranslate(skel[0], skel[1], skel[2])

        if humanIdx == 0:  # or humanIdx==1:
            glutSolidCube(10)
        elif False:  # humanIdx==1:#or humanIdx==1:
            glutWireSphere(10, 10, 10)
        else:
            glutSolidSphere(10, 10, 10)
        glPopMatrix()

        # Draw body normal
        if g_bodyNormals is not None and len(g_bodyNormals) > humanIdx:

            if g_bodyNormals[humanIdx].shape[1] > g_frameIdx:
                normal3D = g_bodyNormals[humanIdx][:, g_frameIdx]  # 3x1
                # drawbodyNormal(normal3D, skel, [255, 0, 0])
                drawNormal(normal3D, skel, [255, 0, 0])

        # Draw face normal
        if g_faceNormals is not None and len(g_faceNormals) > humanIdx:

            if g_faceNormals[humanIdx].shape[1] > g_frameIdx:
                normal3D = g_faceNormals[humanIdx][:, g_frameIdx]  # 3x1
                drawNormal(normal3D, skel, [0, 255, 0], normalLength=30)


# g_trajectory: a list of np.array. skeNum x  (trajDim:3, frames)
def DrawTrajectory():
    global g_colors
    global g_trajectory, g_frameIdx  # , g_normals

    if g_trajectory is None:
        return

    for humanIdx in range(len(g_trajectory)):

        if (g_frameIdx >= g_trajectory[humanIdx].shape[1]):
            continue

        color = g_colors[humanIdx % len(g_colors)]

        glLineWidth(2.0)

        # Visualize Joints
        glColor3ub(color[0], color[1], color[2])

        glPushMatrix()
        if g_bApplyRootOffset:
            # glTranslatef(40*humanIdx,0,0)
            glTranslatef(ROOT_OFFSET_DIST * humanIdx, 0, 0)

        # Visualize All Trajctory locations
        interval = 10
        for idx in range(0, g_trajectory[humanIdx].shape[1] - interval, interval):
            # root location
            x0 = g_trajectory[humanIdx][0, idx]
            y0 = g_trajectory[humanIdx][1, idx]
            z0 = g_trajectory[humanIdx][2, idx]

            x1 = g_trajectory[humanIdx][0, idx + interval]
            y1 = g_trajectory[humanIdx][1, idx + interval]
            z1 = g_trajectory[humanIdx][2, idx + interval]

            glBegin(GL_LINES)
            glVertex3f(x0, y0, z0)
            glVertex3f(x1, y1, z1)
            glEnd()

        """
        #Visualize current location and normal dirctio
        # x0, y0, z0 for the root location
        x0 = g_trajectory[humanIdx][0,g_frameIdx]
        y0 = g_trajectory[humanIdx][1,g_frameIdx]
        z0 = g_trajectory[humanIdx][2,g_frameIdx]

        x1 = g_trajectory[humanIdx][3,g_frameIdx]
        y1 = g_trajectory[humanIdx][4,g_frameIdx]
        z1 = g_trajectory[humanIdx][5,g_frameIdx]

        glBegin(GL_LINES)
        glVertex3f(x0, y0, z0)
        glVertex3f(x1,y1,z1)
        glEnd()
        """

        glPopMatrix()
        # drawface_70(face3D, g_colors[humanIdx % len(g_colors)])

        # if g_faceNormals is not None and len(g_faceNormals)> humanIdx:

        #     if g_faceNormals[humanIdx].shape[1]<=g_frameIdx:
        #         print("Warning: g_faceNormals[humanIdx].shape[2]<=g_frameId")
        #         continue

        #     normal3D = g_faceNormals[humanIdx][:,g_frameIdx]  #3x1
        #     #drawfaceNormal_70(normal3D, face3D, g_colors[humanIdx % len(g_colors)])
        #     #drawfaceNormal_70(normal3D, face3D, [0, 255, 255])
        #     eyeCenterPoint = 0.5 *(face3D[(45*3):(45*3+3)] + face3D[(36*3):(36*3+3)])
        #     drawNormal(normal3D, eyeCenterPoint, [0, 255, 255])


# g_skeletons should be: #(skeNum, skelDim, skelFrames)
def DrawSkeletons():
    # global g_colors
    global g_skeletons, g_frameIdx  # , g_normals
    global g_speech, g_speechGT
    global g_bApplyRootOffset
    global g_bodyNormals
    if g_skeletons is None:
        return

    # for humanIdx in range(g_skeletons.shape[0]):
    for humanIdx in range(len(g_skeletons)):

        if g_onlyDrawHumanIdx >= 0 and humanIdx != g_onlyDrawHumanIdx:
            continue

        if (g_frameIdx >= g_skeletons[humanIdx]['skeleton'].shape[1]):
            continue

        skel = g_skeletons[humanIdx]['skeleton'][:, g_frameIdx]
        skel_type = g_skeletons[humanIdx]['type']
        skel_color = g_skeletons[humanIdx]['color']

        if skel_color is None:
            skel_color = g_colors[humanIdx % len(g_colors)]

        if g_bApplyRootOffset:
            skel = skel.copy()
            # skel[0::3] = skel[0::3]+ 70 *humanIdx
            skel[0::3] = skel[0::3] + ROOT_OFFSET_DIST * humanIdx

        # If type is not specified
        if skel_type == None:
            if skel.shape[
                0] == 78:  # SMPlCOCO19 + headtop (19) + (leftFoot --toe20 pink21 heel22) + (rightFoot--toe23-pink24-heel25)
                drawbody_SMPLCOCO_TotalCap26(skel, [0, 255, 0])
            elif skel.shape[0] == 57:
                drawbody_SMC19(skel, skel_color)  # Panoptic Studio (SMC19) with 19 joints. Note SMC21 includes headtop

            elif skel.shape[0] == 42:  # LSP14joints also used in HMR
                drawbody_LSP14(skel, skel_color)
            elif skel.shape[0] == 51:  # simpler human36m (17joints)
                drawbody_joint17_human36m(skel, [0, 255, 0])  # skel_color)
            elif skel.shape[0] == 72:  # SMPL LBS skeleton
                drawbody_joint24_smplLBS(skel, skel_color)
            elif skel.shape[0] == 96:  # human36m (32joints)
                drawbody_joint32_human36m(skel, skel_color)
            elif skel.shape[0] == 66:  # Holden's converted form (21joints)
                drawbody_joint22(skel, skel_color)
            elif skel.shape[0] == 93:  # CMU Mocap Raw data (31joints)
                drawbody_joint31(skel, skel_color)
            elif skel.shape[0] == 186:  # Adam model (62 joints: 22 for body 20x2 for fingers)
                drawbody_jointAdam(skel, skel_color)
            elif skel.shape[0] == 189:  # MTC skeleton from Hand2Body paper vis
                drawbody_jointMTC86(skel, skel_color)

            elif skel.shape[0] == 54:  # OpenPose 18
                drawbody_jointOpenPose18(skel, skel_color)

            elif skel.shape[0] == 147:  # SPIN 49 (25 openpose +  24 superset)
                drawbody_jointSpin49(skel, skel_color)
            else:
                drawbody_joint_ptOnly(skel, skel_color)

        elif skel_type == "smplcoco":
            drawbody_SMPLCOCO19(skel, skel_color)  # from HMR, SMPL->COCO19 regression. Same as MTC20's first 19.
        elif skel_type == "spin":
            if skel.shape[0] == 72:  # SPIN 24 (without openpose, 24 superset)
                drawbody_jointSpin24(skel, skel_color)
            elif skel.shape[0] == 147:  # SPIN 49 (25 openpose +  24 superset)
                drawbody_jointSpin49(skel, skel_color)
            else:
                assert False
        # finger
        elif skel_type == "hand_smplx":
            if skel.shape[0] == 63:  # SPIN 49 (25 openpose +  24 superset)
                drawhand_joint21(skel, skel_color, type=skel_type)
            else:
                assert False

        elif skel_type == "hand_panopticdb":
            if skel.shape[0] == 63:  # SPIN 49 (25 openpose +  24 superset)
                drawhand_joint21(skel, skel_color, type=skel_type)
            else:
                assert False

        if g_bodyNormals is not None and len(g_bodyNormals) > humanIdx:

            if g_bodyNormals[humanIdx].shape[1] <= g_frameIdx:
                print("Warning: g_bodyNormals[humanIdx].shape[2]<=g_frameId")
                continue

            normal3D = g_bodyNormals[humanIdx][:, g_frameIdx]  # 3x1
            # drawbodyNormal(normal3D, skel, [255, 0, 0])
            rootPt = skel[(0 * 3):(0 * 3 + 3)]

            if skel.shape[0] == 66:
                i = 12
                rootPt = skel[(3 * i):(3 * i + 3)]
            # drawNormal(normal3D, rootPt, [0, 255, 255])
            # drawNormal(normal3D, rootPt, [255, 0, 0])

            # if g_onlyDrawHumanIdx>=0 and humanIdx!=g_onlyDrawHumanIdx:
            drawNormal(normal3D, rootPt, [0, 255, 0])

        # if g_faceNormals is not None and len(g_faceNormals)> humanIdx:

        #     if g_faceNormals[humanIdx].shape[1]<=g_frameIdx:
        #         print("Warning: g_bodyNormals[humanIdx].shape[2]<=g_frameId")
        #         continue

        #     normal3D = g_faceNormals[humanIdx][:,g_frameIdx]  #3x1
        #     #drawbodyNormal(normal3D, skel, [255, 0, 0])
        #     rootPt = skel[(0*3):(0*3+3)]

        #     if skel.shape[0]==66:
        #         i=13
        #         rootPt = skel[(3*i):(3*i+3)]
        #     # drawNormal(normal3D, rootPt, [0, 255, 255])
        #     drawNormal(normal3D, rootPt, [0, 255, 0])

        # #Draw Speeking Annotation
        # if skel.shape[0]==57 and g_speech is not None: #Coco19
        #     if(g_frameIdx < len(g_speech[humanIdx]['word'])):
        #         #draw_speaking_joint19(skel, g_speech[humanIdx]['indicator'][g_frameIdx], g_speech[humanIdx]['word'][g_frameIdx],  g_colors[humanIdx % len(g_colors)])
        #         draw_speaking_joint19(skel, g_speech[humanIdx]['indicator'][g_frameIdx], None,  skel_color)

        # if skel.shape[0]==66 and g_speech is not None: #Holden's
        #     if(len(g_speech)> humanIdx and  g_frameIdx < len(g_speech[humanIdx]['word'])):
        #         #draw_speaking_joint22(skel, g_speech[humanIdx][g_frameIdx],None,  g_colors[humanIdx % len(g_colors)])
        #         i=13
        #         facePt = skel[(3*i):(3*i+3)]
        #         draw_speaking_general(facePt, g_speech[humanIdx]['indicator'][g_frameIdx], g_speech[humanIdx]['word'][g_frameIdx],  [0, 0, 255], offset=[0,-43,0])

        # if skel.shape[0]==66 and g_speechGT is not None: #Holden's
        #     if(g_frameIdx < len(g_speechGT[humanIdx]['word'])):
        #         #draw_speaking_joint22(skel, g_speechGT[humanIdx][g_frameIdx],"GT: Speaking", [255, 0, 0], 40)
        #         i=13
        #         facePt = skel[(3*i):(3*i+3)]
        #         draw_speaking_general(facePt, g_speechGT[humanIdx]['indicator'][g_frameIdx], g_speechGT[humanIdx]['word'][g_frameIdx],  g_colors[humanIdx % len(g_colors)], offset=[0,-33,0])


# g_skeletons should be: #(skeNum, skelDim, skelFrames)
def DrawSkeletonsGT():
    assert False  # Deprecated

    global g_colors
    # global g_skeletons_GT, g_frameIdx#, g_normals
    global g_speech, g_speechGT
    global g_bApplyRootOffset
    global g_bodyNormals
    # print(g_frameIdx)
    if g_skeletons_GT is None:
        return

    # frameLimit = g_skeletons.shape[2]
    # frameLens = [l.shape[1] for l in g_skeletons]
    # g_frameLimit = min(frameLens)

    # for humanIdx in range(g_skeletons.shape[0]):
    for humanIdx in range(len(g_skeletons_GT)):
        # if skelIdx ==0:
        #     # if g_idx+time_offset>=g_skeletons.shape[2]:
        #     #     continue
        #     skel = g_skeletons[skelIdx, :, g_idx+time_offset]
        #     # normal = g_normals[skelIdx, :, g_idx+time_offset]
        # else:
        # skel = g_skeletons[humanIdx, :, g_frameIdx]
        if (g_frameIdx >= g_skeletons_GT[humanIdx].shape[1]):
            continue

        skel = g_skeletons_GT[humanIdx][:, g_frameIdx]
        # normal = g_normals[skelIdx, :, g_idx]

        if g_bApplyRootOffset:
            skel = skel.copy()
            # skel[0::3] = skel[0::3]+ 70 *humanIdx
            skel[0::3] = skel[0::3] + ROOT_OFFSET_DIST * humanIdx

        if skel.shape[
            0] == 78:  # SMPlCOCO19 + headtop (19) + (leftFoot --toe20 pink21 heel22) + (rightFoot--toe23-pink24-heel25)
            drawbody_SMPLCOCO_TotalCap26(skel, [0, 255, 0])
        elif skel.shape[0] == 57:  # Panoptic Studio (SMC19) with 19 joints. Note SMC21 includes headtop
            drawbody_SMC19(skel, g_colors[humanIdx % len(g_colors)])
        elif skel.shape[0] == 96:  # human36
            drawbody_joint32_human36m(skel, g_colors[humanIdx % len(g_colors)])
        elif skel.shape[0] == 66:  # Holden's converted form
            drawbody_joint22(skel, g_colors[humanIdx % len(g_colors)])
        elif skel.shape[0] == 93:  # CMU Mocap Raw data (31joints)
            drawbody_joint31(skel, g_colors[humanIdx % len(g_colors)])
        else:
            drawbody_joint_ptOnly(skel, g_colors[humanIdx % len(g_colors)])

        # if False:#g_bodyNormals is not None and len(g_bodyNormals)> humanIdx:

        #     if g_bodyNormals[humanIdx].shape[1]<=g_frameIdx:
        #         print("Warning: g_bodyNormals[humanIdx].shape[2]<=g_frameId")
        #         continue

        #     normal3D = g_bodyNormals[humanIdx][:,g_frameIdx]  #3x1
        #     #drawbodyNormal(normal3D, skel, [255, 0, 0])
        #     rootPt = skel[(0*3):(0*3+3)]
        #     drawNormal(normal3D, rootPt, [255, 0, 0])

        # #Draw Speeking Annotation
        # if skel.shape[0]==57 and g_speech is not None: #Coco19
        #     if(g_frameIdx < len(g_speech[humanIdx]['word'])):
        #         draw_speaking_joint19(skel, g_speech[humanIdx]['indicator'][g_frameIdx], g_speech[humanIdx]['word'][g_frameIdx],  g_colors[humanIdx % len(g_colors)])

        # # if skel.shape[0]==66 and g_speech is not None: #Holden's
        # #     if(g_frameIdx < len(g_speech[humanIdx])):
        # #         draw_speaking_joint22(skel, g_speech[humanIdx][g_frameIdx],None,  g_colors[humanIdx % len(g_colors)])

        # # if skel.shape[0]==66 and g_speechGT is not None: #Holden's
        # #     if(g_frameIdx < len(g_speechGT[humanIdx])):
        # #         draw_speaking_joint22(skel, g_speechGT[humanIdx][g_frameIdx],"GT: Speaking", [255, 0, 0], 40)


# LSP14 format (used in HMR)
# Right ankle 1
# Right knee 2
# Right hip 3
# Left hip 4
# Left knee 5
# Left ankle 6
# Right wrist 7
# Right elbow 8
# Right shoulder 9
# Left shoulder 10
# Left elbow 11
# Left wrist 12
# Neck 13
# Head top 14
g_connMat_lsp14 = [[13, 3], [3, 2], [2, 1],  # Right leg
                   [13, 4], [4, 5], [5, 6],  # Left leg
                   [13, 10], [10, 11], [11, 12],  # Left Arm
                   [13, 9], [9, 8], [8, 7],  # Right shoulder
                   [13, 14]  # Nect -> Headtop
                   ]
g_connMat_lsp14 = np.array(g_connMat_lsp14, dtype=int) - 1  # zero Idx


# LSP14 format (used in HMR)
def drawbody_LSP14(joints, color):
    # Visualize Joints
    glColor3ub(color[0], color[1], color[2])
    for i in range(int(len(joints) / 3)):

        if g_bSimpleHead and (i >= 15 or i == 1):
            continue
        glPushMatrix()
        glTranslate(joints[3 * i], joints[3 * i + 1], joints[3 * i + 2])
        glutSolidSphere(2, 10, 10)
        glPopMatrix()

    connMat = g_connMat_lsp14
    # Visualize Bones
    for conn in connMat:
        # x0, y0, z0 is the coordinate of the base point
        x0 = joints[3 * conn[0]]
        y0 = joints[3 * conn[0] + 1]
        z0 = joints[3 * conn[0] + 2]
        # x1, y1, z1 is the vector points from the base to the target
        x1 = joints[3 * conn[1]] - x0
        y1 = joints[3 * conn[1] + 1] - y0
        z1 = joints[3 * conn[1] + 2] - z0

        if g_bSimpleHead and conn[0] == 0 and conn[1] == 1:
            x1 = x1 * 0.5
            y1 = y1 * 0.5
            z1 = z1 * 0.5

        length = math.sqrt(x1 * x1 + y1 * y1 + z1 * z1)
        theta = math.degrees(math.acos(z1 / length))
        phi = math.degrees(math.atan2(y1, x1))

        glPushMatrix()
        glTranslate(x0, y0, z0)
        glRotatef(phi, 0, 0, 1)
        glRotatef(theta, 0, 1, 0)
        glutSolidCone(2, length, 10, 10)
        glPopMatrix()


# In Zero index
# This is regressed joint from SMPL model defined from HMR
# MTC20 is exactly same as this except it has one more joint on spine (19)
g_connMat_smplcoco19 = [[12, 2], [2, 1], [1, 0],  # Right leg
                        [12, 3], [3, 4], [4, 5],  # Left leg
                        [12, 9], [9, 10], [10, 11],  # Left Arm
                        [12, 8], [8, 7], [7, 6],  # Right shoulder
                        [12, 14], [14, 16], [16, 18],  # Neck(12)->Nose(14)->rightEye(16)->rightEar(18)
                        [14, 15], [15, 17],  # Nose(14)->leftEye(15)->leftEar(17).
                        [14, 13]  # Nose->headTop(13)
                        ]
g_connMat_smplcoco19 = np.array(g_connMat_smplcoco19, dtype=int)  # zero Idx


def drawbody_SMPLCOCO19(joints, color, normal=None):
    bBoneIsLeft = [0, 0, 0,
                   1, 1, 1,
                   1, 1, 1,
                   0, 0, 0,
                   1, 0, 0,
                   1, 1,
                   1]  # To draw left as different color. Torso is treated as left

    # Visualize Joints
    glColor3ub(color[0], color[1], color[2])
    for i in range(int(len(joints) / 3)):
        # if i!=2:
        #     continue
        glPushMatrix()
        glTranslate(joints[3 * i], joints[3 * i + 1], joints[3 * i + 2])
        glutSolidSphere(2, 10, 10)
        glPopMatrix()

    connMat = g_connMat_smplcoco19
    # Visualize Bones
    for i, conn in enumerate(connMat):
        if bBoneIsLeft[i]:  # Left as a color
            glColor3ub(color[0], color[1], color[2])
        else:  # Right as black
            glColor3ub(0, 0, 0)

        # x0, y0, z0 is the coordinate of the base point
        x0 = joints[3 * conn[0]]
        y0 = joints[3 * conn[0] + 1]
        z0 = joints[3 * conn[0] + 2]
        # x1, y1, z1 is the vector points from the base to the target
        x1 = joints[3 * conn[1]] - x0
        y1 = joints[3 * conn[1] + 1] - y0
        z1 = joints[3 * conn[1] + 2] - z0

        length = math.sqrt(x1 * x1 + y1 * y1 + z1 * z1)
        theta = math.degrees(math.acos(z1 / length))
        phi = math.degrees(math.atan2(y1, x1))

        glPushMatrix()
        glTranslate(x0, y0, z0)
        glRotatef(phi, 0, 0, 1)
        glRotatef(theta, 0, 1, 0)
        glutSolidCone(2, length, 10, 10)
        glPopMatrix()

    # Visualize Normals
    if normal is not None:
        i = 1
        facePt = joints[(3 * i):(3 * i + 3)]
        normalPt = facePt + normal * 50

        glColor3ub(0, 255, 255)
        glPushMatrix()
        glTranslate(normalPt[0], normalPt[1], normalPt[2])
        glutSolidSphere(1, 10, 10)
        glPopMatrix()

        glBegin(GL_LINES)
        glVertex3f(facePt[0], facePt[1], facePt[2])
        glVertex3f(normalPt[0], normalPt[1], normalPt[2])
        glEnd()


# Panoptic Studio's Skeleton Output (AKA SMC19). Note SM21 has headTop(19) and spine(20)
g_bSimpleHead = False
if g_bSimpleHead == False:
    g_connMat_smc19 = [[1, 2], [1, 4], [4, 5], [5, 6], [1, 3], [3, 7], [7, 8], [8, 9], [3, 13], [13, 14], [14, 15],
                       [1, 10], [10, 11], [11, 12], [2, 16], [16, 17], [2, 18], [18, 19]]
else:
    g_connMat_smc19 = [[1, 2], [1, 4], [4, 5], [5, 6], [1, 3], [3, 7], [7, 8], [8, 9], [3, 13], [13, 14], [14, 15],
                       [1, 10], [10, 11], [11, 12]]
g_connMat_smc19 = np.array(g_connMat_smc19, dtype=int) - 1  # zero Idx


def drawbody_SMC19(joints, color, normal=None):
    # Visualize Joints
    glColor3ub(color[0], color[1], color[2])
    for i in range(int(len(joints) / 3)):

        if g_bSimpleHead and (i >= 15 or i == 1):
            continue
        glPushMatrix()
        glTranslate(joints[3 * i], joints[3 * i + 1], joints[3 * i + 2])
        glutSolidSphere(2, 10, 10)
        glPopMatrix()

    connMat = g_connMat_smc19
    # Visualize Bones
    for conn in connMat:
        # x0, y0, z0 is the coordinate of the base point
        x0 = joints[3 * conn[0]]
        y0 = joints[3 * conn[0] + 1]
        z0 = joints[3 * conn[0] + 2]
        # x1, y1, z1 is the vector points from the base to the target
        x1 = joints[3 * conn[1]] - x0
        y1 = joints[3 * conn[1] + 1] - y0
        z1 = joints[3 * conn[1] + 2] - z0

        if g_bSimpleHead and conn[0] == 0 and conn[1] == 1:
            x1 = x1 * 0.5
            y1 = y1 * 0.5
            z1 = z1 * 0.5

        length = math.sqrt(x1 * x1 + y1 * y1 + z1 * z1)
        theta = math.degrees(math.acos(z1 / length))
        phi = math.degrees(math.atan2(y1, x1))

        glPushMatrix()
        glTranslate(x0, y0, z0)
        glRotatef(phi, 0, 0, 1)
        glRotatef(theta, 0, 1, 0)
        glutSolidCone(2, length, 10, 10)
        glPopMatrix()

    # Visualize Normals
    if normal is not None:
        i = 1
        facePt = joints[(3 * i):(3 * i + 3)]
        normalPt = facePt + normal * 50

        glColor3ub(0, 255, 255)
        glPushMatrix()
        glTranslate(normalPt[0], normalPt[1], normalPt[2])
        glutSolidSphere(1, 10, 10)
        glPopMatrix()

        glBegin(GL_LINES)
        glVertex3f(facePt[0], facePt[1], facePt[2])
        glVertex3f(normalPt[0], normalPt[1], normalPt[2])
        glEnd()


# Panoptic Studio's Skeleton Output (AKA SMC19). Note SM21 has headTop(19) and spine(20)
def drawbody_SMPLCOCO_TotalCap26(joints, color, normal=None):
    connMat = [[12, 2], [2, 1], [1, 0],  # Right leg
               [12, 3], [3, 4], [4, 5],  # Left leg
               [12, 9], [9, 10], [10, 11],  # Left Arm
               [12, 8], [8, 7], [7, 6],  # Right shoulder
               [12, 14], [14, 16], [16, 18],  # Neck(12)->Nose(14)->rightEye(16)->rightEar(18)
               [14, 15], [15, 17],  # Nose(14)->leftEye(15)->leftEar(17).
               [14, 13],  # Nose->headMidle(13)
               [12, 19],  # headTop19
               [5, 20], [5, 21], [5, 22],  # leftFoot
               [0, 23], [0, 24], [0, 25]  # rightFoot
               ]
    connMat = np.array(connMat, dtype=int)  # zero Idx

    # Visualize Joints
    glColor3ub(color[0], color[1], color[2])
    for i in range(int(len(joints) / 3)):

        if g_bSimpleHead and (i >= 15 or i == 1):
            continue
        glPushMatrix()
        glTranslate(joints[3 * i], joints[3 * i + 1], joints[3 * i + 2])
        glutSolidSphere(2, 10, 10)
        glPopMatrix()

    # Visualize Bones
    for conn in connMat:
        # x0, y0, z0 is the coordinate of the base point
        x0 = joints[3 * conn[0]]
        y0 = joints[3 * conn[0] + 1]
        z0 = joints[3 * conn[0] + 2]
        # x1, y1, z1 is the vector points from the base to the target
        x1 = joints[3 * conn[1]] - x0
        y1 = joints[3 * conn[1] + 1] - y0
        z1 = joints[3 * conn[1] + 2] - z0

        if g_bSimpleHead and conn[0] == 0 and conn[1] == 1:
            x1 = x1 * 0.5
            y1 = y1 * 0.5
            z1 = z1 * 0.5

        length = math.sqrt(x1 * x1 + y1 * y1 + z1 * z1)
        theta = math.degrees(math.acos(z1 / length))
        phi = math.degrees(math.atan2(y1, x1))

        glPushMatrix()
        glTranslate(x0, y0, z0)
        glRotatef(phi, 0, 0, 1)
        glRotatef(theta, 0, 1, 0)
        glutSolidCone(2, length, 10, 10)
        glPopMatrix()

    # Visualize Normals
    if normal is not None:
        i = 1
        facePt = joints[(3 * i):(3 * i + 3)]
        normalPt = facePt + normal * 50

        glColor3ub(0, 255, 255)
        glPushMatrix()
        glTranslate(normalPt[0], normalPt[1], normalPt[2])
        glutSolidSphere(1, 10, 10)
        glPopMatrix()

        glBegin(GL_LINES)
        glVertex3f(facePt[0], facePt[1], facePt[2])
        glVertex3f(normalPt[0], normalPt[1], normalPt[2])
        glEnd()


g_connMat_coco14 = [[13, 3], [3, 2], [2, 1],  # Right leg
                    [13, 4], [4, 5], [5, 6],  # Left leg
                    [13, 10], [10, 11], [11, 12],  # Left Arm
                    [13, 9], [9, 8], [8, 7],  # Right shoulder
                    ]
g_connMat_coco14 = np.array(g_connMat_coco14, dtype=int) - 1  # zero Idx


def drawbody_joint_ptOnly(joints, color, normal=None):
    # Visualize Joints
    glColor3ub(color[0], color[1], color[2])
    for i in range(int(len(joints) / 3)):
        glPushMatrix()
        glTranslate(joints[3 * i], joints[3 * i + 1], joints[3 * i + 2])
        glutSolidSphere(2, 10, 10)
        glPopMatrix()


# Human 36m DB's mocap data. 32 joints
g_connMat_joint32_human36m = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
                              # RightLeg: root(0), rHip(1), rKnee(2), rAnkle(3), rFootMid(4), rFootEnd(5)
                              [0, 6], [6, 7], [7, 8], [8, 9], [9, 10],
                              # LeftLeg: root, lHip(6), lKnee(7), lAnkle(8), lFootMid(9), lFootEnd(10)
                              [11, 12], [12, 13], [13, 14], [14, 15],
                              # root2(11), spineMid(12), neck(13), nose(14), head(15) #0,11 are the same points?
                              [16, 17], [17, 18], [18, 19], [20, 21], [20, 22],
                              # Left Arms. neck(16==13), lshoulder(17),  lElbow(18), lWrist (19=20), lThumb(21), lMiddleFinger(22)
                              [24, 25], [25, 26], [26, 27], [27, 29], [27, 30]
                              # Right Arm, neck(24==13), rshoulder(25),  rElbow(26), rWrist (27=28), rThumb(29), rMiddleFinger(30)
                              ]


def drawbody_joint32_human36m(joints, color, normal=None):
    bBoneIsLeft = [0, 0, 0, 0, 0,
                   1, 1, 1, 1, 1,
                   1, 1, 1, 1,
                   1, 1, 1, 1, 1,
                   0, 0, 0, 0, 0]  # To draw left as different color. Torso is treated as left

    # Visualize Joints
    glColor3ub(color[0], color[1], color[2])

    for i in range(int(len(joints) / 3)):
        glPushMatrix()
        glTranslate(joints[3 * i], joints[3 * i + 1], joints[3 * i + 2])
        glutSolidSphere(2, 10, 10)
        glPopMatrix()

    # Visualize Bones
    for i, conn in enumerate(g_connMat_joint32_human36m):
        if bBoneIsLeft[i]:  # Left as a color
            glColor3ub(color[0], color[1], color[2])
        else:  # Right as black
            glColor3ub(0, 0, 0)
        # x0, y0, z0 is the coordinate of the base point
        x0 = joints[3 * conn[0]]
        y0 = joints[3 * conn[0] + 1]
        z0 = joints[3 * conn[0] + 2]
        # x1, y1, z1 is the vector points from the base to the target
        x1 = joints[3 * conn[1]] - x0
        y1 = joints[3 * conn[1] + 1] - y0
        z1 = joints[3 * conn[1] + 2] - z0

        length = math.sqrt(x1 * x1 + y1 * y1 + z1 * z1)
        theta = math.degrees(math.acos(z1 / length))
        phi = math.degrees(math.atan2(y1, x1))

        glPushMatrix()
        glTranslate(x0, y0, z0)
        glRotatef(phi, 0, 0, 1)
        glRotatef(theta, 0, 1, 0)
        glutSolidCone(2, length, 10, 10)
        glPopMatrix()


# Simpler version of Human 36m DB's mocap data. with 17 joints
# 5 for torso + neck (root, spine, nect, nose, headtop)
g_connMat_joint17_human36m = [[0, 1], [1, 2], [2, 3],  # ,#root(0), rHip(1), rKnee(2), rAnkle(3)
                              [0, 4], [4, 5], [5, 6],  # ,[8,9], [9,10], #root(0, lHip(4), lKnee(5), lAnkle(6)
                              [0, 7], [7, 8], [8, 9], [9, 10],
                              # root(0, spineMid(7), neck(8), nose(9), head(10) #0,11 are the same points?
                              [8, 11], [11, 12], [12, 13],
                              # Left Arms. neck(8). lshoulder(11),  lElbow(12), lWrist (13)
                              [8, 14], [14, 15], [15, 16]  # Right Arm, neck(8), rshoulder(14),  rElbow(15), rWrist (16)
                              ]


def drawbody_joint17_human36m(joints, color, normal=None):
    bBoneIsLeft = [0, 0, 0,
                   1, 1, 1,
                   1, 1, 1, 1,
                   1, 1, 1,
                   0, 0, 0]  # To draw left as different color. Torso is treated as left

    # Visualize Joints
    glColor3ub(color[0], color[1], color[2])
    for i in range(int(len(joints) / 3)):
        glPushMatrix()
        glTranslate(joints[3 * i], joints[3 * i + 1], joints[3 * i + 2])
        glutSolidSphere(2, 10, 10)
        glPopMatrix()

    # Visualize Bones
    for i, conn in enumerate(g_connMat_joint17_human36m):
        if bBoneIsLeft[i]:  # Left as a color
            glColor3ub(color[0], color[1], color[2])
        else:  # Right as black
            glColor3ub(0, 0, 0)

        # x0, y0, z0 is the coordinate of the base point
        x0 = joints[3 * conn[0]]
        y0 = joints[3 * conn[0] + 1]
        z0 = joints[3 * conn[0] + 2]
        # x1, y1, z1 is the vector points from the base to the target
        x1 = joints[3 * conn[1]] - x0
        y1 = joints[3 * conn[1] + 1] - y0
        z1 = joints[3 * conn[1] + 2] - z0

        length = math.sqrt(x1 * x1 + y1 * y1 + z1 * z1)
        theta = math.degrees(math.acos(z1 / length))
        phi = math.degrees(math.atan2(y1, x1))

        glPushMatrix()
        glTranslate(x0, y0, z0)
        glRotatef(phi, 0, 0, 1)
        glRotatef(theta, 0, 1, 0)
        glutSolidCone(2, length, 10, 10)
        glPopMatrix()


# SMPL 24 joints used for LBS
g_connMat_joint24_smpl = [[0, 3], [3, 6], [6, 9], [9, 12], [12, 15],  # root-> torso -> head
                          [9, 13], [13, 16], [16, 18], [18, 20], [20, 22],  # Nect-> left hand
                          [9, 14], [14, 17], [17, 19], [19, 21], [21, 23],  # Nect-> right hand
                          [0, 1], [1, 4], [4, 7], [7, 10],  # left Leg
                          [0, 2], [2, 5], [5, 8], [8, 11]  # right leg
                          ]


def drawbody_joint24_smplLBS(joints, color, normal=None):
    bBoneIsLeft = [1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1,
                   0, 0, 0, 0, 0,
                   0, 0, 0, 0,
                   1, 1, 1, 1]  # To draw left as different color. Torso is treated as left

    # Visualize Joints
    glColor3ub(color[0], color[1], color[2])
    for i in range(int(len(joints) / 3)):
        glPushMatrix()
        glTranslate(joints[3 * i], joints[3 * i + 1], joints[3 * i + 2])
        glutSolidSphere(2, 10, 10)
        glPopMatrix()

    # Visualize Bones
    for i, conn in enumerate(g_connMat_joint24_smpl):
        if True:  # bBoneIsLeft[i]:          #Left as a color
            glColor3ub(color[0], color[1], color[2])
        else:  # Right as black
            glColor3ub(0, 0, 0)

        # x0, y0, z0 is the coordinate of the base point
        x0 = joints[3 * conn[0]]
        y0 = joints[3 * conn[0] + 1]
        z0 = joints[3 * conn[0] + 2]
        # x1, y1, z1 is the vector points from the base to the target
        x1 = joints[3 * conn[1]] - x0
        y1 = joints[3 * conn[1] + 1] - y0
        z1 = joints[3 * conn[1] + 2] - z0

        length = math.sqrt(x1 * x1 + y1 * y1 + z1 * z1)
        theta = math.degrees(math.acos(z1 / length))
        phi = math.degrees(math.atan2(y1, x1))

        glPushMatrix()
        glTranslate(x0, y0, z0)
        glRotatef(phi, 0, 0, 1)
        glRotatef(theta, 0, 1, 0)
        glutSolidCone(2, length, 10, 10)
        glPopMatrix()


# g_connMat_coco31 = [ [0,1],[1,2],[2,3],[3,4],[4,5], #root(0), rHip(1), rKnee(2), rAnkle(3), RFootMid(4), rFootEnd(5)
#                      [0,6],[6,7],[7,8],[8,9], [9,10], #root, lHip(6), lKnee(7), lAnkle(8), lFootMid(9), lFootEnd(10)
#                      [11,12], [12,13], [13,14], [14,15], #root2(11), spineMid(12), neck(13), nose(14), head(15) #0,11 are the same points?
#                      [16,17], [17,18], [18,19], [18,19], [20,21], [20,22] ,
#                      [24,25], [25,26], [26,27], [27,29], [27,30] ]

# Note that left right of torso are flipped, compared human36m format
g_connMat_coco31 = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
                    # root(0), lHip(1), lKnee(2), lAnkle(3), lFootMid(4), lFootEnd(5)
                    [0, 6], [6, 7], [7, 8], [8, 9], [9, 10],
                    # root, rHip(6), rKnee(7), lAnkle(8), lFootMid(9), lFootEnd(10)
                    [11, 12], [12, 13], [14, 15], [15, 16],
                    # root2(11), lowerback(12), spine(13=14), neck(15), head(16) #(0,11), (13=14) are the same points
                    [17, 18], [18, 19], [19, 20], [21, 22],
                    # spine2(14=17), lShoulder(18), lElbow(19), lWrist(20==21), lHand(22), LThumb(23, not valid and same as 20)
                    [24, 25], [25, 26], [26, 27], [28, 29]
                    # spine2(14=24), rShoulder(25), rElbow(26), rWrist(27==28), rHand(29), rThumb(30, not valid and same as 27)
                    ]


# g_connMat_coco31 = [ [0,18]
#                       ]
def drawbody_joint31(joints, color, normal=None):
    # Visualize Joints
    glColor3ub(color[0], color[1], color[2])
    for i in range(int(len(joints) / 3)):
        glPushMatrix()
        glTranslate(joints[3 * i], joints[3 * i + 1], joints[3 * i + 2])
        glutSolidSphere(2, 10, 10)
        glPopMatrix()

    # Visualize Bones
    for conn in g_connMat_coco31:
        # x0, y0, z0 is the coordinate of the base point
        x0 = joints[3 * conn[0]]
        y0 = joints[3 * conn[0] + 1]
        z0 = joints[3 * conn[0] + 2]
        # x1, y1, z1 is the vector points from the base to the target
        x1 = joints[3 * conn[1]] - x0
        y1 = joints[3 * conn[1] + 1] - y0
        z1 = joints[3 * conn[1] + 2] - z0

        length = math.sqrt(x1 * x1 + y1 * y1 + z1 * z1)
        theta = math.degrees(math.acos(z1 / length))
        phi = math.degrees(math.atan2(y1, x1))

        glPushMatrix()
        glTranslate(x0, y0, z0)
        glRotatef(phi, 0, 0, 1)
        glRotatef(theta, 0, 1, 0)
        glutSolidCone(2, length, 10, 10)
        glPopMatrix()


# Adam model
# Body (22) + Lhand (20) +  Rhand
g_connMat_Adam = [
    [0, 1], [1, 4], [4, 7], [7, 10],  # left leg
    [0, 2], [2, 5], [5, 8], [8, 11],  # right leg
    [9, 13], [13, 16], [16, 18], [18, 20],  # left arm
    [9, 14], [14, 17], [17, 19], [19, 21],  # right arm
    [0, 3], [3, 6], [6, 9], [9, 12], [12, 15],  # torso -> head

    [20, 22], [22, 23], [23, 24], [24, 25],  # LeftHand Thumb
    [20, 26], [26, 27], [27, 28], [28, 29],
    [20, 30], [30, 31], [31, 32], [32, 33],
    [20, 34], [34, 35], [35, 36], [36, 37],
    [20, 38], [38, 39], [39, 40], [40, 41],  # Left Pinky

    [21, 42], [42, 43], [43, 44], [44, 45],  # RightHand Thumb
    [21, 46], [46, 47], [47, 48], [48, 49],
    [21, 50], [50, 51], [51, 52], [52, 53],
    [21, 54], [54, 55], [55, 56], [56, 57],
    [21, 58], [58, 59], [59, 60], [60, 61]  # Right Pinky
]


def drawbody_jointAdam(joints, color, normal=None, ignore_root=False):
    # Visualize Joints
    glColor3ub(color[0], color[1], color[2])
    for i in range(1, int(len(joints) / 3)):
        # for i in range(22):

        glPushMatrix()
        glTranslate(joints[3 * i], joints[3 * i + 1], joints[3 * i + 2])

        if i < 22:
            glutSolidSphere(2, 10, 10)
        else:
            glutSolidSphere(0.5, 10, 10)
        glPopMatrix()

    connMat_adam = g_connMat_Adam
    # Visualize Bones
    for conn in connMat_adam:
        # x0, y0, z0 is the coordinate of the base point
        x0 = joints[3 * conn[0]]
        y0 = joints[3 * conn[0] + 1]
        z0 = joints[3 * conn[0] + 2]
        # x1, y1, z1 is the vector points from the base to the target
        x1 = joints[3 * conn[1]] - x0
        y1 = joints[3 * conn[1] + 1] - y0
        z1 = joints[3 * conn[1] + 2] - z0

        length = math.sqrt(x1 * x1 + y1 * y1 + z1 * z1)
        theta = math.degrees(math.acos(z1 / length))
        phi = math.degrees(math.atan2(y1, x1))

        glPushMatrix()
        glTranslate(x0, y0, z0)
        glRotatef(phi, 0, 0, 1)
        glRotatef(theta, 0, 1, 0)
        if conn[1] < 22:  # body
            glutSolidCone(2, length, 10, 10)
        else:
            glutSolidCone(0.4, length, 10, 10)
        glPopMatrix()

    # Spine to ground projection
    conn = [0, 1]
    x0 = joints[3 * conn[0]]
    y0 = joints[3 * conn[0] + 1]
    z0 = joints[3 * conn[0] + 2]
    # x1, y1, z1 is the vector points from the base to the target
    x1 = joints[3 * conn[1]]
    y1 = joints[3 * conn[1] + 1]
    z1 = joints[3 * conn[1] + 2]

    glBegin(GL_LINES)
    glVertex3f(x0, y0, z0)
    glVertex3f(x1, y1, z1)
    glEnd()


# SMC21
g_connMat_MTC = [
    [2, 20], [20, 0],  # Right Leg
    [0, 3], [3, 4], [4, 5],  # Left Arm
    [0, 9], [9, 10], [10, 11],  # Left Arm
    [2, 6], [6, 7], [7, 8],  # Left Leg
    [2, 12], [12, 13], [13, 14],  # Right Leg

    [5, 22], [22, 23], [23, 24], [24, 25],  # LeftHand Thumb
    [5, 26], [26, 27], [27, 28], [28, 29],
    [5, 30], [30, 31], [31, 32], [32, 33],
    [5, 34], [34, 35], [35, 36], [36, 37],
    [5, 38], [38, 39], [39, 40], [40, 41],  # Left Pinky

    [11, 43], [43, 44], [44, 45], [45, 46],  # RightHand Thumb
    [11, 47], [47, 48], [48, 49], [49, 50],
    [11, 51], [51, 52], [52, 53], [53, 54],
    [11, 55], [55, 56], [56, 57], [57, 58],
    [11, 59], [59, 60], [60, 61], [61, 62]  # Right Pinky
]


def drawbody_jointMTC86(joints, color, normal=None, ignore_root=False):
    # Visualize Joints
    glColor3ub(color[0], color[1], color[2])
    for i in range(1, int(len(joints) / 3)):
        # for i in range(22):

        glPushMatrix()
        glTranslate(joints[3 * i], joints[3 * i + 1], joints[3 * i + 2])

        if i < 22:
            glutSolidSphere(2, 10, 10)
        else:
            glutSolidSphere(0.5, 10, 10)
        glPopMatrix()

    connMat_adam = g_connMat_MTC
    # Visualize Bones
    for conn in connMat_adam:
        # x0, y0, z0 is the coordinate of the base point
        x0 = joints[3 * conn[0]]
        y0 = joints[3 * conn[0] + 1]
        z0 = joints[3 * conn[0] + 2]
        # x1, y1, z1 is the vector points from the base to the target
        x1 = joints[3 * conn[1]] - x0
        y1 = joints[3 * conn[1] + 1] - y0
        z1 = joints[3 * conn[1] + 2] - z0

        length = math.sqrt(x1 * x1 + y1 * y1 + z1 * z1)
        theta = math.degrees(math.acos(z1 / length))
        phi = math.degrees(math.atan2(y1, x1))

        glPushMatrix()
        glTranslate(x0, y0, z0)
        glRotatef(phi, 0, 0, 1)
        glRotatef(theta, 0, 1, 0)
        if conn[1] < 22:  # body
            glutSolidCone(2, length, 10, 10)
        else:
            glutSolidCone(0.4, length, 10, 10)
        glPopMatrix()

    # Spine to ground projection
    conn = [0, 1]
    x0 = joints[3 * conn[0]]
    y0 = joints[3 * conn[0] + 1]
    z0 = joints[3 * conn[0] + 2]
    # x1, y1, z1 is the vector points from the base to the target
    x1 = joints[3 * conn[1]]
    y1 = joints[3 * conn[1] + 1]
    z1 = joints[3 * conn[1] + 2]

    glBegin(GL_LINES)
    glVertex3f(x0, y0, z0)
    glVertex3f(x1, y1, z1)
    glEnd()


def drawbody_jointSpin49(joints, color, normal=None, ignore_root=False):
    # skelSize = 10
    skelSize = 2

    # Openpose25 + SPIN global 24
    link_openpose = [[8, 1], [1, 0], [0, 16], [16, 18], [0, 15], [15, 17],
                     [1, 2], [2, 3], [3, 4],  # Right Arm
                     [1, 5], [5, 6], [6, 7],  # Left Arm
                     [8, 12], [12, 13], [13, 14], [14, 21], [14, 19], [14, 20],
                     [8, 9], [9, 10], [10, 11], [11, 24], [11, 22], [11, 23]
                     ]

    link_spin24 = [[14, 16], [16, 12], [12, 17], [17, 18],
                   [12, 9], [9, 10], [10, 11],  # Right Arm
                   [12, 8], [8, 7], [7, 6],  # Left Arm
                   [14, 3], [3, 4], [4, 5],
                   [14, 2], [2, 1], [1, 0]
                   ]
    link_spin24 = np.array(link_spin24) + 25

    # Visualize Joints
    glColor3ub(color[0], color[1], color[2])
    # for i in range(1,int(len(joints)/3)):
    for i in range(25, int(len(joints) / 3)):
        # if i-25 == 13:
        #     glColor3f(1.0,0,0)
        # elif i-25 == 18:
        #     glColor3f(0,1.0,0)
        # elif i-25 == 15:
        #     glColor3f(1.0,1.0,0)
        # elif i-25 == 17:
        #     glColor3f(0.0,1.0,1.0)
        # else:
        #     glColor3f(0,0.0,1.0)

        glPushMatrix()
        glTranslate(joints[3 * i], joints[3 * i + 1], joints[3 * i + 2])
        glutSolidSphere(skelSize, 10, 10)
        glPopMatrix()

    # #Visualize Bones
    # glColor3ub(0,0,255)
    # for conn in link_openpose:
    #     # x0, y0, z0 is the coordinate of the base point
    #     x0 = joints[3*conn[0]]
    #     y0 = joints[3*conn[0]+1]
    #     z0 = joints[3*conn[0]+2]
    #     # x1, y1, z1 is the vector points from the base to the target
    #     x1 = joints[3*conn[1]] - x0
    #     y1 = joints[3*conn[1]+1] - y0
    #     z1 = joints[3*conn[1]+2] - z0

    #     length = math.sqrt(x1*x1 + y1*y1 + z1*z1)
    #     theta = math.degrees(math.acos(z1/length))
    #     phi = math.degrees(math.atan2(y1, x1))

    #     glPushMatrix()
    #     glTranslate(x0, y0, z0)
    #     glRotatef(phi, 0, 0, 1)
    #     glRotatef(theta, 0, 1, 0)
    #     glutSolidCone(0.4, length, 10, 10)
    #     glPopMatrix()

    # Visualize Bones
    # glColor3ub(255,0,0)
    glColor3ub(color[0], color[1], color[2])
    for conn in link_spin24:
        # x0, y0, z0 is the coordinate of the base point
        x0 = joints[3 * conn[0]]
        y0 = joints[3 * conn[0] + 1]
        z0 = joints[3 * conn[0] + 2]
        # x1, y1, z1 is the vector points from the base to the target
        x1 = joints[3 * conn[1]] - x0
        y1 = joints[3 * conn[1] + 1] - y0
        z1 = joints[3 * conn[1] + 2] - z0

        length = math.sqrt(x1 * x1 + y1 * y1 + z1 * z1)
        theta = math.degrees(math.acos(z1 / length))
        phi = math.degrees(math.atan2(y1, x1))

        glPushMatrix()
        glTranslate(x0, y0, z0)
        glRotatef(phi, 0, 0, 1)
        glRotatef(theta, 0, 1, 0)
        glutSolidCone(skelSize, length, 10, 10)

        glPopMatrix()

    # Spine to ground projection
    conn = [0, 1]
    x0 = joints[3 * conn[0]]
    y0 = joints[3 * conn[0] + 1]
    z0 = joints[3 * conn[0] + 2]
    # x1, y1, z1 is the vector points from the base to the target
    x1 = joints[3 * conn[1]]
    y1 = joints[3 * conn[1] + 1]
    z1 = joints[3 * conn[1] + 2]

    glBegin(GL_LINES)
    glVertex3f(x0, y0, z0)
    glVertex3f(x1, y1, z1)
    glEnd()


def drawbody_jointOpenPose18(joints, color, normal=None, ignore_root=False):
    skelSize = 2
    # Openpose 18
    link_openpose = [[1, 0], [0, 14], [14, 16], [0, 15], [15, 17],
                     [1, 2], [2, 3], [3, 4],  # Right Arm
                     [1, 5], [5, 6], [6, 7],  # Left Arm
                     [1, 11], [11, 12], [12, 13],  # Left Leg
                     [8, 1], [8, 9], [9, 10]  # Right Leg
                     ]

    # Visualize Joints
    glColor3ub(color[0], color[1], color[2])
    for i in range(1, int(len(joints) / 3)):
        glPushMatrix()
        glTranslate(joints[3 * i], joints[3 * i + 1], joints[3 * i + 2])
        glutSolidSphere(skelSize, 10, 10)
        glPopMatrix()

    # Visualize Bones
    glColor3ub(color[0], color[1], color[2])
    for conn in link_openpose:
        # x0, y0, z0 is the coordinate of the base point
        x0 = joints[3 * conn[0]]
        y0 = joints[3 * conn[0] + 1]
        z0 = joints[3 * conn[0] + 2]
        # x1, y1, z1 is the vector points from the base to the target
        x1 = joints[3 * conn[1]] - x0
        y1 = joints[3 * conn[1] + 1] - y0
        z1 = joints[3 * conn[1] + 2] - z0

        length = math.sqrt(x1 * x1 + y1 * y1 + z1 * z1)
        theta = math.degrees(math.acos(z1 / length))
        phi = math.degrees(math.atan2(y1, x1))

        glPushMatrix()
        glTranslate(x0, y0, z0)
        glRotatef(phi, 0, 0, 1)
        glRotatef(theta, 0, 1, 0)
        glutSolidCone(skelSize, length, 10, 10)

        glPopMatrix()

    # Spine to ground projection
    conn = [0, 1]
    x0 = joints[3 * conn[0]]
    y0 = joints[3 * conn[0] + 1]
    z0 = joints[3 * conn[0] + 2]
    # x1, y1, z1 is the vector points from the base to the target
    x1 = joints[3 * conn[1]]
    y1 = joints[3 * conn[1] + 1]
    z1 = joints[3 * conn[1] + 2]

    glBegin(GL_LINES)
    glVertex3f(x0, y0, z0)
    glVertex3f(x1, y1, z1)
    glEnd()


def drawbody_jointSpin24(joints, color, normal=None, ignore_root=False):
    link_spin24 = [  # [14,16], [16,12]       #TODO: debug
        # [12,17] , [17,18] ,
        [12, 9], [9, 10], [10, 11],  # Right Arm
        [12, 8], [8, 7], [7, 6],  # Left Arm
        [14, 3], [3, 4], [4, 5],
        [14, 2], [2, 1], [1, 0]
    ]
    link_spin24 = np.array(link_spin24)

    # Visualize Joints
    glColor3ub(color[0], color[1], color[2])
    for i in range(1, int(len(joints) / 3)):
        # for i in range(22):

        glPushMatrix()
        glTranslate(joints[3 * i], joints[3 * i + 1], joints[3 * i + 2])

        glutSolidSphere(2, 10, 10)
        glPopMatrix()

    # Visualize Bones
    # glColor3ub(255,0,0)
    glColor3ub(color[0], color[1], color[2])
    for conn in link_spin24:
        if joints[3 * conn[0]] == 0 or joints[3 * conn[1]] == 0:
            continue
        # x0, y0, z0 is the coordinate of the base point
        x0 = joints[3 * conn[0]]
        y0 = joints[3 * conn[0] + 1]
        z0 = joints[3 * conn[0] + 2]
        # x1, y1, z1 is the vector points from the base to the target
        x1 = joints[3 * conn[1]] - x0
        y1 = joints[3 * conn[1] + 1] - y0
        z1 = joints[3 * conn[1] + 2] - z0

        length = math.sqrt(x1 * x1 + y1 * y1 + z1 * z1)
        if length > 0.001:
            theta = math.degrees(math.acos(z1 / length))
            phi = math.degrees(math.atan2(y1, x1))

            glPushMatrix()
            glTranslate(x0, y0, z0)
            glRotatef(phi, 0, 0, 1)
            glRotatef(theta, 0, 1, 0)
            glutSolidCone(2, length, 10, 10)
            # glutSolidCone(20, length, 10, 10)

            glPopMatrix()

    # Spine to ground projection
    conn = [0, 1]
    x0 = joints[3 * conn[0]]
    y0 = joints[3 * conn[0] + 1]
    z0 = joints[3 * conn[0] + 2]
    # x1, y1, z1 is the vector points from the base to the target
    x1 = joints[3 * conn[1]]
    y1 = joints[3 * conn[1] + 1]
    z1 = joints[3 * conn[1] + 2]

    glBegin(GL_LINES)
    glVertex3f(x0, y0, z0)
    glVertex3f(x1, y1, z1)
    glEnd()


# 0 : Wrist
# 1 : Index_00
# 2 : Index_01
# 3 : Index_02
# 4 : Middle_00
# 5 : Middle_01
# 6 : Middle_02
# 7 : Little_00
# 8 : Little_01
# 9 : Little_02
# 10 : Ring_00
# 11 : Ring_01
# 12 : Ring_02
# 13 : Thumb_00
# 14 : Thumb_01
# 15 : Thumb_02
# 16 : Index_03
# 17 : Middle_03
# 18 : Little_03
# 19 : Ring_03
# 20 : Thumb_03
def drawhand_joint21(joints, color, normal=None, ignore_root=False, type='hand_smplx'):
    if type == "hand_panopticdb":
        link_panoptic_hand = [[0, 1], [1, 2], [2, 3], [3, 4],  # thumb
                              [0, 5], [5, 6], [6, 7], [7, 8],  # index
                              [0, 9], [9, 10], [10, 11], [11, 12],
                              [0, 13], [13, 14], [14, 15], [15, 16],
                              [0, 17], [17, 18], [18, 19], [19, 20]
                              ]
        link_hand = np.array(link_panoptic_hand)
    else:
        link_smplx_hand = [[0, 1], [1, 2], [2, 3], [3, 16],
                           [0, 4], [4, 5], [5, 6], [6, 17],
                           [0, 7], [7, 8], [8, 9], [9, 18],
                           [0, 10], [10, 11], [11, 12], [12, 19],
                           [0, 13], [13, 14], [14, 15], [15, 20]
                           ]
        link_hand = np.array(link_smplx_hand)

    # Visualize Joints
    glColor3ub(color[0], color[1], color[2])
    for i in range(1, int(len(joints) / 3)):
        # for i in range(22):

        glPushMatrix()
        glTranslate(joints[3 * i], joints[3 * i + 1], joints[3 * i + 2])

        glutSolidSphere(0, 5, 10, 10)
        glPopMatrix()

    # Visualize Bones
    # glColor3ub(255,0,0)
    glColor3ub(color[0], color[1], color[2])
    for conn in link_hand:
        # x0, y0, z0 is the coordinate of the base point
        x0 = joints[3 * conn[0]]
        y0 = joints[3 * conn[0] + 1]
        z0 = joints[3 * conn[0] + 2]
        # x1, y1, z1 is the vector points from the base to the target
        x1 = joints[3 * conn[1]] - x0
        y1 = joints[3 * conn[1] + 1] - y0
        z1 = joints[3 * conn[1] + 2] - z0

        length = math.sqrt(x1 * x1 + y1 * y1 + z1 * z1)
        if length > 0.001:
            theta = math.degrees(math.acos(z1 / length))
            phi = math.degrees(math.atan2(y1, x1))

            glPushMatrix()
            glTranslate(x0, y0, z0)
            glRotatef(phi, 0, 0, 1)
            glRotatef(theta, 0, 1, 0)
            # glutSolidCone(2, length, 10, 10)
            glutSolidCone(0.5, length, 10, 10)

            glPopMatrix()

    # Spine to ground projection
    conn = [0, 1]
    x0 = joints[3 * conn[0]]
    y0 = joints[3 * conn[0] + 1]
    z0 = joints[3 * conn[0] + 2]
    # x1, y1, z1 is the vector points from the base to the target
    x1 = joints[3 * conn[1]]
    y1 = joints[3 * conn[1] + 1]
    z1 = joints[3 * conn[1] + 2]

    glBegin(GL_LINES)
    glVertex3f(x0, y0, z0)
    glVertex3f(x1, y1, z1)
    glEnd()


# D. Holden's Data type
# root (3pts on the floor) + 21joints
parents = np.array([-1, 0, 1, 2, 3, 4, 1, 6, 7, 8, 1, 10, 11, 12, 12, 14, 15, 16, 12, 18, 19, 20])

g_connMat_coco22 = [[1, 2], [2, 3], [3, 4], [4, 5],  # right leg
                    [1, 6], [6, 7], [7, 8], [8, 9],  # left leg
                    [1, 10], [10, 11], [11, 12], [12, 13],  # spine, head
                    [12, 14], [14, 15], [15, 16], [16, 17],  # right arm
                    [12, 18], [18, 19], [19, 20], [20, 21]]  # left arm


# g_connMat_coco22 = np.array(g_connMat_coco22, dtype=int) - 1 #zero Idx
def drawbody_joint22(joints, color, normal=None, ignore_root=False):
    # Visualize Joints
    glColor3ub(color[0], color[1], color[2])
    for i in range(1, int(len(joints) / 3)):
        glPushMatrix()
        glTranslate(joints[3 * i], joints[3 * i + 1], joints[3 * i + 2])
        glutSolidSphere(2, 10, 10)
        glPopMatrix()

    connMat_coco22 = g_connMat_coco22
    # Visualize Bones
    for conn in connMat_coco22:
        # x0, y0, z0 is the coordinate of the base point
        x0 = joints[3 * conn[0]]
        y0 = joints[3 * conn[0] + 1]
        z0 = joints[3 * conn[0] + 2]
        # x1, y1, z1 is the vector points from the base to the target
        x1 = joints[3 * conn[1]] - x0
        y1 = joints[3 * conn[1] + 1] - y0
        z1 = joints[3 * conn[1] + 2] - z0

        length = math.sqrt(x1 * x1 + y1 * y1 + z1 * z1)
        theta = math.degrees(math.acos(z1 / length))
        phi = math.degrees(math.atan2(y1, x1))

        glPushMatrix()
        glTranslate(x0, y0, z0)
        glRotatef(phi, 0, 0, 1)
        glRotatef(theta, 0, 1, 0)
        glutSolidCone(2, length, 10, 10)
        glPopMatrix()

    # Spine to ground projection
    conn = [0, 1]
    x0 = joints[3 * conn[0]]
    y0 = joints[3 * conn[0] + 1]
    z0 = joints[3 * conn[0] + 2]
    # x1, y1, z1 is the vector points from the base to the target
    x1 = joints[3 * conn[1]]
    y1 = joints[3 * conn[1] + 1]
    z1 = joints[3 * conn[1] + 2]

    glBegin(GL_LINES)
    glVertex3f(x0, y0, z0)
    glVertex3f(x1, y1, z1)
    glEnd()


# Input: skel_list (skelNum, dim, frames): nparray or list of arrays (dim, frames)
# def setSkeleton(skel_list, bIsGT = False, jointType=None):
def setSkeleton(skel_list, jointType=None, colorRGB=None, bReset=True):
    global g_skeletons, g_frameLimit  # nparray: (skelNum, skelDim, frameNum)
    # global g_skeletons_GT #nparray: (skelNum, skelDim, frameNum)

    # If skel_list is not a list
    if isinstance(skel_list, list) == False and len(skel_list.shape) == 2:
        skel_list = skel_list[np.newaxis, :]

    # add joint type
    if g_skeletons is None or bReset:
        g_skeletons = []

    # if color is None:
    #     color = (255,0,0)
    # color can None

    for s in skel_list:
        g_skeletons.append({"skeleton": s, "color": colorRGB, "type": jointType})

    # else:# bisGT == False:      TODO: no need to have g_skeletons_GT anymore?
    #     g_skeletons_GT =[]
    #     for s in skel_list:

    #         g_skeletons_GT.append({"skeleton":s, "color":(255,0,0), "type":jointType})

    # if jointType =='smpl':
    #     print("Use smplcoco instead of smpl!")
    #     assert(False)

    # if bIsGT==False:
    #     #Add Skeleton Data

    #     # if len(skel_list)>1:
    #     #     lens = [len(l) for l in skel_list]
    #     #     minLeng=max(lens)

    #     #     for i in range(0,len(skel_list)):
    #     #         skel_list[i] = skel_list[i][:,:minLeng]

    #     #g_skeletons = np.asarray(skel_list)  #no effect if skel_list is already np.array
    #     g_skeletons = skel_list #List of 2dim np.array

    #     #g_frameLimit = g_skeletons.shape[2]
    #     # frameLens = [l.shape[1] for l in g_skeletons]
    #     # g_frameLimit = max(g_frameLimit,min(frameLens))

    # else:
    #      #Add Skeleton Data
    #     g_skeletons_GT = skel_list #List of 2dim np.array

    #     #g_frameLimit = g_skeletons.shape[2]
    #     # frameLens = [l.shape[1] for l in g_skeletons_GT]
    #     # g_frameLimit = max(g_frameLimit,min(frameLens))

    setFrameLimit()


# Input: skel_list (skelNum, dim, frames): nparray or list of arrays (dim, frames)
def showSkeleton(skel_list):
    # Add Skeleton Data
    global g_skeletons, g_frameLimit  # nparray: (skelNum, skelDim, frameNum)

    # if len(skel_list)>1:
    #     lens = [len(l) for l in skel_list]
    #     minLeng=max(lens)

    #     for i in range(0,len(skel_list)):
    #         skel_list[i] = skel_list[i][:,:minLeng]

    # g_skeletons = np.asarray(skel_list)  #no effect if skel_list is already np.array
    g_skeletons = skel_list  # List of 2dim np.array

    # g_frameLimit = g_skeletons.shape[2]
    frameLens = [l.shape[1] for l in g_skeletons]
    g_frameLimit = max(g_frameLimit, min(frameLens))

    # init_gl()


# Input: skel_list peopelNum x {'ver': vertexInfo, 'f': faceInfo}
#: vertexInfo should be (frames x vertexNum x 3 )
#: if vertexInfo has (vertexNum x 3 ), this function automatically changes it to (1 x vertexNum x 3)
#: faceInfo should be (vertexNum x 3 )
# 'normal': if missing, draw mesh by wireframes
def setMeshData(mesh_list, bComputeNormal=False):
    global g_meshes

    # g_skeletons = np.asarray(skel_list)  #no effect if skel_list is already np.array

    ##
    g_meshes = [d.copy() for d in mesh_list]

    if len(g_meshes) == 0:
        return

    if len(g_meshes) > 40:
        print("Warning: too many meshes ({})".format(len(g_meshes)))
        g_meshes = g_meshes[:40]

    if len(g_meshes) == 0:
        return

    if len(g_meshes) > 40:
        print("Warning: too many meshes ({})".format(len(g_meshes)))
        g_meshes = g_meshes[:40]

    for element in g_meshes:
        if len(element['ver'].shape) == 2:
            # print("## setMeshData: Warning: input size should be (N, verNum, 3). Current input is (verNum, 3). I am automatically fixing this.")
            element['ver'] = element['ver'][np.newaxis, :, :]
            if 'normal' in element.keys():
                element['normal'] = element['normal'][np.newaxis, :, :]

    # Auto computing normal
    if bComputeNormal:
        # print("## setMeshData: Computing face normals automatically.")
        for element in g_meshes:
            element['normal'] = ComputeNormal(element['ver'], element['f'])  # output: (N, 18540, 3)

    # g_frameLimit = g_skeletons.shape[2]
    # mesh_list[0]['ver'].shape
    # frameLens = [l['ver'].shape[0] for l in g_meshes]
    # g_frameLimit = max(g_frameLimit,min(frameLens))

    setFrameLimit()


def setFrameLimit():
    global g_frameLimit
    g_frameLimit = 0

    if g_meshes is not None:
        frameLens = [1] + [l['ver'].shape[0] for l in g_meshes]
        g_frameLimit = max(g_frameLimit, min(frameLens))

    if g_skeletons is not None:
        # frameLens = [1] + [l['skeleton'].shape[1] for l in g_skeletons]
        frameLens = [l['skeleton'].shape[1] for l in g_skeletons]
        if len(frameLens) > 0:
            g_frameLimit = max(g_frameLimit, min(frameLens))

    # if g_skeletons_GT is not None:
    #     frameLens = [1] + [l.shape[1] for l in g_skeletons_GT]
    #     g_frameLimit = max(g_frameLimit,min(frameLens))

    if g_faces is not None:
        frameLens = [1] + [l.shape[1] for l in g_faces]
        g_frameLimit = max(g_frameLimit, min(frameLens))


def DrawPyramid(camWidth, camHeight, camDepth, lineWith=1):
    # glColorMaterial(GL_FRONT, GL_DIFFUSE);
    # glEnable(GL_COLOR_MATERIAL);
    # glColor4f(color.first.x,color.first.y,color.first.z,color.second);

    glLineWidth(lineWith)

    glBegin(GL_LINES)
    glVertex3f(0, 0, 0)
    glVertex3f(camWidth * 0.5, camHeight * 0.5, camDepth * 1)
    glVertex3f(0, 0, 0)
    glVertex3f(camWidth * 0.5, camHeight * -0.5, camDepth * 1)
    glVertex3f(0, 0, 0)
    glVertex3f(camWidth * -0.5, camHeight * 0.5, camDepth * 1)
    glVertex3f(0, 0, 0)
    glVertex3f(camWidth * -0.5, camHeight * -0.5, camDepth * 1)
    glEnd()

    glBegin(GL_LINE_STRIP)
    glVertex3f(camWidth * 0.5, camHeight * 0.5, camDepth * 1)
    glVertex3f(camWidth * -0.5, camHeight * 0.5, camDepth * 1)
    glVertex3f(camWidth * -0.5, camHeight * -0.5, camDepth * 1)
    glVertex3f(camWidth * 0.5, camHeight * -0.5, camDepth * 1)
    glVertex3f(camWidth * 0.5, camHeight * 0.5, camDepth * 1)
    glEnd()
    glDisable(GL_COLOR_MATERIAL)

    glBegin(GL_QUADS)
    # glColor4f(color.first.x,color.first.y,color.first.z,color.second*0.1)
    glVertex3f(camWidth * 0.5, camHeight * 0.5, camDepth * 1)
    glVertex3f(camWidth * -0.5, camHeight * 0.5, camDepth * 1)
    glVertex3f(camWidth * -0.5, camHeight * -0.5, camDepth * 1)
    glVertex3f(camWidth * 0.5, camHeight * -0.5, camDepth * 1)
    glVertex3f(camWidth * 0.5, camHeight * 0.5, camDepth * 1)
    glEnd()
    # glDisable(GL_COLOR_MATERIAL)


def DrawPtCloud():
    # glColor3f(0,1,0)

    glPointSize(g_ptSize);
    glBegin(GL_POINTS)
    for i in range(g_ptCloud.shape[0]):
        glColor3f(g_ptCloudColor[i, 0], g_ptCloudColor[i, 1], g_ptCloudColor[i, 2])
        glVertex3f(g_ptCloud[i, 0], g_ptCloud[i, 1], g_ptCloud[i, 2])
    glEnd()


def DrawCameras():
    if g_cameraPoses is None:
        return

    # glColor3f(0,1,0)
    for i in range(len(g_cameraPoses)):
        glPushMatrix()

        # glTranslatef(g_cameraPoses[i,0],g_cameraPoses[i,1],g_cameraPoses[i,2])
        glTranslatef(g_cameraPoses[i][0], g_cameraPoses[i][1], g_cameraPoses[i][2])
        glMultMatrixd(g_cameraRots[i])
        glutSolidSphere(1, 10, 10)
        DrawPyramid(20, 20, 20)

        glPopMatrix()

    # glColor3f(g_meshColor[0],g_meshColor[1],g_meshColor[2])


def renderscene():
    global g_xRotate, g_rotateView_counter, g_saveFrameIdx

    start = timeit.default_timer()
    # global xrot
    # global yrot
    # global view_dist
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Some anti-aliasing code (seems not working, though)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_LINE_SMOOTH)
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
    # glEnable(GL_POLYGON_SMOOTH)
    glEnable(GL_MULTISAMPLE)
    # glHint(GL_MULTISAMPLE_FILTER_HINT_NV, GL_NICEST)

    # Set up viewing transformation, looking down -Z axis
    glLoadIdentity()
    # gluLookAt(0, 0, -g_fViewDistance, 0, 0, 0, -.1, 0, 0)   #-.1,0,0
    gluLookAt(0, 0, 0, 0, 0, 1, 0, -1, 0)

    if g_viewMode == 'camView':  # Set View Point in MTC Camera View
        # camidlist = ''.join(g_camid)
        # camid = int(camidlist)
        if g_bOrthoCam:
            setCameraViewOrth()
        else:
            setCameraView()

    else:  # Free Mode
        # Set perspective (also zoom)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        # gluPerspective(zoom, float(g_Width)/float(g_Height), g_nearPlane, g_farPlane)
        gluPerspective(65, float(g_Width) / float(g_Height), g_nearPlane,
                       g_farPlane)  # This should be called here (not in the reshpe)
        glMatrixMode(GL_MODELVIEW)
        # Render the scene

        # setFree3DView()
        glTranslatef(0, 0, g_zoom)

        glRotatef(-g_yRotate, 1.0, 0.0, 0.0)
        glRotatef(-g_xRotate, 0.0, 1.0, 0.0)
        glRotatef(g_zRotate, 0.0, 0.0, 1.0)

        # glutSolidSphere(3, 10, 10)        #Draw Origin

        glTranslatef(g_xTrans, 0.0, 0.0)
        glTranslatef(0.0, g_yTrans, 0.0)
        glTranslatef(0.0, 0, g_zTrans)

        glColor3f(0, 1, 0)

    # This should be drawn first, without depth test (it should be always back)
    if g_bShowBackground:
        if g_bOrthoCam:
            DrawBackgroundOrth()
        else:
            DrawBackground()

    glEnable(GL_LIGHTING)
    glEnable(GL_CULL_FACE)
    # drawbody(bodyDatas[cur_ind], connMat_coco19)
    # drawbody_haggling(m_landmarks[:, cur_ind], connMat_coco19)
    # if g_bSaveToFile:

    # #Debug
    # glColor3f(1,0,0)
    # glutSolidTeapot(100, 10, 10)
    # RenderDomeFloor()
    # glutSwapBuffers()
    # return

    # glUseProgram(0)
    glPolygonMode(GL_FRONT, GL_FILL)
    glPolygonMode(GL_BACK, GL_FILL)

    if g_bShowSkeleton:
        DrawSkeletons()
        # DrawSkeletonsGT()
    DrawTrajectory()

    DrawFaces()
    DrawHands()

    if g_bShowMesh:
        DrawMeshes()
    DrawPosOnly()

    glDisable(GL_LIGHTING)
    glDisable(GL_CULL_FACE)

    DrawCameras()
    if g_ptCloud is not None:
        DrawPtCloud()

    if g_bShowFloor:
        RenderDomeFloor()

    global g_frameIdx  # , g_frameLimit
    global g_fps

    # swap the screen buffers for smooth animation
    glutSwapBuffers()

    if g_bRotateView:
        # g_rotateInnterval = 2.0
        g_xRotate += g_rotateInterval

        # print("{0}/rotview_{1:04d}.jpg".format("RENDER_DIR",g_rotateView_counter))

        g_saveFrameIdx = g_rotateView_counter
        g_rotateView_counter += 1

    if g_bSaveToFile:
        SaveScenesToFile()

    g_frameIdx += 1
    # time.sleep(1)
    if g_frameIdx >= g_frameLimit:
        # global g_bSaveOnlyMode
        if g_bSaveOnlyMode:
            # exit opengl
            global g_stopMainLoop
            g_stopMainLoop = True
            g_frameIdx = 0
        else:
            g_frameIdx = 0


def setNearPlane(p):
    global g_nearPlane
    g_nearPlane = p


# Aliasing since the "init_gl" is a bit ugly name
def show(maxIter=-10):
    init_gl(maxIter)
