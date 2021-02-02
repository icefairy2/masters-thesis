import argparse


class ArgumentOptions:

    def __init__(self):
        parser = argparse.ArgumentParser()

        # input options
        parser.add_argument('--input_path', type=str, default=None,
                            help="""Path of video, image, or a folder where image files exists""")
        parser.add_argument('--pkl_dir', type=str, help='Path of storing pkl files that store the predicted results')
        parser.add_argument('--openpose_dir', type=str,
                            help='Directory of storing the prediction of openpose prediction')

        # output options
        parser.add_argument('--out_dir', type=str, default=None, help='Folder of output images.')
        parser.add_argument('--save_pred_pkl', action='store_true',
                            help='Save the predictions (bboxes, params, meshes in pkl format')
        parser.add_argument("--save_mesh", action='store_true', help="Save the predicted vertices and faces")
        parser.add_argument("--save_frame", action='store_true',
                            help='Save the extracted frames from video input or webcam')

        self.parser = parser

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt
