import threading
import tkinter as tk
from tkinter import ttk
from tkinter.font import Font

import numpy as np

from constants import BODY_JOINT_NAMES_DISPLAY, LEFT_HAND_JOINT_NAMES_DISPLAY, \
    RIGHT_HAND_JOINT_NAMES_DISPLAY, BODY_JOINT_IDS_DISPLAY, LEFT_HAND_JOINT_IDS_DISPLAY, RIGHT_HAND_JOINT_IDS_DISPLAY

PARTS_NAMES_DISPLAY = ['body', 'left_hand', 'right_hand']


class ParametersWindow(threading.Thread):

    def __init__(self, body_parameters, left_hand_parameters, right_hand_parameters, continue_frames, stop_frames):
        threading.Thread.__init__(self)
        self.params = {PARTS_NAMES_DISPLAY[0]: np.degrees(np.reshape(body_parameters, (-1, 3)), dtype="float32"),
                       PARTS_NAMES_DISPLAY[1]: np.degrees(np.reshape(left_hand_parameters, (-1, 3)), dtype="float32"),
                       PARTS_NAMES_DISPLAY[2]: np.degrees(np.reshape(right_hand_parameters, (-1, 3)), dtype="float32")}
        self.orig_params = {PARTS_NAMES_DISPLAY[0]: self.params[PARTS_NAMES_DISPLAY[0]].copy(),
                            PARTS_NAMES_DISPLAY[1]: self.params[PARTS_NAMES_DISPLAY[1]].copy(),
                            PARTS_NAMES_DISPLAY[2]: self.params[PARTS_NAMES_DISPLAY[2]].copy()}
        self.constant_conditions = {PARTS_NAMES_DISPLAY[0]: np.zeros(shape=self.params[PARTS_NAMES_DISPLAY[0]].shape,
                                                                     dtype="uint8"),
                                    PARTS_NAMES_DISPLAY[1]: np.zeros(shape=self.params[PARTS_NAMES_DISPLAY[1]].shape,
                                                                     dtype="uint8"),
                                    PARTS_NAMES_DISPLAY[2]: np.zeros(shape=self.params[PARTS_NAMES_DISPLAY[2]].shape,
                                                                     dtype="uint8")}
        self._callbacks = []
        self.continue_frames = continue_frames
        self.stop_frames = stop_frames

        self.joint_names_display = BODY_JOINT_NAMES_DISPLAY
        self.joint_ids_display = BODY_JOINT_IDS_DISPLAY

        self.start()

    def get_body_params(self):
        reformatted_params = np.radians(np.reshape(self.params[PARTS_NAMES_DISPLAY[0]], (1, -1)), dtype="float32")
        return reformatted_params

    def get_left_param_params(self):
        reformatted_params = np.radians(np.reshape(self.params[PARTS_NAMES_DISPLAY[1]], (1, -1)), dtype="float32")
        return reformatted_params

    def get_right_param_params(self):
        reformatted_params = np.radians(np.reshape(self.params[PARTS_NAMES_DISPLAY[2]], (1, -1)), dtype="float32")
        return reformatted_params

    def set_body_params(self, parameters):
        new_params = np.degrees(np.reshape(parameters, (-1, 3)), dtype="float32")
        self.params[PARTS_NAMES_DISPLAY[0]] = np.where(self.constant_conditions[PARTS_NAMES_DISPLAY[0]],
                                                       self.params[PARTS_NAMES_DISPLAY[0]], new_params)

        # wrist correction
        self.params[PARTS_NAMES_DISPLAY[0]][BODY_JOINT_IDS_DISPLAY["L_Wrist"]] = np.where(
            self.params[PARTS_NAMES_DISPLAY[0]][BODY_JOINT_IDS_DISPLAY["L_Wrist"]] < 40,
            self.params[PARTS_NAMES_DISPLAY[0]][BODY_JOINT_IDS_DISPLAY["L_Wrist"]], 0)
        self.params[PARTS_NAMES_DISPLAY[0]][BODY_JOINT_IDS_DISPLAY["R_Wrist"]] = np.where(
            self.params[PARTS_NAMES_DISPLAY[0]][BODY_JOINT_IDS_DISPLAY["R_Wrist"]] < 40,
            self.params[PARTS_NAMES_DISPLAY[0]][BODY_JOINT_IDS_DISPLAY["R_Wrist"]], 0)
        self.params[PARTS_NAMES_DISPLAY[0]][BODY_JOINT_IDS_DISPLAY["L_Wrist"]] = np.where(
            self.params[PARTS_NAMES_DISPLAY[0]][BODY_JOINT_IDS_DISPLAY["L_Wrist"]] > -40,
            self.params[PARTS_NAMES_DISPLAY[0]][BODY_JOINT_IDS_DISPLAY["L_Wrist"]], 0)
        self.params[PARTS_NAMES_DISPLAY[0]][BODY_JOINT_IDS_DISPLAY["R_Wrist"]] = np.where(
            self.params[PARTS_NAMES_DISPLAY[0]][BODY_JOINT_IDS_DISPLAY["R_Wrist"]] > -40,
            self.params[PARTS_NAMES_DISPLAY[0]][BODY_JOINT_IDS_DISPLAY["R_Wrist"]], 0)

        self.orig_params[PARTS_NAMES_DISPLAY[0]] = self.params[PARTS_NAMES_DISPLAY[0]].copy()
        for callback in self._callbacks:
            callback()

    def set_left_hand_params(self, parameters):
        new_params = np.degrees(np.reshape(parameters, (-1, 3)), dtype="float32")
        self.params[PARTS_NAMES_DISPLAY[1]] = np.where(self.constant_conditions[PARTS_NAMES_DISPLAY[1]],
                                                       self.params[PARTS_NAMES_DISPLAY[1]], new_params)
        self.orig_params[PARTS_NAMES_DISPLAY[1]] = self.params[PARTS_NAMES_DISPLAY[1]].copy()
        for callback in self._callbacks:
            callback()

    def set_right_hand_params(self, parameters):
        new_params = np.degrees(np.reshape(parameters, (-1, 3)), dtype="float32")
        self.params[PARTS_NAMES_DISPLAY[2]] = np.where(self.constant_conditions[PARTS_NAMES_DISPLAY[2]],
                                                       self.params[PARTS_NAMES_DISPLAY[2]], new_params)
        self.orig_params[PARTS_NAMES_DISPLAY[2]] = self.params[PARTS_NAMES_DISPLAY[2]].copy()
        for callback in self._callbacks:
            callback()

    def run(self):
        self.root = tk.Tk()

        def stop(event):
            self.root.quit()

        # self.root.bind('<Escape>', stop)

        """""""""""""""""""""""""""""""""""
               POSE PARAMETERS
        """""""""""""""""""""""""""""""""""
        part_var = tk.StringVar(self.root)
        joint_var = tk.StringVar(self.root)
        var_x = tk.DoubleVar(self.root)
        var_y = tk.DoubleVar(self.root)
        var_z = tk.DoubleVar(self.root)

        part_var.set(PARTS_NAMES_DISPLAY[0])
        joint_var.set(self.joint_names_display[0])

        # ************ UTILITIES *****************

        def reset_x():
            var_x.set(self.orig_params[part_var.get()][self.joint_ids_display[joint_var.get()]][0])
            set_x(var_x.get())

        def reset_y():
            var_y.set(self.orig_params[part_var.get()][self.joint_ids_display[joint_var.get()]][1])
            set_y(var_y.get())

        def reset_z():
            var_z.set(self.orig_params[part_var.get()][self.joint_ids_display[joint_var.get()]][2])
            set_z(var_z.get())

        def set_x(a=None, b=None, c=None):
            self.params[part_var.get()][self.joint_ids_display[joint_var.get()]][0] = var_x.get()

        def set_y(a=None, b=None, c=None):
            self.params[part_var.get()][self.joint_ids_display[joint_var.get()]][1] = var_y.get()

        def set_z(a=None, b=None, c=None):
            self.params[part_var.get()][self.joint_ids_display[joint_var.get()]][2] = var_z.get()

        def constant_x():
            self.constant_conditions[part_var.get()][self.joint_ids_display[joint_var.get()]][0] = check_button_x.get()

        def constant_y():
            self.constant_conditions[part_var.get()][self.joint_ids_display[joint_var.get()]][1] = check_button_y.get()

        def constant_z():
            self.constant_conditions[part_var.get()][self.joint_ids_display[joint_var.get()]][2] = check_button_z.get()

        def set_xyz(value):
            joint_var.set(value)
            var_x.set(self.params[part_var.get()][self.joint_ids_display[value]][0])
            var_y.set(self.params[part_var.get()][self.joint_ids_display[value]][1])
            var_z.set(self.params[part_var.get()][self.joint_ids_display[value]][2])
            check_button_x.set(self.constant_conditions[part_var.get()][self.joint_ids_display[joint_var.get()]][0])
            check_button_y.set(self.constant_conditions[part_var.get()][self.joint_ids_display[joint_var.get()]][1])
            check_button_z.set(self.constant_conditions[part_var.get()][self.joint_ids_display[joint_var.get()]][2])

        def set_part(value):
            part_var.set(value)
            if value == PARTS_NAMES_DISPLAY[0]:
                self.joint_ids_display = BODY_JOINT_IDS_DISPLAY
                self.joint_names_display = BODY_JOINT_NAMES_DISPLAY
            if value == PARTS_NAMES_DISPLAY[1]:
                self.joint_ids_display = LEFT_HAND_JOINT_IDS_DISPLAY
                self.joint_names_display = LEFT_HAND_JOINT_NAMES_DISPLAY
            if value == PARTS_NAMES_DISPLAY[2]:
                self.joint_ids_display = RIGHT_HAND_JOINT_IDS_DISPLAY
                self.joint_names_display = RIGHT_HAND_JOINT_NAMES_DISPLAY
            joint_var.set(self.joint_names_display[0])
            set_xyz(joint_var.get())
            reset_option_menu()

        def reset_option_menu():
            menu = self.joint_dropdown["menu"]
            menu.delete(0, "end")
            for string in self.joint_names_display:
                menu.add_command(label=string,
                                 command=lambda value=string: set_xyz(value))

        # ***********************************

        var_x.trace('w', set_x)
        var_y.trace('w', set_y)
        var_z.trace('w', set_z)

        self._callbacks.append(reset_x)
        self._callbacks.append(reset_y)
        self._callbacks.append(reset_z)

        check_button_x = tk.IntVar()
        check_button_y = tk.IntVar()
        check_button_z = tk.IntVar()

        set_xyz(joint_var.get())

        frame0 = tk.Frame(self.root)
        frame0.pack(anchor=tk.W)

        # ************ JOINTS *********************

        text_joint = tk.Label(frame0, text='Joint:', font=Font(family='Helvetica', size=24))
        text_joint.pack(side=tk.LEFT)

        part_dropdown = tk.OptionMenu(frame0, part_var, *PARTS_NAMES_DISPLAY, command=set_part)
        part_dropdown.pack(side=tk.LEFT)

        self.joint_dropdown = tk.OptionMenu(frame0, joint_var, *self.joint_names_display, command=set_xyz)
        self.joint_dropdown.pack(side=tk.LEFT)

        separator0 = ttk.Separator(self.root, orient='horizontal')
        separator0.pack(fill=tk.X)

        # ************ X PARAMETER *****************

        text_x = tk.Label(self.root, text='X parameter', font=Font(family='Helvetica', size=24))
        text_x.pack(anchor=tk.W)

        frame_x = tk.Frame(self.root)
        frame_x.pack(anchor=tk.W)

        box_x = tk.Spinbox(frame_x, from_=-180, to=180, textvariable=var_x, command=set_x,
                           font=Font(family='Helvetica', size=16))
        box_x.pack(side=tk.LEFT)

        reset_button_x = tk.Button(frame_x, text="RESET", command=reset_x)
        reset_button_x.pack(side=tk.LEFT)

        checkbox_x = tk.Checkbutton(frame_x, text="Keep constant X", variable=check_button_x, onvalue=1, offvalue=0,
                                    command=constant_x)
        checkbox_x.pack(side=tk.LEFT)

        slider_x = tk.Scale(self.root, from_=-180, to=180, length=500, variable=var_x, orient=tk.HORIZONTAL)
        slider_x.pack()

        separator1 = ttk.Separator(self.root, orient='horizontal')
        separator1.pack(fill=tk.X)

        # ************ Y PARAMETER *****************

        text_y = tk.Label(self.root, text='Y parameter', font=Font(family='Helvetica', size=24))
        text_y.pack(anchor=tk.W)

        frame_y = tk.Frame(self.root)
        frame_y.pack(anchor=tk.W)

        box_y = tk.Spinbox(frame_y, from_=-180, to=180, textvariable=var_y, command=set_y,
                           font=Font(family='Helvetica', size=16))
        box_y.pack(side=tk.LEFT)

        reset_button_y = tk.Button(frame_y, text="RESET", command=reset_y)
        reset_button_y.pack(side=tk.LEFT)

        checkbox_y = tk.Checkbutton(frame_y, text="Keep constant Y", variable=check_button_y, onvalue=1, offvalue=0,
                                    command=constant_y)
        checkbox_y.pack(side=tk.LEFT)

        slider_y = tk.Scale(self.root, from_=-180, to=180, length=500, variable=var_y, orient=tk.HORIZONTAL, )
        slider_y.pack()

        separator2 = ttk.Separator(self.root, orient='horizontal')
        separator2.pack(fill=tk.X)

        # ************ Z PARAMETER *****************

        text_z = tk.Label(self.root, text='Z parameter', font=Font(family='Helvetica', size=24))
        text_z.pack(anchor=tk.W)

        frame_z = tk.Frame(self.root)
        frame_z.pack(anchor=tk.W)

        box_z = tk.Spinbox(frame_z, from_=-180, to=180, textvariable=var_z, command=set_z,
                           font=Font(family='Helvetica', size=16))
        box_z.pack(side=tk.LEFT)

        reset_button_z = tk.Button(frame_z, text="RESET", command=reset_z)
        reset_button_z.pack(side=tk.LEFT)

        checkbox_z = tk.Checkbutton(frame_z, text="Keep constant Z", variable=check_button_z, onvalue=1, offvalue=0,
                                    command=constant_z)
        checkbox_z.pack(side=tk.LEFT)

        slider_z = tk.Scale(self.root, from_=-180, to=180, length=500, variable=var_z, orient=tk.HORIZONTAL)
        slider_z.pack()
        """""""""""""""""""""""""""""""""""
            BODY POSE PARAMETERS END
        """""""""""""""""""""""""""""""""""

        stop_button = tk.Button(self.root, text="STOP", command=self.stop_frames)
        stop_button.pack(side=tk.RIGHT)

        continue_button = tk.Button(self.root, text="CONTINUE", command=self.continue_frames)
        continue_button.pack(side=tk.RIGHT)

        self.root.mainloop()
