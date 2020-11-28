import threading
import tkinter as tk
from tkinter import ttk
from tkinter.font import Font

import numpy as np

from constants import JOINT_NAMES_DISPLAY, JOINT_IDS_DISPLAY


class ParametersWindow(threading.Thread):

    def __init__(self, parameters):
        threading.Thread.__init__(self)
        self.params = np.degrees(np.reshape(parameters, (-1, 3)), dtype="float32")
        self.orig_params = self.params.copy()
        self.constant_conditions = np.zeros(shape=self.params.shape, dtype="uint8")
        self._callbacks = []
        self.start()

    def get_params(self):
        reformatted_params = np.radians(np.reshape(self.params, (1, -1)), dtype="float32")
        return reformatted_params

    def set_params(self, parameters):
        new_params = np.degrees(np.reshape(parameters, (-1, 3)), dtype="float32")
        self.params = np.where(self.constant_conditions, self.params, new_params)
        self.orig_params = self.params.copy()
        for callback in self._callbacks:
            callback()

    def run(self):
        self.root = tk.Tk()

        def stop(event):
            self.root.quit()

        # self.root.bind('<Escape>', stop)

        joint_var = tk.StringVar(self.root)
        var_x = tk.DoubleVar(self.root)
        var_y = tk.DoubleVar(self.root)
        var_z = tk.DoubleVar(self.root)

        joint_var.set(JOINT_NAMES_DISPLAY[0])

        def reset_x():
            var_x.set(self.orig_params[JOINT_IDS_DISPLAY[joint_var.get()]][0])
            set_x(var_x.get())

        def reset_y():
            var_y.set(self.orig_params[JOINT_IDS_DISPLAY[joint_var.get()]][1])
            set_y(var_y.get())

        def reset_z():
            var_z.set(self.orig_params[JOINT_IDS_DISPLAY[joint_var.get()]][2])
            set_z(var_z.get())

        def set_x(value):
            self.params[JOINT_IDS_DISPLAY[joint_var.get()]][0] = value

        def set_y(value):
            self.params[JOINT_IDS_DISPLAY[joint_var.get()]][1] = value

        def set_z(value):
            self.params[JOINT_IDS_DISPLAY[joint_var.get()]][2] = value

        self._callbacks.append(reset_x)
        self._callbacks.append(reset_y)
        self._callbacks.append(reset_z)

        check_button_x = tk.IntVar()
        check_button_y = tk.IntVar()
        check_button_z = tk.IntVar()

        def constant_x():
            self.constant_conditions[JOINT_IDS_DISPLAY[joint_var.get()]][0] = check_button_x.get()

        def constant_y():
            self.constant_conditions[JOINT_IDS_DISPLAY[joint_var.get()]][1] = check_button_y.get()

        def constant_z():
            self.constant_conditions[JOINT_IDS_DISPLAY[joint_var.get()]][2] = check_button_z.get()

        def set_xyz(value):
            var_x.set(self.params[JOINT_IDS_DISPLAY[value]][0])
            var_y.set(self.params[JOINT_IDS_DISPLAY[value]][1])
            var_z.set(self.params[JOINT_IDS_DISPLAY[value]][2])
            check_button_x.set(self.constant_conditions[JOINT_IDS_DISPLAY[joint_var.get()]][0])
            check_button_y.set(self.constant_conditions[JOINT_IDS_DISPLAY[joint_var.get()]][1])
            check_button_z.set(self.constant_conditions[JOINT_IDS_DISPLAY[joint_var.get()]][2])

        set_xyz(joint_var.get())

        frame0 = tk.Frame(self.root)
        frame0.pack(anchor=tk.W)

        text_joint = tk.Label(frame0, text='Joint:', font=Font(family='Helvetica', size=24))
        text_joint.pack(side=tk.LEFT)

        joint_dropdown = tk.OptionMenu(frame0, joint_var, *JOINT_NAMES_DISPLAY, command=set_xyz)
        joint_dropdown.pack(side=tk.LEFT)

        separator0 = ttk.Separator(self.root, orient='horizontal')
        separator0.pack(fill=tk.X)

        text_x = tk.Label(self.root, text='X parameter', font=Font(family='Helvetica', size=24))
        text_x.pack(anchor=tk.W)

        frame_x = tk.Frame(self.root)
        frame_x.pack(anchor=tk.W)

        box_x = tk.Spinbox(frame_x, from_=-180, to=180, textvariable=var_x,
                           font=Font(family='Helvetica', size=16))
        box_x.pack(side=tk.LEFT)

        reset_button_x = tk.Button(frame_x, text="RESET", command=reset_x)
        reset_button_x.pack(side=tk.LEFT)

        checkbox_x = tk.Checkbutton(frame_x, text="Keep constant X", variable=check_button_x, onvalue=1, offvalue=0,
                                    command=constant_x)
        checkbox_x.pack(side=tk.LEFT)

        slider_x = tk.Scale(self.root, from_=-180, to=180, length=500, variable=var_x, orient=tk.HORIZONTAL,
                            command=set_x)
        slider_x.pack()

        separator1 = ttk.Separator(self.root, orient='horizontal')
        separator1.pack(fill=tk.X)

        text_y = tk.Label(self.root, text='Y parameter', font=Font(family='Helvetica', size=24))
        text_y.pack(anchor=tk.W)

        frame_y = tk.Frame(self.root)
        frame_y.pack(anchor=tk.W)

        box_y = tk.Spinbox(frame_y, from_=-180, to=180, textvariable=var_y,
                           font=Font(family='Helvetica', size=16))
        box_y.pack(side=tk.LEFT)

        reset_button_y = tk.Button(frame_y, text="RESET", command=reset_y)
        reset_button_y.pack(side=tk.LEFT)

        checkbox_y = tk.Checkbutton(frame_y, text="Keep constant Y", variable=check_button_y, onvalue=1, offvalue=0,
                                    command=constant_y)
        checkbox_y.pack(side=tk.LEFT)

        slider_y = tk.Scale(self.root, from_=-180, to=180, length=500, variable=var_y, orient=tk.HORIZONTAL,
                            command=set_y)
        slider_y.pack()

        separator2 = ttk.Separator(self.root, orient='horizontal')
        separator2.pack(fill=tk.X)

        text_z = tk.Label(self.root, text='Z parameter', font=Font(family='Helvetica', size=24))
        text_z.pack(anchor=tk.W)

        frame_z = tk.Frame(self.root)
        frame_z.pack(anchor=tk.W)

        box_z = tk.Spinbox(frame_z, from_=-180, to=180, textvariable=var_z,
                           font=Font(family='Helvetica', size=16))
        box_z.pack(side=tk.LEFT)

        reset_button_z = tk.Button(frame_z, text="RESET", command=reset_z)
        reset_button_z.pack(side=tk.LEFT)

        checkbox_z = tk.Checkbutton(frame_z, text="Keep constant Z", variable=check_button_z, onvalue=1, offvalue=0,
                                    command=constant_z)
        checkbox_z.pack(side=tk.LEFT)

        slider_z = tk.Scale(self.root, from_=-180, to=180, length=500, variable=var_z, orient=tk.HORIZONTAL,
                            command=set_z)
        slider_z.pack()

        self.root.mainloop()
