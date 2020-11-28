import threading
import tkinter as tk
from tkinter import ttk
from tkinter.font import Font

import numpy as np

from constants import JOINT_NAMES_DISPLAY, JOINT_IDS_DISPLAY


class ParametersWindow(threading.Thread):

    def __init__(self, parameters):
        threading.Thread.__init__(self)
        self.params = np.degrees(np.reshape(parameters, (-1, 3)))
        self._callbacks = []
        self.start()

    def get_params(self):
        reformatted_params = np.radians(np.reshape(self.params, (1, -1)))
        return reformatted_params

    def set_params(self, parameters):
        self.params = np.degrees(np.reshape(parameters, (-1, 3)))
        for callback in self._callbacks:
            callback()

    def run(self):
        self.root = tk.Tk()

        def stop(event):
            self.root.quit()

        # self.root.bind('<Escape>', stop)

        jointVar = tk.StringVar(self.root)
        varX = tk.DoubleVar(self.root)
        varY = tk.DoubleVar(self.root)
        varZ = tk.DoubleVar(self.root)

        jointVar.set(JOINT_NAMES_DISPLAY[0])

        def setXYZ(value):
            varX.set(self.params[JOINT_IDS_DISPLAY[value]][0])
            varY.set(self.params[JOINT_IDS_DISPLAY[value]][1])
            varZ.set(self.params[JOINT_IDS_DISPLAY[value]][2])

        def resetX():
            varX.set(self.params[JOINT_IDS_DISPLAY[jointVar.get()]][0])

        def resetY():
            varY.set(self.params[JOINT_IDS_DISPLAY[jointVar.get()]][1])

        def resetZ():
            varZ.set(self.params[JOINT_IDS_DISPLAY[jointVar.get()]][2])

        def setX(value):
            self.params[JOINT_IDS_DISPLAY[jointVar.get()]][0] = value

        def setY(value):
            self.params[JOINT_IDS_DISPLAY[jointVar.get()]][1] = value

        def setZ(value):
            self.params[JOINT_IDS_DISPLAY[jointVar.get()]][2] = value

        self._callbacks.append(resetX)
        self._callbacks.append(resetY)
        self._callbacks.append(resetZ)

        setXYZ(jointVar.get())

        frame0 = tk.Frame(self.root)
        frame0.pack(anchor=tk.W)

        textJoint = tk.Label(frame0, text='Joint:', font=Font(family='Helvetica', size=24))
        textJoint.pack(side=tk.LEFT)

        jointDropdown = tk.OptionMenu(frame0, jointVar, *JOINT_NAMES_DISPLAY, command=setXYZ)
        jointDropdown.pack(side=tk.LEFT)

        separator0 = ttk.Separator(self.root, orient='horizontal')
        separator0.pack(fill=tk.X)

        textX = tk.Label(self.root, text='X parameter', font=Font(family='Helvetica', size=24))
        textX.pack(anchor=tk.W)

        frameX = tk.Frame(self.root)
        frameX.pack(anchor=tk.W)

        boxX = tk.Spinbox(frameX, from_=-180, to=180, textvariable=varX,
                          font=Font(family='Helvetica', size=16))
        boxX.pack(side=tk.LEFT)

        resetX = tk.Button(frameX, text="RESET", command=resetX)
        resetX.pack(side=tk.LEFT)

        sliderX = tk.Scale(self.root, from_=-180, to=180, length=500, variable=varX, orient=tk.HORIZONTAL, command=setX)
        sliderX.pack()

        separator1 = ttk.Separator(self.root, orient='horizontal')
        separator1.pack(fill=tk.X)

        textY = tk.Label(self.root, text='Y parameter', font=Font(family='Helvetica', size=24))
        textY.pack(anchor=tk.W)

        frameY = tk.Frame(self.root)
        frameY.pack(anchor=tk.W)

        boxY = tk.Spinbox(frameY, from_=-180, to=180, textvariable=varY,
                          font=Font(family='Helvetica', size=16))
        boxY.pack(side=tk.LEFT)

        resetY = tk.Button(frameY, text="RESET", command=resetY)
        resetY.pack(side=tk.LEFT)

        sliderY = tk.Scale(self.root, from_=-180, to=180, length=500, variable=varY, orient=tk.HORIZONTAL, command=setY)
        sliderY.pack()

        separator2 = ttk.Separator(self.root, orient='horizontal')
        separator2.pack(fill=tk.X)

        textZ = tk.Label(self.root, text='Z parameter', font=Font(family='Helvetica', size=24))
        textZ.pack(anchor=tk.W)

        frameZ = tk.Frame(self.root)
        frameZ.pack(anchor=tk.W)

        boxZ = tk.Spinbox(frameZ, from_=-180, to=180, textvariable=varZ,
                          font=Font(family='Helvetica', size=16))
        boxZ.pack(side=tk.LEFT)

        resetZ = tk.Button(frameZ, text="RESET", command=resetZ)
        resetZ.pack(side=tk.LEFT)

        sliderZ = tk.Scale(self.root, from_=-180, to=180, length=500, variable=varZ, orient=tk.HORIZONTAL, command=setZ)
        sliderZ.pack()

        self.root.mainloop()
