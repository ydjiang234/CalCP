import os
import numpy as np
import tempfile
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from CalCP_Class import CalCP

A, I, L, revK = np.loadtxt('Para.txt')

backbone = np.loadtxt('backbone.txt').T
backbone[1] = backbone[1]*1.0e6
targetData = np.loadtxt('targetData.txt').T
targetData[1] = targetData[1] * 1.0e6
ampFactor = 20.0
d_incr = 0.00125
cal = CalCP(A, I, L, revK, backbone, targetData, ampFactor, d_incr, templatePath='./Tcl_Template', workingPath='/Users/JYD/Documents/tmp')

data, fitness = cal.Analyze(np.array([10.0,10.0,10.0]))


#plt.plot(cal.backboneShifted[0], cal.backboneShifted[1])
#plt.plot(cal.backbone[0], cal.backbone[1])
plt.plot(targetData[0], targetData[1])
plt.plot(data[0], data[1])
plt.show()
