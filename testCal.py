import os
import sys
import numpy as np
import tempfile
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from CalCP_Class import CalCP
sys.path.insert(0, '../Optimization_Algrithm')
from Genetic_Algorithm import Genetic_Algorithm

A, I, L, revK = np.loadtxt('Para.txt')

backbone = np.loadtxt('backbone.txt').T
backbone[1] = backbone[1]*1.0e6
targetData = np.loadtxt('targetData.txt').T
targetData[1] = targetData[1] * 1.0e6
ampFactor = 20.0
d_incr = 0.00125
cal = CalCP(A, I, L, revK, backbone, targetData, ampFactor, d_incr, templatePath='./Tcl_Template', workingPath='D:/Tmp/')


var_num = 3
var_range = np.array([
    [0.0, 30.0],
    [0.0, 30.0],
    [0.0, 30.0],
    ])
var_digit = [100, 100, 100]
population = 10
mutation_prop = 0.1
tol = 1.0e-03 
max_iter = 1


GA = Genetic_Algorithm(var_range, var_digit, population, cal.fit_fun, cross_num=1, sel_por=0.3, mutation_prop=0.1)
fitness = np.array([])
vector = np.zeros((1,var_num))
for i in range(max_iter):
    GA.evolve()
    vector_max, fitness_max = GA.Optimized()
    print('Iteration {0}, current solution is {1} {2} {3}'.format(i, vector_max[0], vector_max[1], vector_max[2]))
    fitness = np.append(fitness, fitness_max)
    vector = np.vstack((vector, vector_max))
    if i == 0:
        vector = vector[1:]
    if fitness_max >= -tol:
        break

print('Best solution is {1} {2} {3}'.format(i, vector_max[0], vector_max[1], vector_max[2]))
data, fitness1 = cal.Analyze(vector_max)
fig, axes = plt.subplots(1,2)
axes[0].plot(fitness)
axes[1].plot(data[0], data[1])
axes[1].plot(targetData[0], targetData[1])
cal.savePara('./Para', vector_max)


plt.show()
