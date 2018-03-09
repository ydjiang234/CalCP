import os
import string
import random
import numpy as np
import subprocess
from scipy.interpolate import interp1d

class CalCP:

    def __init__(self, A, I, L, revK, backbone, targetData, ampFactor, d_incr, templatePath='./Tcl_Template', workingPath='./Working'):
        self.A = A
        self.I = I
        self.L = L
        self.revK = revK
        self.backbone = backbone
        self.targetX = targetData[0]
        self.targetY = targetData[1]
        self.ampFactor = ampFactor
        self.d_incr = d_incr
        self.considerNum = 10
        self.workingPath = workingPath
        self.templatePath = templatePath
        self.cS= 1.0
        self.cC = 1.0
        self.cA = 1.0
        self.cK = 1.0
        self.D = 1.0
        self.initialize()

    def initialize(self):
        #stiffness of the elastic member
        self.K = self.backbone[1,1] / self.backbone[0,1]
        self.E = self.K * self.L / self.I / 3.0
        #To get the property of the CP backbone curve
        self.shiftBackbone()
        self.thetay = self.backboneShifted[0,1]
        self.Fy = self.backboneShifted[1,1]
        self.thetac = self.backboneShifted[0,2]
        self.Fc = self.backboneShifted[1,2]
        self.thetau = self.backbone[0,3] #here should be the backbone instead of backboneShifted
        self.K0 = self.Fy / self.thetay
        self.K0amp = self.K0 * (self.ampFactor + 1.0)
        self.thetap = self.thetac - self.thetay
        self.thetapc = self.backboneShifted[0,3] - self.thetac
        self.a_s = (self.Fc - self.Fy) / self.thetap / self.K0 #be carefule, use K0 or K0amp
        self.a_samp = self.a_s * (self.ampFactor + 1.0) #be carefule, use K0 or K0amp
        #for the reverse material
        self.revKamp = self.revK
        #other
        self.ind, self.dataTX, self.dataTY = self.findTurning(self.targetX, self.targetY)
        self.convertDisp();
        self.preRender();
        #initial backbone
        self.BBcyclicX, self.BBcyclicY = self.monoData(self.dataTX, self.dataTY)
        self.xRange = np.linspace(self.BBcyclicX[0], self.BBcyclicX[-1], self.considerNum)

    def preRender(self):
        #load template
        filePath = '{0}/ModifiedCPTemplate.tcl'.format(self.templatePath)
        f = open(filePath, 'r')
        self.template = f.read()
        f.close()
        #pre-render template
        replaceContents = {\
            '{{E}}': '{0}'.format(self.E),
            '{{A}}':'{0}'.format(self.A),
            '{{I}}':'{0}'.format(self.I),
            '{{revE}}':'{0}'.format(self.revKamp),
            '{{ampFactor}}':'{0}'.format(self.ampFactor),
            '{{d_incr}}':'{0}'.format(self.d_incr),
            '{{DispList}}':'{0}'.format(' '.join(self.DispList)),
        }
        for key in replaceContents:
            self.template = self.template.replace(key, replaceContents[key])

    def shiftBackbone(self):
        self.backboneShifted = np.vstack((self.backbone[0], self.backbone[0] * self.revK + self.backbone[1]))
        self.Res = self.backboneShifted[1,-1] / self.backboneShifted[1,-2]
        x1 = self.backboneShifted[0,2]
        y1 = self.backboneShifted[1,2]
        x2 = self.backboneShifted[0,3]
        y2 = self.backboneShifted[1,3]
        x3 = x1 - y1 / (y1 - y2) * (x1 - x2)
        self.backboneShifted[0,3] = x3
        self.backboneShifted[1,3] = 0.0

    def CP_cmdLine(self, vector):
        lambda_S, lambda_C, lambda_A, lambda_K = vector[0], vector[1], vector[0], vector[2]
        tempList = [self.K0, self.a_s, self.a_s, self.Fy, -self.Fy, lambda_S, lambda_C, lambda_A, lambda_K, self.cS, self.cC, self.cA, self.cK, self.thetap, self.thetap, self.thetapc, self.thetapc, self.Res, self.Res, self.thetau, self.thetau, self.D, self.D]
        return ' '.join([str(item) for item in tempList])

    def savePara(self, filePath, vector):
        lambda_S, lambda_C, lambda_A, lambda_K = vector[0], vector[1], vector[0], vector[2]
        tempList = np.array([[self.K0amp, self.a_samp, self.a_s, self.Fy, -self.Fy, lambda_S, lambda_C, lambda_A, lambda_K, self.cS, self.cC, self.cA, self.cK, self.thetap, self.thetap, self.thetapc, self.thetapc, self.Res, self.Res, self.thetau, self.thetau, self.D, self.D]])
        np.savetxt('{0}_IMK.out'.format(filePath), tempList)
        tempList = np.array([[self.E, self.A, self.I]])
        np.savetxt('{0}_El.out'.format(filePath), tempList)
        tempList = np.array([self.revKamp])
        np.savetxt('{0}_revK.out'.format(filePath), tempList)
        
        
    def renderTemplate(self, curID, vector):
        filePath = '{0}/{1}'.format(self.workingPath, curID)
        temp = self.template
        #replace parameters
        replaceContents = {\
            '{{CP_CMDLine}}': self.CP_cmdLine(vector),
            '{{outname}}': filePath,
        }

        for key in replaceContents:
            temp = temp.replace(key, replaceContents[key])
        f = open('{0}/{1}.tcl'.format(self.workingPath, curID), 'w')
        f.write(temp)
        f.close()

    def runOpenSees(self, curID):
        filePath = '{0}/{1}'.format(self.workingPath, curID)
        subprocess.check_call('opensees {0}.tcl'.format(filePath), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        dataX = np.loadtxt('{0}_rotation.out'.format(filePath)) * -1.0
        dataY = np.loadtxt('{0}_moment.out'.format(filePath))
        try:
            os.remove('{0}.tcl'.format(filePath))
            os.remove('{0}_rotation.out'.format(filePath))
            os.remove('{0}_moment.out'.format(filePath))
        except:
            pass
        return dataX, dataY

    def Analyze(self, vector):
        curID = self.id_generator()
        self.renderTemplate(curID, vector)
        dataX, dataY = self.runOpenSees(curID)
        fitness = self.Fitness(dataX, dataY)
        return np.vstack((dataX, dataY)), fitness

    def fit_fun(self, vector):
        output, fitness = self.Analyze(vector)
        return fitness

    def Fitness(self, dataX, dataY):
        ind, data1, data2 = self.findTurning(dataX, dataY)
        data1, data2 = self.monoData(data1, data2)
        data1, data2 = self.simplifyData(data1, data2, self.BBcyclicX)
        fitness = -1.0 * np.sum(np.abs(data2 - self.BBcyclicY))
        return fitness

    def convertDisp(self):
        self.DispList = [str(item) for item in self.dataTX]

    def getEnergy(self, dataX, dataY):
        data_d = dataX[1:] - dataX[:-1]
        temp1 = np.cumsum(np.abs(data_d))
        temp2 = np.cumsum(data_d * (dataY[1:] - dataY[:-1]) / 2.0)
        return np.vstack((temp1, temp2))

    def simplifyData(self, dataX, dataY, xRange):
        outX = xRange
        f = interp1d(dataX, dataY)
        outY = f(xRange)
        return outX, outY

    def monoData(self, dataX, dataY):
        output = np.array([dataX[0], dataY[0]])
        pre_x = dataX[0]
        for i in range(1,dataX.size):
            if dataX[i] > pre_x:
                output = np.vstack((output, [dataX[i], dataY[i]]))
                pre_x = dataX[i]

        return output[:,0], output[:,1]
    
    def findTurning(self, dataX, dataY):
        x_temp = dataX[1:] - dataX[:-1]
        pre_dx = x_temp[0]
        i = 2
        output = np.array([0.0, dataX[0], dataY[0]])
        while i<dataX.size-2:
            if pre_dx != 0.0:
                cur_dx = x_temp[i]
                if pre_dx * cur_dx < 0:
                    output = np.vstack((output, [i, dataX[i], dataY[i]]))
                    pre_dx = cur_dx
                i = i + 1
            else:
                pre_dx = x_temp[i]
        output = np.vstack((output, [dataX.size-1, dataX[-1], dataY[-1]]))
        return output[:,0], output[:,1], output[:,2]

    def id_generator(self, size=6, chars=string.ascii_uppercase + string.digits):
        return ''.join(random.choice(chars) for _ in range(size))
