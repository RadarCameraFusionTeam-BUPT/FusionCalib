from collections import deque
from matplotlib import pyplot
import numpy as np
from math import *
from mpl_toolkits.mplot3d import Axes3D
import copy
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise

class MeasurePoint:
    def __init__(self,x=0,y=0,z=0,\
                length=0,width=0,height=0,\
                vx=0,vy=0,vz=0,\
                cx=float('inf'),cy=float('inf'),cz=float('inf'),\
                r1=0,c1=0,r2=0,c2=0,\
                typ=None,confidence=0.0,rho=1,P=[0,1,0],\
                T=[0,0,0]):
        self.x,self.y,self.z=x,y,z
        self.vx,self.vy,self.vz=vx,vy,vz
        self.cx,self.cy,self.cz=cx,cy,cz
        self.length=length
        self.width=width
        self.height=height
        self.r1,self.c1=r1,c1
        self.r2,self.c2=r2,c2
        self.typ=typ
        self.confidence=confidence
        self.rho=rho
        self.P=P
        self.T=T

    def __str__(self):
        return 'position:\n--x={0},y={1},z={2}\nsize:\n--length={3},width={4},height={5}\nvelocity:\n--vx={6},vy={7},vz={8}\ntype={9}\nimage:\n--c1={10},r1={11},c2={12},r2={13},conf={14}\ncx={15},cy={16},cz={17}\nrho={18},P={19}\n'\
            .format(self.x,self.y,self.z,\
                    self.length,self.width,self.height,\
                    self.vx,self.vy,self.vz,self.typ,\
                    self.c1,self.r1,self.c2,self.r2,self.confidence,\
                    self.cx,self.cy,self.cz,self.rho,self.P)


class TrackEKF:
    def __init__(self,pStart,s2p):
        self.maxLen=200
        self.trackPoint=deque(maxlen=self.maxLen)
        self.trackPoint.append(pStart)
        self.fail=0
        self.s2p=s2p
        self.fx,self.fy=self.s2p.cameraMatrix[0,0],self.s2p.cameraMatrix[1,1]
        self.u0,self.v0=self.s2p.cameraMatrix[0,2],self.s2p.cameraMatrix[1,2]

        self.ekf = ExtendedKalmanFilter(dim_x=6, dim_z=5)
        self.ekf.x=np.array([pStart.x,pStart.y,pStart.z,pStart.vx,pStart.vy,pStart.vz]).reshape(6,1)

    def AddPoint(self,z,dt):     
        self.PredictNextPos(dt)
        self.update(z,dt)
        ptmp=MeasurePoint()
        ptmp.x,ptmp.y,ptmp.z=self.ekf.x[0:3,0]
        ptmp.vx,ptmp.vy,ptmp.vz=self.ekf.x[3:6,0]

        self.trackPoint.append(ptmp)
    
    def PredictNextPos(self,dt):
        self.ekf.F=np.identity(6)
        for i in range(3):
            self.ekf.F[i][3+i]=dt
        self.ekf.Q=np.array([[dt**4/4,0,0,dt**3/2,0,0],\
                            [0,dt**4/4,0,0,dt**3/2,0],\
                            [0,0,dt**4/4,0,0,dt**3/2],\
                            [dt**3/2,0,0,dt**2,0,0],\
                            [0,dt**3/2,0,0,dt**2,0],\
                            [0,0,dt**3/2,0,0,dt**2]])
        self.ekf.predict()
        
    def update(self,z,dt):
        z=z.reshape([5,1])
        R=np.array([[10,0,0,0,0],\
                    [0,10,0,0,0],\
                    [0,0,1,0,0],\
                    [0,0,0,1,0],\
                    [0,0,0,0,1]])

        def HJacobian(x):
            px,py,pz=x[:3,0]
            vx,vy,vz=x[3:6,0]
            return np.array([[self.fx/pz,0,-px*self.fx/pz**2,0,0,0],\
                            [0,self.fy/pz,-py*self.fy/pz**2,0,0,0],\
                            [0,0,0,1,0,0],\
                            [0,0,0,0,1,0],\
                            [0,0,0,0,0,1]])
        def Hx(x):
            px,py,pz=x[:3,0]
            vx,vy,vz=x[3:6,0]
            u,v=self.s2p.xcyczc2uv([[px,py,pz]])[0]
            ret=np.array([u,v,vx,vy,vz]).reshape([5,1])
            return ret
        self.ekf.update(z,HJacobian,Hx,R)