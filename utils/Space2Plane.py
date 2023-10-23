import numpy as np
from math import *
import cv2
import json
import configparser
class Space2Plane:
    """
    Project from space to vedio plane.

    Parameters
    ------------
    config: int / string
        0: Oculii radar to Hikvision 8mm 720p
        1: Lidar to Hikvision 8mm 720p

    """
    def __init__(self,config=0):
        if type(config)==str:
            conf=configparser.ConfigParser()
            conf.read(config)
            self.cameraMatrix=np.array(json.loads(conf.get('cameraIntrinsic','cameramatrix')))
            self.distCoeffs=np.array(json.loads(conf.get('cameraIntrinsic','distcoeffs')))
            self.RT=np.array(json.loads(conf.get('radar2camera','rt')))
            self.R=self.RT[:,:3]
            self.T=self.RT[:,3]
            self.InvR=np.linalg.inv(self.R)
            self.RT=self.RT.T
        else:
            if config==0:
                self.cameraMatrix = np.array([[1.90753223e+03,0.00000000e+00,6.44360570e+02],\
                            [0.00000000e+00,1.98064319e+03,3.65685545e+02],\
                            [0.00000000e+00,0.00000000e+00,1.00000000e+00]],np.float)
                self.distCoeffs=np.array([0.0812306431723410,-0.100591860946000,0,0],np.float)
                self.gamma,self.theta,self.fa=(506-500)/1000*2*pi,(499-500)/1000*2*pi,0
                self.dx,self.dy,self.dz,self.k=0,0,0,1
            elif config==1:
                self.cameraMatrix = np.array([[1.90753223e+03,0.00000000e+00,6.44360570e+02],\
                            [0.00000000e+00,1.98064319e+03,3.65685545e+02],\
                            [0.00000000e+00,0.00000000e+00,1.00000000e+00]],np.float)
                self.distCoeffs=np.array([0.0812306431723410,-0.100591860946000,0,0],np.float)
                self.gamma,self.theta,self.fa=(266-500)/1000*2*pi,pi/2,(495-500)/1000*2*pi
                self.dx,self.dy,self.dz,self.k=0,0,0,1

            m1=np.array([[cos(self.gamma),0,sin(self.gamma)],[0,1,0],[-sin(self.gamma),0,cos(self.gamma)]])
            m2=np.array([[1,0,0],[0,cos(self.theta),-sin(self.theta)],[0,sin(self.theta),cos(self.theta)]])
            m3=np.array([[cos(self.fa),-sin(self.fa),0],[sin(self.fa),cos(self.fa),0],[0,0,1]])
            self.R=self.k*np.matmul(np.matmul(m1,m2),m3)
            self.InvR=np.linalg.inv(self.R)
            self.T=np.array([[self.dx],[self.dy],[self.dz]])
            self.RT=np.c_[self.R,self.T].T

    def ColorFrame(self,frame,row,col):
        """
        Private function. Color frame's pixels located in (row,col).

        Parameters
        ------------
        frame: numpy array
            One frame.

        row,col: digital number
            Color (row,col) in frame.
        """
        if row<self.sideLen or row>=frame.shape[0]-self.sideLen or\
            col<self.sideLen or col>=frame.shape[1]-self.sideLen:
            return
        if np.isnan(row) or np.isnan(col):
            return
        row,col=floor(row),floor(col)
        frame[row-self.sideLen:row+self.sideLen,col-self.sideLen:col+self.sideLen,:]=self.color

    def DrawMeasurementPoint(self,frame,p):
        """
        Draw measurement points.

        Parameters
        --------------
        frame: numpy array
            Frame received from camera.

        p: numpy array
            p.shape=(-1,3), p is measurement points.
        """
        res=cv2.projectPoints(p,self.R,self.T,self.cameraMatrix,self.distCoeffs)[0]
        for item in res:
            self.ColorFrame(frame,item[0][1],item[0][0])

    def DrawBBox(self,frame,p):
        """
        Draw bounding boxes.

        Parameters
        --------------
        frame: numpy array
            Frame received from camera.

        p: numpy array
            p.shape=(-1,3), p is left top quarter and right bottom quarter,
            every 2 point represent one bounding box.
        """
        res=cv2.projectPoints(p,self.R,self.T,self.cameraMatrix,self.distCoeffs)[0]
        for i in range(0,len(res),2):
            try:
                cv2.rectangle(frame,(floor(res[i][0][0]), floor(res[i][0][1])), \
                            (floor(res[i+1][0][0]), floor(res[i+1][0][1])),
                            (0,0,255), thickness = 2 )
            except TypeError:
                pass
            except OverflowError:
                pass
    
    def radar2cameraXYZ(self,data):
        # tmp=np.ones([len(data),1])
        # temp=np.c_[data,tmp]
        # return np.matmul(temp,self.RT)
        return np.matmul(data+self.T.T,self.R.T)

    def cameraXYZ2radar(self,data):
        # tmp=data-self.T.T
        # return np.matmul(tmp,self.InvR.T)
        return np.matmul(data,self.InvR.T)-self.T.T

    def xcyczc2uv(self,data):
        ret=np.array([])
        for item in data:
            zc=item[2]
            ret=np.append(ret,self.cameraMatrix[0,2]+self.cameraMatrix[0,0]*item[0]/zc)
            ret=np.append(ret,self.cameraMatrix[1,2]+self.cameraMatrix[1,1]*item[1]/zc)
        ret=np.reshape(ret,[-1,2])
        return ret

    # def uv2xcyc(self,data,zc):
    #     ret=np.array([])
    #     for item in data:
    #         ret=np.append(ret,(item[0]-self.cameraMatrix[0,2])/self.cameraMatrix[0,0]*zc)
    #         ret=np.append(ret,(item[1]-self.cameraMatrix[1,2])/self.cameraMatrix[1,1]*zc)
    #     ret=np.reshape(ret,[-1,2])
    #     return ret

    def uv2xcyc(self,data,xc1,yc1,zc1,u1,v1):
        ret=np.array([])
        for item in data:
            # ret=np.append(ret,xc1+zc1/self.cameraMatrix[0,0]*(item[0]-u1))
            # ret=np.append(ret,yc1+zc1/self.cameraMatrix[1,1]*(item[1]-v1))
            ret=np.append(ret,zc1/self.cameraMatrix[0,0]*(item[0]-self.cameraMatrix[0,2]))
            ret=np.append(ret,zc1/self.cameraMatrix[1,1]*(item[1]-self.cameraMatrix[1,2]))
        ret=np.reshape(ret,[-1,2])
        return ret

    # def uv2xczc(self,data,yc):
    #     ret=np.array([])
    #     for item in data:
    #         zc=yc*self.cameraMatrix[1,1]/(item[1]-self.cameraMatrix[1,2])
    #         ret=np.append(ret,(item[0]-self.cameraMatrix[0,2])*zc/self.cameraMatrix[0,0])
    #         ret=np.append(ret,zc)
    #     ret=np.reshape(ret,[-1,2])
    #     return ret

    def uv2xczc(self,data,xc1,yc1,zc1,u1,v1):
        ret=np.array([])
        for item in data:
            # zc2=self.cameraMatrix[1,1]*yc1*zc1/((item[1]-v1)*zc1+yc1*self.cameraMatrix[1,1])
            # ret=np.append(ret,((item[0]-u1)/self.cameraMatrix[0,0]+xc1/zc1)*zc2)
            # ret=np.append(ret,zc2)
            zc2=self.cameraMatrix[1,1]*yc1/(item[1]-self.cameraMatrix[1,2])
            ret=np.append(ret,(item[0]-self.cameraMatrix[0,2])/self.cameraMatrix[0,0]*zc2)
            ret=np.append(ret,zc2)
        ret=np.reshape(ret,[-1,2])
        return ret
