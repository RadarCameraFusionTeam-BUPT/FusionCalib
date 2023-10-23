import numpy as np
from math import *
from utils.FunTools import *
import argparse
import os
import configparser

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("dataPath",help='Folder which include radarTrack.npy and images-bottom-ByteTrack.npy',nargs=1,type=str)
    parser.add_argument("-s", "--start_frame", help='Start frame which need to be processed', default=0)  
    parser.add_argument("-e", "--end_frame", help='End frame which need to be processed', default=600)  
    parser.add_argument("--minv", help='Minimum row index of the image which is considered', default=0)  
    parser.add_argument("--maxv", help='Maximum row index of the image which is considered', default=700)  
    parser.add_argument("--tao", help='CATS parameter tao', default=3)  
    parser.add_argument("--eps", help='CATS parameter eps', default=20)  
    parser.add_argument("--visualize",help='Show ground plane points', action='store_true')
    args = parser.parse_args()

    ####### parameters ##########
    DATA_PATH=args.dataPath[0]
    # start and end frame
    start,end=int(args.start_frame),int(args.end_frame)
    # range of v which can distinguish bottom
    minv,maxv=int(args.minv),int(args.maxv)
    # CATS parameters
    tao,eps=int(args.tao),float(args.eps)

    ####### parameters ##########

    if not os.path.exists('{}/radarTrack.npy'.format(DATA_PATH)):
        print('{}/radarTrack.npy not exist!'.format(DATA_PATH))
        exit(0)
    if not os.path.exists('{}/output-ByteTrack.npy'.format(DATA_PATH)):
        print('{}/output-ByteTrack.npy not exist!'.format(DATA_PATH))
        exit(0)
    if not os.path.exists('{}/config'.format(DATA_PATH)):
        print('{}/config not exist!'.format(DATA_PATH))
        exit(0)
    radarTrack=np.load('{}/radarTrack.npy'.format(DATA_PATH),allow_pickle=True)
    imageTrack=np.load('{}/output-ByteTrack.npy'.format(DATA_PATH),allow_pickle=True)

    maxlen=len(radarTrack)
    end=min(end,maxlen)

    ################### CATS factors ################ 
    imageTrackID=dict()
    radarTrackID=dict()

    for i in range(start,end):
        for j,item in enumerate(imageTrack[i]['ids']):
            a,b,c,d=[imageTrack[i]['bottoms'][j][k] for k in range(4)]
            pos=CalBackMiddle(a,b,c,d)
            # a,b,c,d=[imageTrack[i]['boxes'][j][k] for k in range(4)]
            # pos=np.array([(b+d)/2,c])
            now={'frame':i,'pos':pos}
            
            imageTrackID[item]=np.array([now]) if not item in imageTrackID.keys() else np.append(imageTrackID[item],now)

        for j,item in enumerate(radarTrack[i]['data']):
            vx,vy,vz=item['vx'],item['vy'],item['vz']
            if sqrt(vx**2+vy**2+vz**2)<0.01:
                continue
            ID,x,y,z=item['id'],item['x'],item['y'],item['z']
            now={'frame':i,'pos':np.array([x,y,z]),'time':radarTrack[i]['time'],'v':np.array([item['vx'],item['vy'],item['vz']])}

            radarTrackID[ID]=np.array([now]) if not ID in radarTrackID.keys() else np.append(radarTrackID[ID],now)


    ################# project radar to image ######################
    from utils.Space2Plane import Space2Plane
    s2p=Space2Plane(config='{}/config'.format(DATA_PATH))
    fx,fy=s2p.cameraMatrix[0,0],s2p.cameraMatrix[1,1]
    u0,v0=s2p.cameraMatrix[0,2],s2p.cameraMatrix[1,2]
    for ID in radarTrackID.keys():
        for item in radarTrackID[ID]:
            item['posRep']=s2p.radar2cameraXYZ([item['pos']])[0]
            item['pos']=s2p.xcyczc2uv(s2p.radar2cameraXYZ([item['pos']]))[0]
            item['v']=s2p.radar2cameraXYZ([item['v']])[0]

    ###################### CATS ##################################
    img2radar=dict()
    radar2img=dict()
    import time
    for idImg in imageTrackID.keys():
        now=[CATS(imageTrackID[idImg], radarTrackID[idRadar], tao, eps) for idRadar in radarTrackID.keys()]
        IDs=list(radarTrackID.keys())
        if np.max(now)<0.3:
            img2radar[idImg]=-1
        else:
            img2radar[idImg]=IDs[np.argmax(now)]

    for idRadar in radarTrackID.keys():
        now=[CATS(radarTrackID[idRadar], imageTrackID[idImg], tao, eps) for idImg in imageTrackID.keys()]
        IDs=list(imageTrackID.keys())
        if np.max(now)<0.3:
            radar2img[idRadar]=-1
        else:
            radar2img[idRadar]=IDs[np.argmax(now)]

    tmpimg2radar=dict()
    tmpradar2img=dict()
    for idImg in img2radar.keys():
        if img2radar[idImg]!=-1 and idImg==radar2img[img2radar[idImg]]:
            tmpimg2radar[idImg]=img2radar[idImg]
            tmpradar2img[img2radar[idImg]]=idImg
    img2radar,radar2img=tmpimg2radar,tmpradar2img
    imgID=list(img2radar.keys())
    radarID=list(radar2img.keys())

    ############ track ################
    xs,ys,zs=[],[],[]
    xrs,yrs,zrs=[],[],[]
    dis=[]

    from utils.Track import *
    from math import *

    def inboard(u,v):
        return v>=minv and v<=maxv

    for idImg in imgID:
        print(idImg)
        nowImg=imageTrackID[idImg]
        idRadar=img2radar[idImg]
        nowRadar=radarTrackID[idRadar]
        idx1,idx2=0,0

        lastTime=-1
        while idx1<len(nowImg) and idx2<len(nowRadar):
            if nowImg[idx1]['frame']==nowRadar[idx2]['frame']:
                # if nowRadar[idx2]['posRep'][2]>120:
                #     idx1+=1
                #     idx2+=1
                #     continue
                u,v=nowImg[idx1]['pos']
                if not inboard(u,v):
                    idx1+=1
                    idx2+=1
                    continue
                vxR,vyR,vzR=nowRadar[idx2]['v']
                vxc,vyc,vzc=s2p.radar2cameraXYZ([[vxR,vyR,vzR]])[0]
                xR,yR,zR=nowRadar[idx2]['posRep']
                xrs.append(xR)
                yrs.append(yR)
                zrs.append(zR)
                if lastTime!=-1:
                    dt=nowRadar[idx2]['time']-lastTime
                    tk.AddPoint(np.array([u,v,vxc,vyc,vzc]),dt)
                    rho=tk.trackPoint[-1].rho
                    
                    px,py,pz=tk.trackPoint[-1].x,tk.trackPoint[-1].y,tk.trackPoint[-1].z
                    vx,vy,vz=tk.trackPoint[-1].vx,tk.trackPoint[-1].vy,tk.trackPoint[-1].vz

                    dd=sqrt((px-xR)**2+(py-yR)**2+(pz-zR)**2)
                    dis.append(dd)
                    xs.append(px)
                    ys.append(py)
                    zs.append(pz)
                else:
                    tk=TrackEKF(MeasurePoint(x=xR,y=yR,z=zR,vx=vxR,vy=vyR,vz=vzR),s2p)

                lastTime=nowRadar[idx2]['time']
                idx1+=1
                idx2+=1
            elif nowImg[idx1]['frame']<nowRadar[idx2]['frame']:
                idx1+=1
            else:
                idx2+=1

    result=np.vstack((xs,ys,zs)).T
    np.save('{}/PlanePoint.npy'.format(DATA_PATH),result,allow_pickle=True)

    if args.visualize:
        import matplotlib.pyplot as plt
        ax = plt.gca(projection="3d")
        ax.scatter(xs,ys,zs)
        plt.show()
        ax = plt.gca(projection="3d")
        ax.scatter(xrs,yrs,zrs)
        plt.show()
        plt.plot(range(len(dis)),dis)
        plt.show()
    ########### track ################
