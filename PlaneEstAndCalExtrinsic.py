import numpy as np
from utils.Space2Plane import Space2Plane
import pyransac3d as pyrsc
import argparse
import os
import json
import configparser

def PlaneEstimation(points):
    plane = pyrsc.Plane()
    best_eq , best_inliers = plane.fit(points , 0.3 )
    return best_inliers, best_eq

def CalExtrinsicPara(coefficients, vp1, s2p):
    n=np.array(coefficients[:3])
    d=-coefficients[3]
    if d>0:
        d=-d
        n=-n

    po=d/(np.linalg.norm(n)**2)*n
    o_im=s2p.xcyczc2uv([po])[0]
    vp1=np.array(vp1)
    ps_im=o_im+100*(vp1-o_im)/np.linalg.norm(vp1-o_im)

    u0,v0=s2p.cameraMatrix[0,2],s2p.cameraMatrix[1,2]
    fx,fy=s2p.cameraMatrix[0,0],s2p.cameraMatrix[1,1]
    zc=d/(n[0]*(ps_im[0]-u0)/fx+n[1]*(ps_im[1]-v0)/fy+n[2])
    xc=zc*(ps_im[0]-u0)/fx
    yc=zc*(ps_im[1]-v0)/fy
    ps=np.array([xc,yc,zc])

    s=(ps-po)/np.linalg.norm(ps-po)
    m=np.cross(s,n)

    R=np.vstack((m,s,n))
    T=np.array([0,0,np.linalg.norm(po)]).T
    return R, T


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("dataPath",help='Folder which include PlanePoint.npy and config',nargs=1,type=str)
    args = parser.parse_args()

    ####### parameters ##########
    DATA_PATH=args.dataPath[0]

    if not os.path.exists('{}/PlanePoint.npy'.format(DATA_PATH)):
        print('{}/PlanePoint.npy not exist!'.format(DATA_PATH))
        exit(0)
    points=np.load('{}/PlanePoint.npy'.format(DATA_PATH),allow_pickle=True)

    # Fit plane
    indices, coefficients = PlaneEstimation(points)
    if len(indices) == 0:
        print('Could not estimate a planar model for the given dataset.')
    print('Plane coefficients: ' + str(coefficients[0]) + ', ' + str(
            coefficients[1]) + ', ' + str(coefficients[2]) + ', ' + str(coefficients[3]))

    # Calulate extrinsic parameters
    if not os.path.exists('{}/config'.format(DATA_PATH)):
        print('{}/config not exist!'.format(DATA_PATH))
        exit(0)

    config=configparser.ConfigParser()
    config.read('{}/config'.format(DATA_PATH))

    s2p=Space2Plane('{}/config'.format(DATA_PATH))
    vp1=json.loads(config.get('vps', 'vp1'))

    R,T=CalExtrinsicPara(coefficients,vp1,s2p)
    print('Extrinsic parameters:')
    print('R = '+str(R))
    print('T = '+str(T))

    # Wtire results to res file
    res=configparser.ConfigParser()
    if os.path.exists('{}/res'.format(DATA_PATH)):
        res.read('{}/res'.format(DATA_PATH))
    if not 'ours' in res.sections():
        res.add_section('ours')
    res.set('ours','r',json.dumps(R.tolist()))
    res.set('ours','t',json.dumps(T.tolist()))
    with open('{}/res'.format(DATA_PATH),'w+') as f:
        res.write(f)

    print('Write to file: {}/res'.format(DATA_PATH))
