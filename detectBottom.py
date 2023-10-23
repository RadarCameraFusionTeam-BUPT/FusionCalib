import os
import sys
import random
import math
import numpy as np
import argparse,configparser
import cv2
import json

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = [
    'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush']

def Cal3dBBox( boxes, masks, class_ids, scores, vp):
    N=boxes.shape[0]
    ret=[]
    if not N:
        return ret
    assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    for i in range(N):
        class_id=class_ids[i]
        if class_id not in [2,3,4,6,7,8]:
            continue
        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        now=dict()
        now['box']=boxes[i]
        now['class_id']=class_id
        now['class_name']=class_names[class_id]
        now['score']=scores[i]
        y1, x1, y2, x2 = boxes[i]
        maskvec=[[[y-v[1],x-v[0]] for x in range(x1,x2) for y in range(y1,y2) if masks[y][x][i]] for v in vp]
        
        def CMPF(x,y):
            return math.atan2(x[1],x[0])-math.atan2(y[1],y[0])
        def CMPF1(x,y):
            return math.atan2(x[1],-x[0])-math.atan2(y[1],-y[0])
        
        def lineIntersection(a,b,c,d):
            a,b,c,d=np.array(a),np.array(b),np.array(c),np.array(d)
            denominator=np.cross(b-a,d-c)
            if abs(denominator)<1e-6:
                return False
            x=a+(b-a)*(np.cross(c-a,d-c)/denominator)
            return x

        from functools import cmp_to_key

        for j in range(2):
            maskvec[j].sort(key=cmp_to_key(CMPF))
        maskvec[2].sort(key=cmp_to_key(CMPF1))

        maskvec=np.array(maskvec)
        vp=np.array(vp)
        edg=[[maskvec[i][0][::-1],maskvec[i][-1][::-1]] if abs(math.atan2(maskvec[i][0][1],maskvec[i][0][0]))<abs(math.atan2(maskvec[i][-1][1],maskvec[i][-1][0])) else [maskvec[i][-1][::-1],maskvec[i][0][::-1]] for i in range(2)]
        tmp=[maskvec[2][0][::-1],maskvec[2][-1][::-1]] if abs(math.atan2(maskvec[2][0][1],-maskvec[2][0][0]))<abs(math.atan2(maskvec[2][-1][1],-maskvec[2][-1][0])) else [maskvec[2][-1][::-1],maskvec[2][0][::-1]] 
        edg.append(tmp)

        if edg[0][0][0]*edg[0][-1][0]<0:
            cross1=lineIntersection(vp[1], vp[1]+edg[1][0], vp[2], vp[2]+edg[2][0])
            cross2=lineIntersection(vp[1], vp[1]+edg[1][0], vp[2], vp[2]+edg[2][1])
            if cross1[0]>cross2[0]:
                cross1,cross2=cross2,cross1
            cross5=lineIntersection(vp[0], vp[0]+edg[0][0], vp[1], vp[1]+edg[1][1])
            cross6=lineIntersection(vp[0], vp[0]+edg[0][1], vp[1], vp[1]+edg[1][1])
            if cross5[0]>cross6[0]:
                cross5,cross6=cross6,cross5
            cross3=lineIntersection(vp[0], cross1, vp[2], cross5)
            cross4=lineIntersection(vp[0], cross2, vp[2], cross6)
        elif edg[1][0][0]*edg[1][-1][0]<0:
            cross1=lineIntersection(vp[0], vp[0]+edg[0][0], vp[2], vp[2]+edg[2][0])
            cross2=lineIntersection(vp[0], vp[0]+edg[0][0], vp[2], vp[2]+edg[2][1])
            if cross1[0]>cross2[0]:
                cross1,cross2=cross2,cross1
            cross5=lineIntersection(vp[1], vp[1]+edg[1][0], vp[0], vp[0]+edg[0][1])
            cross6=lineIntersection(vp[1], vp[1]+edg[1][1], vp[0], vp[0]+edg[0][1])
            if cross5[0]>cross6[0]:
                cross5,cross6=cross6,cross5
            cross3=lineIntersection(vp[1], cross1, vp[2], cross5)
            cross4=lineIntersection(vp[1], cross2, vp[2], cross6)
        else:
            cross1=lineIntersection(vp[0], vp[0]+edg[0][0], vp[1], vp[1]+edg[1][0])
            tmp1=lineIntersection(vp[0], vp[0]+edg[0][0], vp[2], vp[2]+edg[2][0])
            tmp2=lineIntersection(vp[1], vp[1]+edg[1][0], vp[2], vp[2]+edg[2][0])
            cross2=tmp1 if tmp1[1]<tmp2[1] else tmp2
            tmp1=lineIntersection(vp[0], vp[0]+edg[0][0], vp[2], vp[2]+edg[2][1])
            tmp2=lineIntersection(vp[1], vp[1]+edg[1][0], vp[2], vp[2]+edg[2][1])
            cross3=tmp1 if tmp1[1]<tmp2[1] else tmp2

            if type(lineIntersection(vp[0], cross1, vp[0], cross2))==bool:
                cross4=lineIntersection(vp[0], cross3, vp[1], cross2)
            else:
                cross4=lineIntersection(vp[0], cross2, vp[1], cross3)

        assert type(cross1)==np.ndarray and type(cross2)==np.ndarray and type(cross3)==np.ndarray and type(cross4)==np.ndarray
        now['bottom']=np.array([cross1,cross2,cross3,cross4]).reshape([-1,2])
        ret.append(now)

    return ret


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataPath', type=str, help='Folder which include output.avi and config',nargs=1)
    parser.add_argument('--model',help='Mask_RCNN model path.',required=True)
    parser.add_argument("--save-vid",help='Draw the detected bottoms on the video and save it to another video file', action='store_true')
    args = parser.parse_args()

    ####### parameters ##########
    DATA_PATH=args.dataPath[0]
    COCO_MODEL_PATH=args.model
    STORE_VID_FLAG=args.save_vid

    ########## Load video and config ###########
    if not os.path.exists('{}/output.avi'.format(DATA_PATH)):
        print('{}/output.avi not exist!'.format(DATA_PATH))
        exit(0)
    if not os.path.exists('{}/config'.format(DATA_PATH)):
        print('{}/config not exist!'.format(DATA_PATH))
        exit(0)

    cap=cv2.VideoCapture('{}/output.avi'.format(DATA_PATH))
    if not cap.isOpened():
        print('Open {}/output.avi failed!'.format(DATA_PATH))
        exit(0)

    config=configparser.ConfigParser()
    config.read('{}/config'.format(DATA_PATH))

    # Load vanishing points
    vp=[json.loads(config.get('vps', 'vp{}'.format(i))) for i in range(1,4)]

    # Create model object in inference mode.
    from mrcnn.config import Config
    class InferenceConfig(Config):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        NAME = 'coco'
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        NUM_CLASSES = 1 + 80

    configCOCO = InferenceConfig()
    configCOCO.display()

    # Create model object in inference mode.
    from mrcnn import utils
    import mrcnn.model as modellib
    model = modellib.MaskRCNN(mode="inference", model_dir='logs', config=configCOCO)

    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    # Solve and save
    if STORE_VID_FLAG:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = cap.get(cv2.CAP_PROP_FPS)

    out=None
    idx=0
    rets=[]
    while cap.isOpened():
        ret,frame=cap.read()
        if not ret:
            break
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results = model.detect([frame], verbose=0)
        r = results[0]
        frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

        ret=Cal3dBBox(r['rois'],r['masks'],r['class_ids'],r['scores'],vp)
        rets.append(ret)
        for item in ret:
            if np.any(item['bottom']<0):
                continue
            for i in range(4):
                for j in range(4):
                    if i==j:
                        continue
                    cv2.line(frame, tuple(item['bottom'][i].astype(int)), tuple(item['bottom'][j].astype(int)), (255,255,255))
        if STORE_VID_FLAG:
            if not out:
                out = cv2.VideoWriter(os.path.join(DATA_PATH,'output-bottom.avi'), fourcc, fps, (width,height))
            out.write(frame)

        print('frame {} finished.'.format(idx))
        idx+=1
    print('savePath=',DATA_PATH)
    rets=np.array(rets,dtype=object)
    np.save(os.path.join(DATA_PATH,'output-bottom.npy'),rets)
    if STORE_VID_FLAG:
        out.release()
