import argparse
import numpy as np
import cv2

def draw_box(image, box, color):
    """Draw 3-pixel width bounding boxes on the given image array.
    color: list of 3 int values for RGB.
    """
    y1, x1, y2, x2 = box
    image[y1:y1 + 2, x1:x2] = color
    image[y2:y2 + 2, x1:x2] = color
    image[y1:y2, x1:x1 + 2] = color
    image[y1:y2, x2:x2 + 2] = color
    return image

def ColorFrame(image, boxes, ids, class_names, scores=None):
    """
    image: numpy array.
    boxes: [num_instance, (y1, x1, y2, x2)] in image coordinates.
    ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == ids.shape[0]

    # Show area outside image boundaries.
    height, width = image.shape[:2]

    #masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        ID = ids[i]
        color = [0,0,0]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        draw_box(image,[y1,x1,y2,x2],np.array(color)*255)

        # Label
        score = scores[i] if scores is not None else None
        label = class_names[i]
        lab = "{} {}".format(ID,label)
        sco = "{:.3f}".format(score)
        cv2.putText(image, lab, (x1, y1+15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 1)
        cv2.putText(image, sco, (x1, y1+15+23), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 1)

    return image

def Solve(args):
    ####### parameters ##########
    DATA_PATH=args.dataPath[0]

    import os
    if not os.path.exists('{}/output-bottom.npy'.format(DATA_PATH)):
        print('{}/output-bottom.npy not exist!'.format(DATA_PATH))
        exit(0)
    if args.save_vid and (not os.path.exists('{}/output.avi'.format(DATA_PATH))):
        print('{}/output.avi not exist!'.format(DATA_PATH))
        exit(0)
    
    ########## Load npy and video ###########
    npyData=np.load('{}/output-bottom.npy'.format(DATA_PATH),allow_pickle=True)
    
    cap=cv2.VideoCapture('{}/output.avi'.format(DATA_PATH))
    if not cap.isOpened():
        print('Open {}/output.avi failed!'.format(DATA_PATH))
        exit(0)
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    if args.save_vid:
        savePath=os.path.join(DATA_PATH,'output-ByteTrack.avi')
        out = cv2.VideoWriter(savePath, fourcc, 20.0, (width,height))
    
    # Load tracker
    from tracker.byte_tracker import BYTETracker
    tracker = BYTETracker(args)

    # Solve and save
    results=[]
    for ID in range(len(npyData)):
        dets=np.array([np.append(item['box'],item['score']) for item in npyData[ID]])
        
        extraInfo=np.array([{'class_id':item['class_id'],'class_name':item['class_name'],'bottom':item['bottom']} for item in npyData[ID]])

        now=dict()
        if len(dets)>0:
            online_targets = tracker.update(dets, [height,width], [height,width], extraInfo)
        else:
            online_targets = []
        online_tlwhs = []
        online_ids = []
        online_scores = []
        online_class_id = []
        online_class_name = []
        online_bottom = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)
                #print('>>>>>>>>>>>>>>>>>')
                #print(t.extraInfo)
                online_class_id.append(t.extraInfo['class_id'])
                online_class_name.append(t.extraInfo['class_name'])
                online_bottom.append(t.extraInfo['bottom'])
                
        for item in online_tlwhs:
            item[2]+=item[0]
            item[3]+=item[1]

        now['boxes']=np.array(online_tlwhs).astype(int)
        now['ids']=np.array(online_ids)
        now['scores']=np.array(online_scores)
        now['class_ids']=np.array(online_class_id)
        now['class_names']=np.array(online_class_name)
        now['bottoms']=np.array(online_bottom)
        results.append(now)
            
        if args.save_vid:
            ret,frame=cap.read()
            if not ret:
                print('Error: the frame numbers of video and npy are different!!')
                break
            frameDis=ColorFrame(frame,now['boxes'],now['ids'],now['class_names'],now['scores'])
            out.write(frameDis)

        print('frame {} finished'.format(ID))

    savePath=os.path.join(DATA_PATH,'output-ByteTrack.npy')
    np.save(savePath,results)
    if args.save_vid:
        out.release()
    cap.release()
    print('Save to {}'.format(DATA_PATH))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataPath', type=str, help='Folder which include output.avi and output-bottom.npy',nargs=1)
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')

    args = parser.parse_args()

    Solve(args)
