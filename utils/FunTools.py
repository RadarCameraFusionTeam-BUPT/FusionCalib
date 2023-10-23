import numpy as np
from math import *

def lineIntersection(a,b,c,d):
    a,b,c,d=np.array(a),np.array(b),np.array(c),np.array(d)
    denominator=np.cross(b-a,d-c)
    if abs(denominator)<1e-6:
        return False
    x=a+(b-a)*(np.cross(c-a,d-c)/denominator)
    return x

def ccw(p,a,b):
    p,a,b=np.array(p),np.array(a),np.array(b)
    return np.cross(a-p,b-p)

def CalMiddle(a,b,c,d):
    if ccw(a,b,c)*ccw(a,b,d)<0:
        return lineIntersection(a, b, c, d)
    elif ccw(a,c,b)*ccw(a,c,d)<0:
        return lineIntersection(a, c, b, d)
    else:
        return lineIntersection(a, d, b, c)
    
def CalBackMiddle(a,b,c,d):
    now=b
    tmp=b-a
    minval=abs(atan2(tmp[0], tmp[1]))

    tmp=c-a
    if abs(atan2(tmp[0], tmp[1]))<minval:
        minval=abs(atan2(tmp[0], tmp[1]))
        now=c

    tmp=d-a
    if abs(atan2(tmp[0], tmp[1]))<minval:
        minval=abs(atan2(tmp[0], tmp[1]))
        now=d
    return (a+now)/2


def dis(p,q):
    return np.sqrt(np.sum((p-q)**2))

def fpq(p,q,eps):
    tmp=dis(p,q)
    if tmp>eps:
        return 0
    else:
        return 1-tmp/eps

def CATS(t1,t2,tao,eps):
    totalScore=0
    from collections import deque
    q=deque()
    idx=0
    for p in t1:
        while idx<len(t2) and t2[idx]['frame']<=p['frame']+tao:
            q.append(t2[idx])
            idx+=1
        while len(q)>0 and q[0]['frame']<p['frame']-tao:
            q.popleft()

        clueScore=0
        for qq in q:
            clueScore=max(clueScore,fpq(p['pos'],qq['pos'],eps))
        totalScore+=clueScore
    return totalScore/len(t1)
