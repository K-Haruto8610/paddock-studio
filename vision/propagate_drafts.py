"""Short optical-flow propagation of manually drafted training boxes inside the same shot."""
import argparse

import cv2
import numpy as np
import pandas as pd

from .pack import pack_inputs, sample_split
from .check_pack import refs
from .io_local import read_json
from .footage import reader, frames


def advect(a,b,boxes):
    result=[]
    for l,t,r,bot in boxes:
        x=max(0,int(l)); y=max(0,int(t))
        roi=a[y:min(a.shape[0],int(bot)+1),x:min(a.shape[1],int(r)+1)]
        if min(roi.shape)<4:
            continue
        pts=cv2.goodFeaturesToTrack(roi,20,.02,3)
        if pts is None or len(pts)<3:
            continue
        pts+=np.float32([x,y])
        nxt,ok,_=cv2.calcOpticalFlowPyrLK(a,b,pts,None,maxLevel=2)
        back,rev,_=cv2.calcOpticalFlowPyrLK(b,a,nxt,None,maxLevel=2)
        good=ok.ravel().astype(bool)&rev.ravel().astype(bool)&(np.linalg.norm((back-pts)[:,0],axis=1)<1.2)
        if good.sum()>=3:
            delta=np.median((nxt-pts)[good,0],axis=0)
            result.append([l+delta[0],t+delta[1],r+delta[0],bot+delta[1]])
    return np.asarray(result).reshape(-1,4)


def propagate(key):
    root,_,video,crops=pack_inputs(key)
    folder=root/"drafts"
    manual=read_json(folder/"manual_drafts.json")["frames"]
    samples=pd.read_csv(folder/"samples.csv").set_index("frame")
    anchors=[k for k in manual if samples.loc[k,"split"]=="train"]
    cuts=read_json(root.parent/"setup.json")["confirmed_cuts"]
    total=0
    with reader(video) as c:
        seq={}
        for i,_,im in frames(c):
            if any(abs(i-k)<=9 for k in anchors):
                seq[i]=im.copy()
            for k in anchors:
                if i!=k+9:
                    continue
                rect=crops.loc[k].to_numpy()
                l,t,r,b=rect
                boxes=refs(folder/"labels/train"/f"{k}.txt",(b-t,r-l))+[l,t,l,t]
                for direction in [-1,1]:
                    moved=boxes*.5
                    last=cv2.resize(cv2.cvtColor(seq[k],cv2.COLOR_BGR2GRAY),None,fx=.5,fy=.5)
                    for offset in range(1,10):
                        n=k+direction*offset
                        if n not in seq or max(n,n-direction) in cuts or sample_split(n,crops.index[0]) == "val":
                            break
                        gray=cv2.resize(cv2.cvtColor(seq[n],cv2.COLOR_BGR2GRAY),None,fx=.5,fy=.5)
                        moved=advect(last,gray,moved)
                        last=gray
                        if offset not in (4,9) or len(moved)!=len(boxes):
                            continue
                        l,t,r,b=crops.loc[n]
                        local=(2*moved-[l,t,l,t]).clip(0,[r-l,b-t,r-l,b-t])
                        wh=local[:,2:]-local[:,:2]
                        if np.any(wh<2):
                            continue
                        stem=f"flow-{k}-{n}"
                        dest=folder/"labels/train"/(stem+".txt")
                        if dest.exists():
                            continue
                        dims=np.array([r-l,b-t])
                        xy=(local[:,:2]+local[:,2:])/(2*dims)
                        np.savetxt(dest,np.c_[np.zeros(len(xy)),xy,wh/dims],fmt="%.7f")
                        cv2.imwrite(str(folder/"images/train"/(stem+".jpg")),seq[n][t:b,l:r])
                        total+=1
                for old in list(seq):
                    if old<k+9 and not any(j>k and abs(old-j)<=9 for j in anchors):
                        del seq[old]
    print("Additional training frames:",total)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("key")
    propagate(p.parse_args().key)
