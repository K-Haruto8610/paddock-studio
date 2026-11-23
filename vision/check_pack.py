"""Score one-to-one box matches on visually drafted frames and render owner review sheets."""
import argparse

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .io_local import DATA, read_json, write_json
from .pack import PackModel, pack_inputs


def refs(path, shape):
    labels=np.loadtxt(path,dtype=float,ndmin=2)[:,1:]
    h,w=shape[:2]
    center=labels[:,:2]*[w,h]
    half=labels[:,2:]*[w/2,h/2]
    return np.c_[center-half,center+half]


def correct(reference, boxes):
    if not len(boxes) or not len(reference):
        return 0
    a,b=reference[:,None,:],boxes[None,:,:4]
    intersection=np.maximum(0,np.minimum(a[:,:,2:],b[:,:,2:])-np.maximum(a[:,:,:2],b[:,:,:2])).prod(2)
    area=(a[:,:,2:]-a[:,:,:2]).prod(2)+(b[:,:,2:]-b[:,:,:2]).prod(2)
    iou=intersection/np.maximum(area-intersection,1e-9)
    rows,cols=linear_sum_assignment(np.where(iou>=.5,1-iou,1000))
    return int((iou[rows,cols]>=.5).sum())


def evaluate(key):
    root,*_=pack_inputs(key)
    folder=root/"drafts"
    samples=pd.read_csv(folder/"samples.csv").set_index("frame")
    ids=read_json(folder/"manual_drafts.json")["frames"]
    records=[]
    models=[("pretrained",DATA/"weights/yolo11s.pt"),("tuned",root/"patrol11s.pt")]
    if (root/"patrol11s-reviewed.pt").is_file():
        models.append(("reviewed",root/"patrol11s-reviewed.pt"))
    for name,path in models:
        model=PackModel(path)
        tiles=[]
        for i in ids:
            split=samples.loc[i,"split"]
            im=cv2.imread(str(folder/"images"/split/f"{i}.jpg"))
            ref=refs(folder/"labels"/split/f"{i}.txt",im.shape)
            boxes=model.detect(im,1280,.06)
            for conf in [.06,.1,.15,.22,.3]:
                d=boxes[boxes[:,4]>=conf]
                records.append(dict(model=name,frame=i,split=split,confidence=conf,refs=len(ref),boxes=len(d),tp=correct(ref,d)))
            if name==models[-1][0]:
                for subset,color in [(ref,(0,220,0)),(boxes[boxes[:,4]>=.1,:4],(255,90,210))]:
                    for l,t,r,b in subset.astype(int):
                        cv2.rectangle(im,(l,t),(r,b),color,2)
                scale=520/im.shape[1]
                sm=cv2.resize(im,(520,max(1,round(im.shape[0]*scale))))
                cv2.putText(sm,f"{i} {split}",(5,17),0,.55,(20,255,255),2)
                tiles.append(sm)
                cv2.imwrite(str(folder/f"review-{i}.jpg"),im)
        if tiles:
            cv2.imwrite(str(folder/"owner-review.jpg"),np.vstack(tiles))
    df=pd.DataFrame(records)
    df.to_csv(root/"box_checks.csv",index=False)
    agg=df.groupby(["model","split","confidence"])[["refs","boxes","tp"]].sum()
    agg["precision"]=agg.tp/agg.boxes
    agg["recall"]=agg.tp/agg.refs
    print(agg.to_string())
    write_json(root/"box_checks.json",agg.reset_index().to_dict("records"))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("key")
    evaluate(p.parse_args().key)
