"""Rank/gap/cap assignment across camera changes, with an owner-reviewable mapping."""
import argparse

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .io_local import DATA, read_json, write_json
from .footage import frames, reader
from .track_pack import BOX, EXPERIMENT, source


def ranked(boxes,axis):
    feet=np.c_[(boxes.left+boxes.right)/2,boxes.bottom]
    projected=feet@np.asarray(axis,float)
    order=np.argsort(-projected)
    distance=projected[order[0]]-projected[order]
    relative=distance/max(1,distance[-1])
    return boxes.iloc[order].reset_index(drop=True),relative


def assign(pre,post,axis_pre,axis_post):
    a,sa=ranked(pre,axis_pre)
    b,sb=ranked(post,axis_post)
    ca=a.filter(regex="^hat").to_numpy()
    cb=b.filter(regex="^hat").to_numpy()
    color=1-np.sqrt(ca[:,None]*cb[None]).sum(2)
    rank_a=np.linspace(0,1,len(a))
    rank_b=np.linspace(0,1,len(b))
    geom=.55*np.abs(sa[:,None]-sb)+.40*np.abs(rank_a[:,None]-rank_b)
    cost=geom+.05*color
    padded=np.c_[cost,np.full((len(a),len(a)),.28)]
    ii,jj=linear_sum_assignment(padded)
    optimum=padded[ii,jj].sum()
    pairs=[]
    for i,j in zip(ii,jj):
        if j>=len(b):
            continue
        forbidden=padded.copy()
        forbidden[i,j]=np.inf
        rr,cc=linear_sum_assignment(forbidden)
        margin=forbidden[rr,cc].sum()-optimum
        pairs.append((a.iloc[i],b.iloc[j],float(cost[i,j]),float(margin)))
    return pairs


def run(key,experiment):
    field,video,cfg,root=source(key,experiment)
    df=pd.read_csv(root/"tracks.csv")
    directions=read_json(root.parent/"cut_axes.json")
    wanted = {s["frame"] + offset for s in directions for offset in (-1, 0)}
    with reader(video) as cap:
        pictures = {i: im for i, _, im in frames(cap) if i in wanted}
    out=root/"cuts"
    out.mkdir(exist_ok=True)
    names={int(h):int(h) for h in field.index}
    all_rows=[]
    for spec in directions:
        frame=spec["frame"]
        before=df[(df.frame==frame-1)&(df.direction=="reverse")&(df.evidence=="anchor")]
        direct=df[(df.frame==frame-1)&(df.direction=="forward")&(df.box_id>=0)]
        before_ids=dict(zip(direct.box_id,direct.id))
        after=df[(df.frame==frame)&(df.direction=="forward")&(df.evidence=="anchor")]
        preimage, postimage = pictures[frame-1], pictures[frame]
        pairs=assign(before,after,spec["pre"],spec["post"])
        points=[]
        next_names={}
        for k,(a,b,cost,margin) in enumerate(pairs,1):
            local=before_ids.get(a.box_id)
            horse=names.get(local)
            u,v=(b.left+b.right)/2,(b.top+b.bottom)/2
            if horse is not None:
                next_names[int(b.id)]=horse
                points.append([horse,round(u),round(v)])
            row=dict(frame=frame, pair=k,pre_reverse_id=int(a.id),post_forward_id=int(b.id),
                     pre_forward_id=int(local) if local is not None else None,proposed_horse=horse,
                     u=round(u),v=round(v),cost=round(cost,4),alternative_cost_margin=round(margin,4),
                     confirmed=False)
            all_rows.append(row)
            for image,d in [(preimage,a),(postimage,b)]:
                l,t,r,bot=d[BOX].to_numpy(dtype=float).astype(int)
                color=(0,220,255) if horse is not None else (255,180,70)
                cv2.rectangle(image,(l,t),(r,bot),color,2)
                cv2.putText(image,f"{k}:{horse or '?'}",(l,max(18,t)),0,.65,color,2)
        names=next_names
        images=[cv2.resize(im,(960,540)) for im in (preimage,postimage)]
        cv2.imwrite(str(out/f"cut-{frame}.jpg"),np.vstack(images))
    write_json(out/"identity_anchors.draft.json",dict(status="owner_review_pending",pairs=all_rows))
    print(pd.DataFrame(all_rows).groupby("frame").agg(pairs=("pair","size"),horse_number_drafts=("proposed_horse","count")))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("key");p.add_argument("--experiment",default=EXPERIMENT)
    args=p.parse_args();run(args.key,args.experiment)
