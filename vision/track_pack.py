"""Bidirectional shot-local trials. Predictions, merged observations and draft IDs stay explicit."""
import argparse
from collections import Counter
from dataclasses import dataclass
import time

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .footage import reader, frames, ranges
from .io_local import DATA, race_input, read_json, write_json, mykeibadb_results

BOX = ["left","top","right","bottom"]
EXPERIMENT = "patrol11s-reviewed-1280-0.1"


def cap_hist(image, box):
    l,t,r,b = box
    w,h = r-l,b-t
    hh,ww=image.shape[:2]
    left,right=np.clip([l+.25*w,l+.75*w],0,ww).astype(int)
    top,bottom=np.clip([t-.19*h,t+.12*h],0,hh).astype(int)
    area=image[top:bottom,left:right]
    if area.size == 0:
        return np.zeros(14)
    hsv=cv2.cvtColor(area,cv2.COLOR_BGR2HSV)
    hue,sat,val=hsv.transpose(2,0,1)
    bins=(hue//15).clip(0,11)
    bins=np.where(val<57,12,np.where((sat<51)&(val>135),13,bins))
    counts=np.bincount(bins.ravel(),minlength=14).astype(float)
    return counts/counts.sum()


def ious(a,b):
    a=np.atleast_2d(a)[:,None,:4]
    b=np.atleast_2d(b)[None,:,:4]
    intersect=(np.minimum(a[:,:,2:],b[:,:,2:])-np.maximum(a[:,:,:2],b[:,:,:2])).clip(min=0).prod(2)
    return intersect/np.maximum((a[:,:,2:]-a[:,:,:2]).prod(2)+(b[:,:,2:]-b[:,:,:2]).prod(2)-intersect,1.)


def source(key, experiment=EXPERIMENT):
    code,field,video=race_input(key)
    folder=DATA/"races"/code
    detdir=folder/"pack"/experiment
    return field,video,read_json(folder/"setup.json"),detdir


def evidence(video, detdir):
    table=pd.read_csv(detdir/"detections.csv")
    by_frame={i:d[BOX+["score"]].to_numpy() for i,d in table.groupby("frame")}
    arrays={}
    with reader(video) as cap:
        for i,_,image in frames(cap):
            det=by_frame.get(i)
            if det is None:
                continue
            xywh=det[:,:4].copy()
            xywh[:,2:]-=xywh[:,:2]
            keep=np.asarray(cv2.dnn.NMSBoxes(xywh.tolist(),det[:,4].tolist(),.099,.43),int).ravel()
            det=det[keep]
            colors=np.asarray([cap_hist(image,row[:4]) for row in det]).reshape(-1,14)
            arrays[i]=np.column_stack([det,colors])
    return arrays


@dataclass
class State:
    box: np.ndarray
    velocity: np.ndarray
    last_seen: float
    color: np.ndarray


class MotionTracks:
    def __init__(self):
        self.states={}

    def seed(self, horse, box, color, t):
        self.states[horse]=State(box.copy(),np.zeros(4),t,color.copy())

    def match(self, det, t):
        ids=[h for h,s in self.states.items() if abs(t-s.last_seen)<=1.000001]
        if not ids:
            return {}
        predicted=np.array([self.states[h].box+self.states[h].velocity*(t-self.states[h].last_seen) for h in ids])
        result={}
        for h,p in zip(ids,predicted):
            result[h]=(p,"predicted",-1)
        if not len(det):
            return result
        iou=ious(predicted,det)
        dims=(predicted[:,2:]-predicted[:,:2]).clip(min=3)
        centers=(predicted[:,2:]+predicted[:,:2])*.5
        actual=(det[:,2:4]+det[:,:2])*.5
        dist=np.linalg.norm((centers[:,None]-actual[None])/dims[:,None],axis=2)
        ratios=(det[:,2:4]-det[:,:2]).clip(min=2)[None]/dims[:,None]
        geom=.6*(1-iou)+.36*dist+.14*np.abs(np.log(ratios)).mean(axis=2)
        colors=np.array([self.states[h].color for h in ids])
        color_cost=1-np.sqrt(colors[:,None]*det[None,:,5:]).sum(2)
        cost=geom+.045*color_cost+.03*(1-det[:,4])
        cost=np.where((dist<1.1)&(geom<.96),cost,10.)
        # An unmatched slot costs less than a geometrically implausible box.
        n=len(ids)
        a,b=linear_sum_assignment(np.column_stack([cost,np.full((n,n),.96)]))
        observed=set()
        for j,k in zip(a,b):
            if k>=len(det):
                continue
            h=ids[j]; old=self.states[h]; elapsed=t-old.last_seen
            box=det[k,:4]
            if abs(elapsed)>1e-7:
                vel=(box-old.box)/elapsed
                shift=(vel[:2]+vel[2:])*.5
                old.velocity=.65*old.velocity+.35*np.tile(shift,2)
            old.box=box.copy(); old.last_seen=t
            if iou[j,k]>.3:
                old.color=.85*old.color+.15*det[k,5:]
            result[h]=(box,"observed",int(k))
            observed.add(k)
        for j,h in enumerate(ids):
            if result[h][1]=="predicted" and any(iou[j,k]>.18 for k in observed):
                result[h]=(predicted[j],"merged",-1)
        return result


def seed_list(det, field):
    # Shot-local indices are anonymous. Only a verified gate or finish gives a horse number.
    order=np.argsort(-det[:,4])[:field]
    return [(j+1,det[k,:4],det[k,5:]) for j,k in enumerate(order)]


def run_direction(indices, seeds, observations, clock, shot, direction):
    tr=MotionTracks()
    records=[]
    for i in indices:
        t=clock.at[i]
        det=observations.get(i,np.empty((0,19)))
        anchors = seeds.get(i, [])
        reserved = {}
        if anchors and len(det):
            overlaps = ious(np.array([s[1] for s in anchors]), det)
            rows, cols = linear_sum_assignment(np.where(overlaps > .5, 1-overlaps, 1.e4))
            reserved = {anchors[j][0]: int(k) for j, k in zip(rows, cols) if overlaps[j,k] > .5}
        available = np.array([k for k in range(len(det)) if k not in reserved.values()], dtype=int)
        for horse, _, _ in anchors:
            tr.states.pop(horse, None)
        matched = {h: (box, state, int(available[k]) if k >= 0 else -1)
                   for h, (box, state, k) in tr.match(det[available], t).items()}
        for h,box,color in anchors:
            tr.seed(h,box,color,t)
            matched[h]=(box,"anchor",reserved.get(h, -1))
        for h,(box,state,k) in matched.items():
            records.append((shot,i,t,direction,h,state,k,*box,*tr.states[h].color))
    return records


def compare(fwd,rev):
    common=fwd[fwd.evidence=="observed"].merge(rev[rev.evidence=="observed"],on=["frame","box_id","shot"])
    rows=[]; votes=[]
    for shot,part in common.groupby("shot"):
        tab=pd.crosstab(part.id_x,part.id_y)
        a,b=linear_sum_assignment(-tab.to_numpy())
        mapping={tab.columns[j]:tab.index[i] for i,j in zip(a,b) if tab.iloc[i,j]>0}
        agreed=part.id_y.map(mapping)==part.id_x
        for rid,fid in mapping.items():
            votes.append(dict(shot=int(shot),reverse_id=int(rid),forward_id=int(fid),votes=int(tab.loc[fid,rid])))
        for h,grp in part[~agreed].groupby("id_x"):
            for start,end in ranges(sorted(grp.frame.unique())):
                rows.append(dict(shot=int(shot),local_id=int(h),start=int(start),end=int(end)))
    return rows,votes,len(common)


def finish_seeds(crossings, ranks, field, obs, start, end):
    if crossings.get("confirmed") is not True or set(ranks) != set(field.index):
        return {}
    by_rank = {rank: horse for horse, rank in ranks.items()}
    if len(by_rank) != len(field):
        return {}
    events = crossings["crossings"]
    if (len(events) != len(field) or {e["rank"] for e in events} != set(by_rank)
            or len({(e["frame"], e["box_id"]) for e in events}) != len(events)):
        raise ValueError("Finish anchors must cover each crossing and distinct rank exactly once")
    seeds = {}
    for event in events:
        frame, box_id = event["frame"], event["box_id"]
        if not start <= frame < end or frame not in obs or not 0 <= box_id < len(obs[frame]):
            raise ValueError("Finish anchor falls outside this shot or its detections")
        box = obs[frame][box_id]
        seeds.setdefault(frame, []).append((int(by_rank[event["rank"]]), box[:4], box[5:]))
    return seeds


def run(key, experiment):
    started=time.perf_counter()
    field,video,cfg,dest=source(key,experiment)
    obs=evidence(video,dest)
    clock=pd.read_csv(dest/"counts.csv").set_index("frame").pts
    tracker_counts = np.array([len(obs.get(i, ())) for i in clock.index])
    write_json(dest/"tracker_counts.json", dict(short=int((tracker_counts < len(field)).sum()),
               equal=int((tracker_counts == len(field)).sum()), over=int((tracker_counts > len(field)).sum()),
               frames=len(clock), nms_iou=.43))
    gate=cfg["gate_anchor_frame"]
    with reader(video) as cap:
        for i,_,image in frames(cap):
            if i==gate:
                gate_cache = DATA/"inspection"/video.stem/"gate_tiles.csv"
                if gate_cache.exists():
                    verified = pd.read_csv(gate_cache)
                    starts = verified.loc[(verified.frame==gate) & (verified.score>=.5)].copy()
                else:
                    from .tracking import Detector
                    detector = Detector(DATA/"weights"/"yolov4-608.torchscript", "cuda:0", 608)
                    gate_tiles = cfg["gate_tiles"]
                    low_y, high_y = gate_tiles["foot_y_range"]
                    parts = []
                    for x,y,r,b,low,high in gate_tiles["regions"]:
                        part = detector(image[y:b,x:r])
                        part[:,:4] += [x,y,x,y]
                        cx = (part[:,0]+part[:,2])/2
                        parts.extend(part[(cx>=low)&(cx<high)&(part[:,3]>low_y)&(part[:,3]<high_y)&(part[:,4]>=.5)])
                    starts = pd.DataFrame(parts, columns=[*BOX,"score"])
                starts["axis"]=(starts.left+starts.right)/2*cfg["gate_axis"][0]+(starts.top+starts.bottom)/2*cfg["gate_axis"][1]
                starts=starts.sort_values("axis",ascending=not cfg["reverse_gate_order"])
                if len(starts)!=len(field):
                    raise ValueError("Gate verification and starter count differ")
                gate_seeds=[(int(h),box,cap_hist(image,box)) for h,box in zip(field.index,starts[BOX].to_numpy())]
                break
    edges=[gate,*[i for i in cfg["confirmed_cuts"] if i>gate],int(clock.index.max())+1]
    results=[]
    finish_path=dest/"finish_crossings.json"
    official_back=False
    for shot,(start,end) in enumerate(zip(edges,edges[1:])):
        det=obs.get(start,np.empty((0,19)))
        seeds={start:gate_seeds if shot==0 else seed_list(det,len(field))}
        results.extend(run_direction(range(start,end),seeds,obs,clock,shot,"forward"))
        last=end-1
        seeds={last:seed_list(obs.get(last,np.empty((0,19))),len(field))}
        if end==edges[-1] and finish_path.exists():
            _,ranks=mykeibadb_results(str(field.RACE_CODE.iloc[0]),field)
            named = finish_seeds(read_json(finish_path), ranks, field, obs, start, end)
            if named:
                seeds=named
                official_back=True
        results.extend(run_direction(range(last,start-1,-1),seeds,obs,clock,shot,"reverse"))
    colorcols=[f"hat{j}" for j in range(14)]
    df=pd.DataFrame(results,columns=["shot","frame","pts","direction","id","evidence","box_id",*BOX,*colorcols])
    df.to_csv(dest/"tracks.csv",index=False)
    forward=df[df.direction=="forward"]; reverse=df[df.direction=="reverse"]
    conflicts,votes,shared=compare(forward,reverse)
    write_json(dest/"direction_review.json",dict(official_finish_anchored=official_back,
        comparison="shot-local labels aligned by maximum common observations",shared_observations=shared,
        mapping=votes,candidate_intervals=conflicts))
    summary=dict(forward=forward.evidence.value_counts().to_dict(), reverse=reverse.evidence.value_counts().to_dict(),
        candidate_intervals=len(conflicts),comparison_observations=shared,official_finish_anchored=official_back,
        race_shots=len(edges)-1,seconds=time.perf_counter()-started)
    for label,part in [("gate",forward[forward.shot==0]),("reverse_final",reverse[reverse.shot==len(edges)-2])]:
        direct=part[part.evidence.isin(["observed","anchor"])]
        summary[label+"_direct_full_frames"]=int((direct.groupby("frame").size()==len(field)).sum())
        summary[label+"_last_direct_frame"]=int(direct.frame.max()) if len(direct) else None
    write_json(dest/"track_summary.json",summary)
    print(summary)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("key"); p.add_argument("--experiment",default=EXPERIMENT)
    args=p.parse_args();run(args.key,args.experiment)
