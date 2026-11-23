"""Local YOLO11 pack crops, editable YOLO drafts and full-race detection measurements."""
import argparse
import os
from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
import torch

from .footage import reader, frames
from .io_local import DATA, race_input, read_json, write_json


def offline():
    directory = DATA / "ultralytics"
    directory.mkdir(parents=True, exist_ok=True)
    os.environ.update(YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false", YOLO_CONFIG_DIR=str(directory),
                      YOLO_VERBOSE="false", WANDB_MODE="disabled", MPLBACKEND="Agg")


class PackModel:
    def __init__(self, path, device="cuda:0"):
        offline()
        from ultralytics.nn.tasks import DetectionModel  # Register local checkpoint classes.
        state = torch.load(path, weights_only=False, map_location="cpu")
        self.net = (state.get("ema") or state["model"]).to(device).half().eval()
        self.device = device
        self.horse = next(i for i, name in self.net.names.items() if name == "horse")

    @torch.inference_mode()
    def detect(self, image, size=1280, confidence=.1, overlap=.6):
        h, w = image.shape[:2]
        scale = size / max(h, w)
        sw, sh = max(1, round(w*scale)), max(1, round(h*scale))
        canvas = np.full(((sh+31)//32*32, (sw+31)//32*32, 3), 114, np.uint8)
        canvas[:sh, :sw] = cv2.resize(image, (sw, sh))
        rgb = np.ascontiguousarray(canvas[..., ::-1].transpose(2, 0, 1))
        pred = self.net(torch.from_numpy(rgb).to(self.device).half()[None] / 255)[0][0].T.float()
        pred = pred[pred[:, 4+self.horse] >= confidence]
        boxes = pred[:, :4].cpu().numpy()
        score = pred[:, 4+self.horse].cpu().numpy()
        boxes[:, :2] -= boxes[:, 2:]/2
        keep = np.asarray(cv2.dnn.NMSBoxes(boxes.tolist(), score.tolist(), confidence-1e-6, overlap), int).ravel()
        boxes = boxes[keep]
        boxes[:, 2:] += boxes[:, :2]
        return np.c_[(boxes/scale).clip(0, [w,h,w,h]), score[keep]]


def crop_table(cfg, detections, width, height):
    first, last = cfg["gate_open_frame"], cfg["end_frame"]
    ids = pd.Index(range(first, last+1), name="frame")
    valid = detections[(detections.score >= .22) & (detections.top > 70)]
    table = valid.groupby("frame").agg(x=("left","min"), y=("top","min"), r=("right","max"), b=("bottom","max")).reindex(ids)
    edges = [first, *(i for i in cfg["confirmed_cuts"] if first < i <= last), last+1]
    for a,b in zip(edges, edges[1:]):
        part = table.loc[a:b-1].interpolate().bfill().ffill().rolling(9, center=True, min_periods=1).median()
        part = part.fillna({"x":0,"y":60,"r":width,"b":height})
        dx = (part.r-part.x)*.10+50
        dy = (part.b-part.y)*.35+40
        table.loc[a:b-1] = np.column_stack([(part.x-dx).clip(0,width-1), (part.y-dy).clip(60,height-1),
                                             (part.r+dx).clip(1,width), (part.b+dy).clip(61,height)])
    return table.round().astype(int)


def pack_inputs(key):
    code, field, video = race_input(key)
    folder = DATA/"races"/code
    cfg = read_json(folder/"setup.json")
    inv = read_json(folder/"inventory.json")
    out = folder/"pack"
    out.mkdir(exist_ok=True)
    cache = DATA/"inspection"/video.stem/"detections.csv"
    if cache.exists():
        coarse = pd.read_csv(cache)
    else:
        model = PackModel(DATA/"weights"/"yolo11s.pt")
        records = []
        with reader(video) as cap:
            for i, pts, im in frames(cap):
                if cfg["gate_open_frame"] <= i <= cfg["end_frame"]:
                    records.extend((i, pts, *row) for row in model.detect(im, 960, .15))
        coarse = pd.DataFrame(records, columns=["frame","pts","left","top","right","bottom","score"])
        cache.parent.mkdir(exist_ok=True, parents=True)
        coarse.to_csv(cache, index=False)
    crops = crop_table(cfg, coarse, inv["width"], inv["height"])
    # Reviewed pack envelopes keep stragglers in the crop after a coarse detector miss.
    envelope = out/"envelopes.json"
    if envelope.exists():
        for shot in read_json(envelope):
            pts = np.asarray(shot, float)
            ids = crops.loc[int(pts[0,0]):int(pts[-1,0])].index
            bounds = np.stack([np.interp(ids, pts[:,0], column) for column in pts[:,1:].T], axis=1)
            bounds *= [inv["width"], inv["height"], inv["width"], inv["height"]]
            current = crops.loc[ids].to_numpy().copy()
            current[:,:2] = np.minimum(current[:,:2], bounds[:,:2])
            current[:,2:] = np.maximum(current[:,2:], bounds[:,2:])
            crops.loc[ids] = current
    return out, field, video, crops


def sample_split(frame, first):
    return "val" if (frame-first)//96 in {3, 10, 15, 19, 24, 29} else "train"


def sample(key, size):
    folder, _, video, crops = pack_inputs(key)
    folder = folder/"drafts"
    model = PackModel(DATA/"weights"/"yolo11s.pt")
    # Keep contiguous time blocks out of training; examples cover every race shot.
    selected = list(range(crops.index[0]+20, crops.index[-1], 31))
    sample_rows = []
    with reader(video) as cap:
        for i, t, image in frames(cap):
            if i not in selected:
                continue
            x,y,r,b = crops.loc[i]
            crop = image[y:b,x:r]
            split = sample_split(i, crops.index[0])
            for kind in ("images","labels","preview"):
                (folder/kind/split).mkdir(exist_ok=True, parents=True)
            boxes = model.detect(crop,size,confidence=.22)
            h,w = crop.shape[:2]
            stem = str(i)
            cv2.imwrite(str(folder/"images"/split/(stem+".jpg")),crop)
            label = folder/"labels"/split/(stem+".txt")
            if not label.exists():
                with label.open("w") as f:
                    for l,u,r,b,score in boxes:
                        print(0, (l+r)/(2*w), (u+b)/(2*h), (r-l)/w, (b-u)/h, file=f)
            preview = crop.copy()
            for j,(l,u,r,b,_) in enumerate(boxes):
                cv2.rectangle(preview,(int(l),int(u)),(int(r),int(b)),(50,220,40),2)
                cv2.putText(preview,str(j),(int(l),int(b)),0,.55,(0,255,255),2)
            cv2.imwrite(str(folder/"preview"/split/(stem+".jpg")),preview)
            sample_rows.append(dict(frame=i,pts=t,split=split,draft_boxes=len(boxes)))
    pd.DataFrame(sample_rows).to_csv(folder/"samples.csv",index=False)
    (folder/"dataset.yaml").write_text("path: "+folder.as_posix()+"\ntrain: images/train\nval: images/val\nnames: [horse]\n",encoding="utf8")
    print(pd.DataFrame(sample_rows).groupby("split").draft_boxes.describe())


def infer(key, weights, size, confidence):
    folder, field, video, crops = pack_inputs(key)
    model = PackModel(weights)
    dest = folder/(weights.stem+"-"+str(size)+"-"+str(confidence))
    dest.mkdir(exist_ok=True)
    rows, counts = [], []
    started = time.perf_counter()
    with reader(video) as cap:
        for i,t,image in frames(cap):
            if i not in crops.index:
                continue
            x,y,r,b = crops.loc[i]
            boxes = model.detect(image[y:b,x:r],size,confidence)
            boxes[:, :4] += [x,y,x,y]
            rows.extend((i,t,*box) for box in boxes)
            counts.append((i,t,len(boxes)))
    df = pd.DataFrame(counts,columns=["frame","pts","count"])
    df.to_csv(dest/"counts.csv",index=False)
    pd.DataFrame(rows,columns=["frame","pts","left","top","right","bottom","score"]).to_csv(dest/"detections.csv",index=False)
    result=dict(frames=len(df),short=int((df["count"]<len(field)).sum()),over=int((df["count"]>len(field)).sum()),
                seconds=time.perf_counter()-started, size=size, confidence=confidence, weights=weights.name)
    write_json(dest/"summary.json",result)
    print(result)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["sample","infer"])
    p.add_argument("key")
    p.add_argument("--weights",type=Path,default=DATA/"weights"/"yolo11s.pt")
    p.add_argument("--size",type=int,default=1280)
    p.add_argument("--confidence",type=float,default=.1)
    args=p.parse_args()
    if args.action=="sample":
        sample(args.key,args.size)
    else:
        infer(args.key,args.weights,args.size,args.confidence)
