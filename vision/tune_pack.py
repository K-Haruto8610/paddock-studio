"""Fine-tune the local COCO YOLO11 horse class on the operator's editable patrol drafts."""
import argparse
import copy
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from .io_local import DATA, write_json
from .pack import offline, pack_inputs, sample_split


def fit(folder, epochs, weights, size, name="patrol11s", reviewed_only=False):
    offline()
    from ultralytics.cfg import get_cfg
    from .io_local import read_json

    torch.manual_seed(17)
    rng = random.Random(17)
    ckpt = torch.load(weights, weights_only=False, map_location="cpu")
    model = (ckpt.get("ema") or ckpt["model"]).float().cuda().train()
    model.args = get_cfg()
    for layer in model.model:
        for parameter_name, param in layer.named_parameters():
            param.requires_grad_(layer.i >= 10 and "dfl." not in parameter_name)
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=.00012)
    scaler = torch.amp.GradScaler("cuda")
    paths = sorted((folder/"images/train").glob("*.jpg"))
    train = []
    manual=set(map(str,read_json(folder/"manual_drafts.json")["frames"]))
    first=read_json(folder.parent.parent/"setup.json")["gate_open_frame"]
    for path in paths:
        if sample_split(int(path.stem.split("-")[-1]), first) == "val":
            continue
        if reviewed_only and path.stem not in manual and not path.stem.startswith("flow-"):
            continue
        label = folder/"labels/train"/(path.stem+".txt")
        if label.read_text().strip():
            train.append((path,np.loadtxt(label,ndmin=2,dtype=np.float32)[:,1:]))
    if not train:
        raise ValueError("No local training annotations")
    history = []
    for epoch in range(epochs):
        rng.shuffle(train)
        losses = []
        opt.zero_grad()
        for i,(path,targets) in enumerate(train):
            im = cv2.imread(str(path))
            boxes = targets.copy()
            if rng.random()<.5:
                im = im[:,::-1]; boxes[:,0] = 1-boxes[:,0]
            h,w = im.shape[:2]
            ratio = size/max(h,w)
            sw,sh = max(1,round(w*ratio)),max(1,round(h*ratio))
            arr=np.full(((sh+31)//32*32,(sw+31)//32*32,3),114,np.uint8)
            arr[:sh,:sw] = cv2.resize(im,(sw,sh))
            boxes *= [sw/arr.shape[1],sh/arr.shape[0],sw/arr.shape[1],sh/arr.shape[0]]
            img=torch.from_numpy(np.ascontiguousarray(arr[:,:,::-1].transpose(2,0,1))).cuda().float()[None]/255
            img=(img*rng.uniform(.82,1.15)).clamp_(0,1)
            batch=dict(img=img,bboxes=torch.from_numpy(boxes).cuda(),cls=torch.full((len(boxes),1),17.,device="cuda"),
                       batch_idx=torch.zeros(len(boxes),device="cuda"))
            with torch.autocast("cuda"):
                loss,_=model(batch)
                loss=loss.sum()
            scaler.scale(loss/4).backward()
            if (i+1)%4==0 or i+1==len(train):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
                scaler.step(opt);scaler.update();opt.zero_grad()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
        print(f"epoch {epoch+1}/{epochs}, loss {history[-1]:.3f}",flush=True)
    dest=folder.parent/(name+".pt")
    model.criterion = None
    torch.save({"model":copy.deepcopy(model).cpu().half().eval()},dest)
    write_json(folder.parent/(name+"-training.json"),dict(train_frames=len(train),epochs=epochs,loss=history,size=size,
                                                initial_weights=weights.name,annotation_status="owner_spot_check_pending"))
    return dest


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("key")
    p.add_argument("--epochs",type=int,default=8)
    p.add_argument("--size",type=int,default=960)
    p.add_argument("--weights",type=Path,default=DATA/"weights"/"yolo11s.pt")
    p.add_argument("--name",default="patrol11s")
    p.add_argument("--reviewed-only",action="store_true")
    args=p.parse_args()
    root,*_=pack_inputs(args.key)
    print(fit(root/"drafts",args.epochs,args.weights,args.size,args.name,args.reviewed_only))
