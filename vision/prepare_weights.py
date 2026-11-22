"""Convert a local COCO YOLOv4 Darknet pair to the detector's TorchScript tensor format."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .io_local import DATA


def blocks(cfg):
    layers = []
    for line in cfg.read_text().splitlines():
        line = line.split("#")[0].strip()
        if line.startswith("["):
            layers.append(dict(kind=line.strip("[]")))
        elif "=" in line:
            k, v = line.split("=", 1)
            layers[-1][k.strip()] = v.strip()
    return layers[1:]


class Head(nn.Module):
    def __init__(self, block, size, stride):
        super().__init__()
        if int(block["classes"]) != 80:
            raise ValueError("Expected COCO head with 80 classes")
        anchors = np.fromstring(block["anchors"], sep=",", dtype=np.float32).reshape(-1, 2)
        mask = np.fromstring(block["mask"], sep=",", dtype=int)
        self.register_buffer("anchors", torch.from_numpy(anchors[mask]).reshape(1, -1, 1, 1, 2))
        grid = torch.arange(size//stride)
        y, x = torch.meshgrid(grid, grid, indexing="ij")
        self.register_buffer("grid", torch.stack((x, y), dim=-1))
        self.stride, self.scale = stride, float(block["scale_x_y"])

    def forward(self, x):
        b, _, h, w = x.shape
        a = x.reshape(b, 3, 85, h, w).permute(0, 1, 3, 4, 2)
        xy = (a[..., :2].sigmoid()*self.scale - (self.scale-1)/2 + self.grid)*self.stride
        wh = a[..., 2:4].exp()*self.anchors
        score = a[..., 4:5].sigmoid()*a[..., 5:].sigmoid()
        return torch.cat((xy, wh, score), dim=-1).reshape(b, -1, 84)


class Darknet(nn.Module):
    def __init__(self, cfg, weights, size):
        super().__init__()
        self.layers = nn.ModuleList()
        self.links = []
        channels = []
        stride = 8
        with weights.open("rb") as f:
            header = np.fromfile(f, "<i4", 5)
            if len(header) != 5 or header[:2].tolist() != [0, 2]:
                raise ValueError("Expected Darknet v0.2 weights")

            def load(tensor):
                data = np.fromfile(f, "<f4", tensor.numel())
                tensor.copy_(torch.from_numpy(data).reshape_as(tensor))

            for d in blocks(cfg):
                kind = d["kind"]
                prev = channels[-1] if channels else 3
                out = prev
                links = []
                if kind == "convolutional":
                    out, k = int(d["filters"]), int(d["size"])
                    bn = d.get("batch_normalize") == "1"
                    conv = nn.Conv2d(prev, out, k, int(d["stride"]), k//2, bias=not bn)
                    layers = [conv]
                    if bn:
                        norm = nn.BatchNorm2d(out, eps=1e-5)
                        for tensor in (norm.bias, norm.weight, norm.running_mean, norm.running_var):
                            load(tensor)
                        layers.append(norm)
                    else:
                        load(conv.bias)
                    load(conv.weight)
                    activation = {"mish": nn.Mish, "leaky": lambda: nn.LeakyReLU(.1), "linear": nn.Identity}
                    layer = nn.Sequential(*layers, activation[d["activation"]]())
                elif kind == "route":
                    links = [int(i) for i in d["layers"].split(",")]
                    out = sum(channels[i] for i in links)
                    layer = nn.Identity()
                elif kind == "shortcut":
                    links = [int(d["from"])]
                    layer = nn.Identity()
                elif kind == "maxpool":
                    k = int(d["size"])
                    layer = nn.MaxPool2d(k, int(d["stride"]), k//2)
                elif kind == "upsample":
                    layer = nn.Upsample(scale_factor=int(d["stride"]), mode="nearest")
                elif kind == "yolo":
                    layer = Head(d, size, stride)
                    stride *= 2
                else:
                    raise ValueError(f"Unsupported Darknet layer {kind}")
                self.layers.append(layer)
                self.links.append((kind, links))
                channels.append(out)
            if f.read(1) or stride != 64:
                raise ValueError("Configuration and weights do not match the three-head COCO model")

    def forward(self, x):
        outputs, heads = [], []
        for layer, (kind, refs) in zip(self.layers, self.links):
            if kind == "route":
                x = torch.cat([outputs[j] for j in refs], dim=1)
            elif kind == "shortcut":
                x = x + outputs[refs[0]]
            elif kind == "yolo":
                heads.append(layer(x))
            else:
                x = layer(x)
            outputs.append(x)
        return torch.cat(heads, dim=1).transpose(1, 2)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cfg", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("--size", type=int, default=608)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = Darknet(args.cfg, args.weights, args.size).eval()
    trace = torch.jit.trace(model, torch.zeros(1, 3, args.size, args.size), check_trace=False)
    dest = DATA/"weights"/f"yolov4-{args.size}.torchscript"
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(torch.jit.freeze(trace), str(dest))
    print(dest)


if __name__ == "__main__":
    main()
