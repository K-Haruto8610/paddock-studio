import * as T from 'three';
import type {Horse} from './types';
const coat: Record<string, string> = {
  '01': '#b8773c',
  '02': '#845330',
  '03': '#926039',
  '04': '#45332c',
  '05': '#292b2c',
  '06': '#202126',
  '07': '#c7c9c4',
  '08': '#b69a7f',
  '09': '#958475',
  '10': '#767b81',
  '11': '#f1e8d9',
};
export function label(text: string, width = 512) {
  const c = document.createElement('canvas');
  c.width = width;
  c.height = 128;
  const x = c.getContext('2d')!;
  x.fillStyle = '#fffaf0';
  x.beginPath();
  x.roundRect(4, 12, width - 8, 100, 22);
  x.fill();
  x.fillStyle = '#263b34';
  x.font = width === 128 ? 'bold 82px sans-serif' : 'bold 36px sans-serif';
  x.textAlign = 'center';
  x.textBaseline = 'middle';
  x.fillText(text, width / 2, 64, width - 25);
  const map = new T.CanvasTexture(c);
  map.colorSpace = T.SRGBColorSpace;
  const s = new T.Sprite(new T.SpriteMaterial({map, depthTest: false}));
  s.scale.set((width / 128) * 1.8, 1.8, 1);
  s.renderOrder = 5;
  return s;
}
export function plush(h: Horse) {
  const g = new T.Group();
  const brown = new T.MeshStandardMaterial({color: coat[h.coatCode] || '#999999', roughness: 1});
  const dark = new T.MeshStandardMaterial({color: '#302a29', roughness: 1});
  const white = new T.MeshStandardMaterial({color: '#eee9db', roughness: 1});
  const purple = new T.MeshStandardMaterial({color: '#7250a1', roughness: 1});
  const ball = (mat: T.Material, p: number[], s: number[]) => {
    const m = new T.Mesh(new T.SphereGeometry(1, 16, 12), mat);
    m.position.set(...(p as [number, number, number]));
    m.scale.set(...(s as [number, number, number]));
    m.castShadow = true;
    g.add(m);
    return m;
  };
  ball(brown, [0, 1.5, 0], [1.25, 0.95, 0.75]);
  ball(brown, [0.85, 2.2, 0], [0.6, 0.95, 0.62]);
  ball(brown, [1.3, 2.75, 0], [0.8, 0.78, 0.66]);
  ball(brown, [1.85, 2.5, 0], [0.52, 0.42, 0.58]);
  for (const z of [-0.47, 0.47]) {
    for (const x of [-0.78, 0.73]) {
      ball(brown, [x, 0.65, z], [0.35, 0.65, 0.35]);
      ball(white, [x, 0.3, z], [0.36, 0.23, 0.36]);
      ball(dark, [x, 0.15, z], [0.37, 0.19, 0.38]);
    }
    ball(brown, [1.15, 3.55, z * 0.8], [0.23, 0.5, 0.22]);
    ball(dark, [1.65, 2.9, z * 1.34], [0.12, 0.14, 0.08]);
  }
  for (let i = 0; i < 7; i++) ball(dark, [0.52 + i * 0.12, 2.35 + i * 0.16, 0], [0.27, 0.26, 0.36]);
  for (let i = 0; i < 4; i++) ball(dark, [-1.3 - i * 0.12, 1.7 - i * 0.22, 0], [0.35, 0.35, 0.27]);
  ball(purple, [-0.1, 2.12, 0], [0.8, 0.3, 0.81]);
  if (h.blinker) {
    const blue = new T.MeshStandardMaterial({color: '#436b9f', roughness: 1});
    ball(blue, [1.28, 2.78, 0], [0.83, 0.79, 0.68]);
    for (const z of [-0.69, 0.69]) {
      ball(dark, [1.6, 2.91, z], [0.16, 0.18, 0.04]);
    }
    ball(brown, [1.96, 2.5, 0], [0.46, 0.36, 0.53]);
  }
  const plate = label(`${h.number}  ${h.name}`);
  plate.position.set(0, 8, 0);
  plate.scale.set(18, 4.5, 1);
  plate.name = 'nameplate';
  g.add(plate);
  const num = label(String(h.number), 128);
  num.position.set(-0.2, 1.65, 0.87);
  num.scale.set(0.95, 0.95, 1);
  num.name = 'numberplate';
  g.add(num);
  return g;
}
