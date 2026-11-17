import * as T from 'three';
import type {Race} from './types';
import {framePalette} from './frameColors';

const HEIGHT = 256;
const NUMBER_SIZE = 256;
const NAME_WIDTH = 1024;

function plate(text: string, width: number, gate: number) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = HEIGHT;
  const ctx = canvas.getContext('2d')!;
  const {background, color} = framePalette(gate);
  ctx.beginPath();
  ctx.roundRect(6, 6, width - 12, HEIGHT - 12, 42);
  ctx.fillStyle = background;
  ctx.fill();
  ctx.strokeStyle = 'rgba(30, 37, 34, .28)';
  ctx.lineWidth = 8;
  ctx.stroke();
  ctx.fillStyle = color;
  ctx.font = width === NUMBER_SIZE ? 'bold 158px sans-serif' : 'bold 120px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, width / 2, HEIGHT * 0.53, width - 58);
  const map = new T.CanvasTexture(canvas);
  map.colorSpace = T.SRGBColorSpace;
  return new T.Sprite(new T.SpriteMaterial({
    map, depthTest: false, depthWrite: false, toneMapped: false, fog: false,
  }));
}

type Rect = {x: number; y: number; w: number; h: number};
const overlaps = (a: Rect, b: Rect) =>
  Math.abs(a.x - b.x) < (a.w + b.w) / 2 + 3 && Math.abs(a.y - b.y) < (a.h + b.h) / 2 + 3;

export function addHorseLabels(scene: T.Scene, race: Race, horses: T.Group[]) {
  const labels = race.horses.map((horse) => {
    const number = plate(String(horse.number), NUMBER_SIZE, horse.gate);
    const name = plate(`${horse.number}  ${horse.name}`, NAME_WIDTH, horse.gate);
    const points = new T.BufferAttribute(new Float32Array(6), 3);
    const line = new T.Line(new T.BufferGeometry().setAttribute('position', points),
      new T.LineBasicMaterial({color: '#faf3dc', transparent: true, opacity: 0.88, toneMapped: false,
        fog: false, depthTest: false, depthWrite: false}));
    line.frustumCulled = false;
    line.renderOrder = 1_000;
    number.renderOrder = name.renderOrder = 1_001;
    scene.add(number, name, line);
    return {horse, number, name, line, points};
  });
  const projected = new T.Vector3();
  const world = new T.Vector3();
  return {
    update(camera: T.PerspectiveCamera, width: number, height: number, travel: number,
      gaps: number[], selected: number | null) {
      for (const {number, name, line} of labels) number.visible = name.visible = line.visible = false;
      if (travel < 1) return;
      camera.updateMatrixWorld();
      const focal = height / (2 * Math.tan(T.MathUtils.degToRad(camera.fov / 2)));
      const screen = (point: T.Vector3) => {
        projected.copy(point).project(camera);
        return {x: (projected.x + 1) * width / 2, y: (1 - projected.y) * height / 2, z: projected.z};
      };
      const field = labels.map((label, i) => {
        const horse = horses[i];
        horse.updateWorldMatrix(false, false);
        const back = horse.localToWorld(new T.Vector3(0, 2.7, 0));
        const head = horse.localToWorld(new T.Vector3(1.35, 3.25, 0));
        const onScreen = screen(back);
        const distance = -world.copy(back).applyMatrix4(camera.matrixWorldInverse).z;
        return {label, i, back, onScreen, distance, head: {...screen(head),
          w: Math.min(80, 1.7 * focal / distance), h: Math.min(90, 1.9 * focal / distance)}};
      }).filter(({onScreen, distance}) => distance > 0 && onScreen.z < 1 &&
        onScreen.x >= 0 && onScreen.x <= width && onScreen.y >= 0 && onScreen.y <= height);
      const leader = gaps.indexOf(Math.min(...gaps));
      const priority = ({i, label}: (typeof field)[number]) => label.horse.number === selected ? 0 : i === leader ? 1 : 2;
      field.sort((a, b) => priority(a) - priority(b) || a.distance - b.distance || a.i - b.i);
      const occupied: Rect[] = [];
      for (const item of field) {
        const {label, onScreen, distance, back} = item;
        const named = label.horse.number === selected;
        const sprite = named ? label.name : label.number;
        const h = named ? 24 : Math.round(T.MathUtils.clamp(28 - 120 / distance, 18, 26));
        const w = named ? h * NAME_WIDTH / HEIGHT : h;
        if (w + 8 > width || h + 8 > height) continue;
        const x = T.MathUtils.clamp(onScreen.x, w / 2 + 4, width - w / 2 - 4);
        const originalY = onScreen.y - h / 2 - Math.max(8, Math.min(34, focal / distance));
        for (let shift = 0; shift <= 4; shift++) {
          const y = originalY - shift * (h + 4);
          if (y < h / 2 + 4 || y > height - h / 2 - 4) continue;
          const rect = {x, y, h, w};
          if (occupied.some(box => overlaps(rect, box)) || field.some(other => other !== item && overlaps(rect, other.head))) continue;
          occupied.push(rect);
          const unproject = (px: number, py: number) =>
            new T.Vector3(2 * px / width - 1, 1 - 2 * py / height, onScreen.z).unproject(camera);
          sprite.position.copy(unproject(x, y));
          sprite.scale.set(w * distance / focal, h * distance / focal, 1);
          sprite.visible = true;
          if (shift || Math.abs(x - onScreen.x) > 3) {
            const bottom = unproject(x, y + h / 2);
            label.points.setXYZ(0, back.x, back.y, back.z);
            label.points.setXYZ(1, bottom.x, bottom.y, bottom.z);
            label.points.needsUpdate = true;
            label.line.visible = true;
          }
          break;
        }
      }
    },
    dispose() {
      for (const {line} of labels) {
        line.geometry.dispose();
        line.material.dispose();
        line.removeFromParent();
      }
    },
  };
}
