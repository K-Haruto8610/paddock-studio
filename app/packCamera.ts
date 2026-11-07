import * as T from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import type {TrackPoint} from './trackGeometry';

/**
 * The camera always frames the pack. Dragging takes manual control until the
 * "back to the pack" button bumps followKey.
 * Wheel zoom keeps following (it scales the pack distance); only a drag hands over control.
 */
export function createPackCamera(
  camera: T.PerspectiveCamera,
  controls: OrbitControls,
  canvas: HTMLCanvasElement,
) {
  let following = true;
  let lastFollowKey = -1;
  let first = true;
  let zoom = 1;
  let dragging = false;
  let lastProgress = Number.NaN;

  const onControlStart = () => {
    if (dragging) following = false;
  };
  const onDown = () => {
    dragging = true;
  };
  const onUp = () => {
    dragging = false;
  };
  const onWheel = (e: WheelEvent) => {
    if (!following) return;
    e.preventDefault();
    zoom = T.MathUtils.clamp(zoom * Math.exp(e.deltaY * 0.001), 0.35, 3);
  };
  controls.addEventListener('start', onControlStart);
  canvas.addEventListener('pointerdown', onDown);
  window.addEventListener('pointerup', onUp);
  canvas.addEventListener('wheel', onWheel, {passive: false});

  return {
    /** Moves the camera towards the pack for this frame (or leaves it alone when dragged away). */
    update(opts: {
      dt: number;
      progress: number;
      followKey: number;
      packCenter: T.Vector3;
      fieldSpan: number;
      heading: TrackPoint;
    }) {
      const {dt, progress, followKey, packCenter, fieldSpan, heading} = opts;
      if (followKey !== lastFollowKey) {
        following = true;
        lastFollowKey = followKey;
      }
      if (Number.isNaN(lastProgress)) lastProgress = progress;
      controls.enableZoom = !following;
      const closeDistance = Math.max(30, fieldSpan / (Math.max(camera.aspect, 0.7) * 0.8)) * zoom;
      const packOffset = heading.outward
        .clone()
        .multiplyScalar(closeDistance)
        .addScaledVector(heading.tangent, closeDistance * 0.12);
      packOffset.y = closeDistance * 0.62;
      if (following) {
        const alpha = first || Math.abs(progress - lastProgress) > 0.03 ? 1 : 1 - Math.exp(-dt * 6);
        first = false;
        controls.target.lerp(packCenter, alpha);
        camera.position.lerp(packCenter.clone().add(packOffset), alpha);
      }
      lastProgress = progress;
    },
    dispose() {
      controls.removeEventListener('start', onControlStart);
      canvas.removeEventListener('pointerdown', onDown);
      window.removeEventListener('pointerup', onUp);
      canvas.removeEventListener('wheel', onWheel);
    },
  };
}
