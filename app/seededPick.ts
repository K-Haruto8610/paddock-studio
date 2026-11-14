/** FNV-1a hash of a string, as an unsigned 32-bit integer. */
function hash(text: string) {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) h = Math.imul(h ^ text.charCodeAt(i), 0x01000193);
  // Final mix so that similar keys (consecutive race ids) land far apart.
  h ^= h >>> 16;
  h = Math.imul(h, 0x85ebca6b);
  h ^= h >>> 13;
  return h >>> 0;
}

/** A random-looking choice that is always the same for the same key. */
export function seededPick<V>(key: string, choices: readonly V[]) {
  return choices[hash(key) % choices.length];
}
