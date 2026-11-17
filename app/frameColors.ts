const frameColors = ['#fff', '#303534', '#d96859', '#4f7bab', '#e3bf58', '#55977e', '#d79653', '#ca88a6'];

export const framePalette = (gate: number) => ({
  background: frameColors[gate - 1],
  color: gate === 1 ? '#333' : '#fff',
});
