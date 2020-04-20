pub use super::Lump;

#[derive(Debug, Clone)]
pub struct PaletteColor {
  r: u8,
  g: u8,
  b: u8,
}

pub fn load_palette(wad_file: &Vec<u8>, lumps: &Vec<Lump>) -> Vec<PaletteColor> {
  let palette_lump = lumps.iter().find(|&l| l.name == "PLAYPAL").unwrap();

  let palette_raw = wad_file[palette_lump.filepos..palette_lump.filepos + 768].to_vec();

  let palette = palette_raw
    .chunks(3)
    .map(|c| PaletteColor {
      r: c[0],
      g: c[1],
      b: c[2],
    })
    .collect();

  palette
}
