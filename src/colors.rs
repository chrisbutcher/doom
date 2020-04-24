pub use super::*;

pub fn load_first_palette(wad_file: &Vec<u8>, lumps: &Vec<Lump>) -> Vec<PaletteColor> {
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

pub fn load_first_colormap(wad_file: &Vec<u8>, lumps: &Vec<Lump>) -> Colormap {
  let colormap_lump = lumps.iter().find(|&l| l.name == "COLORMAP").unwrap();
  // TODO

  let colormap_raw: Vec<usize> = wad_file[colormap_lump.filepos..colormap_lump.filepos + 256]
    .to_vec()
    .iter()
    .map(|b| *b as usize)
    .collect();

  Colormap {
    palette_indexes: colormap_raw,
  }
}
