use std::fs::File;
use std::io::prelude::*;

use crate::game::*;

pub struct Scene {
    pub map: maps::Map,
    pub wad_file: Vec<u8>,
    pub textures: Vec<wad_graphics::WallTexture>, // Hide these
    pub patch_names: Vec<wad_graphics::Patch>,
    pub lumps: Vec<lumps::Lump>,
    pub palette: Vec<wad_graphics::PaletteColor>,
    pub colormap: wad_graphics::Colormap,
}

impl Scene {
    pub fn load(wad_path: &str, map: &str) -> Self {
        let mut f = File::open(wad_path).unwrap();
        let mut wad_file = Vec::new();
        f.read_to_end(&mut wad_file).unwrap();

        let lumps = lumps::load(&wad_file);

        let current_map = maps::load(map, &wad_file, &lumps);
        let textures = wad_graphics::load_textures(&wad_file, &lumps);
        let palette = colors::load_first_palette(&wad_file, &lumps);
        let patch_names = wad_graphics::load_patch_names(&wad_file, &lumps);
        let colormap = colors::load_first_colormap(&wad_file, &lumps);

        Scene {
            map: current_map,
            wad_file,
            textures,
            patch_names,
            lumps,
            palette,
            colormap,
        }
    }
}
