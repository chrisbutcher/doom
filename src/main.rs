#[macro_use]
extern crate image;
extern crate lyon;
extern crate nalgebra_glm as glm;
extern crate regex;

use std::fs::File;
use std::io::prelude::*;

// pub mod camera;
pub mod colors;
pub mod lumps;
pub mod map_svg;
pub mod maps;
pub mod model;
pub mod png_dump;
pub mod state;
pub mod texture;
pub mod wad_graphics;

// Graphics

use winit::{
  event::*,
  event_loop::{ControlFlow, EventLoop},
  window::{Window, WindowBuilder},
};

// TODO: Read this https://fasterthanli.me/blog/2020/a-half-hour-to-learn-rust/

// TODO: Convert at least some of the many Vec::new to Vec::with_capacity
fn main() {
  let mut f = File::open("doom.wad").unwrap();
  let mut wad_file = Vec::new();
  f.read_to_end(&mut wad_file).unwrap();

  let lumps = lumps::load(&wad_file);

  let current_map = maps::load("^E1M1$", &wad_file, &lumps);
  // map_svg::draw_map_svg(current_map);

  let textures = wad_graphics::load_textures(&wad_file, &lumps);
  let palette = colors::load_first_palette(&wad_file, &lumps);
  let patch_names = wad_graphics::load_patch_names(&wad_file, &lumps);
  let colormap = colors::load_first_colormap(&wad_file, &lumps);

  let scene = Scene {
    map: current_map,
    wad_file: wad_file,
    textures: textures,
    patch_names: patch_names,
    lumps: lumps,
    palette: palette,
    _colormap: colormap,
  };

  render(scene);
}

fn render(scene: Scene) {
  env_logger::init();
  let event_loop = EventLoop::new();
  let title = env!("CARGO_PKG_NAME");
  let window = winit::window::WindowBuilder::new()
    .with_title(title)
    .build(&event_loop)
    .unwrap();

  use futures::executor::block_on;
  let mut state = block_on(state::State::new(&window));
  event_loop.run(move |event, _, control_flow| {
    *control_flow = ControlFlow::Poll;
    match event {
      Event::MainEventsCleared => window.request_redraw(),
      Event::WindowEvent { ref event, window_id } if window_id == window.id() => {
        if !state.input(event) {
          match event {
            WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
            WindowEvent::KeyboardInput { input, .. } => match input {
              KeyboardInput {
                state: ElementState::Pressed,
                virtual_keycode: Some(VirtualKeyCode::Escape),
                ..
              } => {
                *control_flow = ControlFlow::Exit;
              }
              _ => {}
            },
            WindowEvent::Resized(physical_size) => {
              state.resize(*physical_size);
            }
            WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
              state.resize(**new_inner_size);
            }
            _ => {}
          }
        }
      }
      Event::RedrawRequested(_) => {
        state.update();
        match state.render() {
          Ok(_) => {}
          // Recreate the swap_chain if lost
          Err(wgpu::SwapChainError::Lost) => state.resize(state.size),
          // The system is out of memory, we should probably quit
          Err(wgpu::SwapChainError::OutOfMemory) => *control_flow = ControlFlow::Exit,
          // All other errors (Outdated, Timeout) should be resolved by the next frame
          Err(e) => eprintln!("{:?}", e),
        }
      }
      _ => {}
    }
  });
}

// NOTES regarding texturing.
// From unofficial Doom spec: http://www.gamers.org/dhs/helpdocs/dmsp1666.html
//
// Walls:
//
// [4-4]: SIDEDEFS
// If the wall is wider than the texture to be pasted onto it, then the
// texture is tiled horizontally. A 64-wide texture will be pasted at 0,
// 64, 128, etc., unless an X-offset changes this.
//   If the wall is taller than the texture, than the texture is tiled
// vertically, with one very important difference: it starts new tiles
// ONLY at 128, 256, 384, etc. So if the texture is less than 128 high,
// there will be junk filling the undefined areas, and it looks ugly.
// This is sometimes called the "Tutti Frutti" effect.
//
// Ceilings and floors:
// All the lumpnames for flats are in the directory between the F_START
// and F_END entries. Calling them flats is a good way to avoid confusion
// with wall textures. There is no look-up or meta-structure in flats as
// there is in walls textures. Each flat is 4096 raw bytes, making a square
// 64 by 64 pixels. This is pasted onto a floor or ceiling with the same
// orientation as the automap would imply, i.e. the first byte is the color
// at the NW corner, the 64th byte (byte 63, 0x3f) is the NE corner, etc.
//   The blocks in the automap grid are 128 by 128, so four flats will fit
// in each block. Note that there is no way to offset the placement of flats,
// as can be done with wall textures. They are pasted according to grid lines
// 64 apart, reckoned from the coordinate (0,0). This allows flats to flow
// smoothly even across jagged boundaries between sectors with the same
// floor or ceiling height.

#[derive(Debug)]
pub struct Lump {
  filepos: usize,
  size: usize,
  name: String,
}

#[derive(Debug, Clone)]
pub struct MapVertex {
  x: i16,
  y: i16,
}

#[derive(Debug)]
pub struct Map {
  name: String,
  vertexes: Vec<MapVertex>,
  linedefs: Vec<LineDef>,
  sidedefs: Vec<SideDef>,
  sectors: Vec<Sector>,
  map_centerer: map_svg::MapCenterer,
}

#[derive(Debug, Clone)]
pub struct LineDef {
  // TODO: Flags, special type, sector tag: https://doomwiki.org/wiki/Linedef
  start_vertex: usize,
  end_vertex: usize,
  front_sidedef_index: Option<usize>,
  back_sidedef_index: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct SideDef {
  x_offset: i16,
  y_offset: i16,
  name_of_upper_texture: Option<String>,
  name_of_lower_texture: Option<String>,
  name_of_middle_texture: Option<String>, // NOTE: Most normal walls only have this on their front side
  sector_facing: usize, // ... but some walls like the grates at the end of e1m1 have only a middle on both sides
}

#[derive(Debug, Clone)]
pub struct Sector {
  floor_height: i16,
  ceiling_height: i16,
  name_of_floor_texture: String,   // Make these Option<String> ?
  name_of_ceiling_texture: String, // Make these Option<String> ?
  light_level: i16,
  sector_type: i16,
  tag_number: i16,
}

#[derive(Debug, Clone)]
pub struct Colormap {
  palette_indexes: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct PaletteColor {
  r: u8,
  g: u8,
  b: u8,
}

#[derive(Debug, Clone)]
pub struct Flat {
  pixels: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct Picture {
  width: u16,
  height: u16,
  leftoffset: u16,
  topoffset: u16,
  posts: Vec<PicturePost>,
  lump_name: String,
}

#[derive(Debug, Clone)]
pub struct PictureSpan {
  topdelta: usize,
  length: u8,
  pixels: Vec<usize>,
  last_span_pixel_count: usize,
  last_span_topdelta: usize,
}

#[derive(Debug, Clone)]
pub struct PicturePost {
  pixel_spans: Vec<PictureSpan>,
}

#[derive(Debug, Clone)]
pub struct Patch {
  name: String,
}

#[derive(Debug, Clone)]
pub struct WallPatch {
  originx: i16,
  originy: i16,
  patch_number: usize,
}

#[derive(Debug, Clone)]
pub struct WallTexture {
  name: String,
  masked: bool,
  width: i16,
  height: i16,
  patches: Vec<WallPatch>,
}

pub struct Scene {
  map: Map,
  wad_file: Vec<u8>,
  textures: Vec<WallTexture>,
  patch_names: Vec<Patch>,
  lumps: Vec<Lump>,
  palette: Vec<PaletteColor>,
  _colormap: Colormap,
}
