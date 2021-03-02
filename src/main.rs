use std::fs::File;
use std::io::prelude::*;
use std::iter;

use cgmath::prelude::*;
use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

// -- Doom specific
pub mod colors;
pub mod lumps;
pub mod map_svg;
pub mod maps;
pub mod png_dump;
pub mod wad_graphics;
//

mod camera;
mod model;
mod renderer;
mod texture;

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

pub struct Scene {
    map: maps::Map,
    wad_file: Vec<u8>,
    textures: Vec<wad_graphics::WallTexture>,
    patch_names: Vec<wad_graphics::Patch>,
    lumps: Vec<lumps::Lump>,
    palette: Vec<wad_graphics::PaletteColor>,
    _colormap: wad_graphics::Colormap,
}

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

    //

    env_logger::init();
    let event_loop = EventLoop::new();
    let title = env!("CARGO_PKG_NAME");
    let window = winit::window::WindowBuilder::new()
        .with_title(title)
        .build(&event_loop)
        .unwrap();
    use futures::executor::block_on;
    let mut state = block_on(renderer::State::new(&window, scene)); // NEW!
    let mut last_render_time = std::time::Instant::now();
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::MainEventsCleared => window.request_redraw(),
            Event::DeviceEvent {
                ref event,
                .. // We're not using device_id currently
            } => {
                state.input(event);
            }
            // UPDATED!
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => {
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
            // UPDATED!
            Event::RedrawRequested(_) => {
                let now = std::time::Instant::now();
                let dt = now - last_render_time;
                last_render_time = now;
                state.update(dt);
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
