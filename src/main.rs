#[macro_use]
extern crate glium;
extern crate image;
extern crate lyon;
extern crate nalgebra_glm as glm;
extern crate regex;

use std::collections::HashMap;
use std::fs::File;
use std::io::prelude::*;
use std::io::Cursor;

// use lyon::geometry_builder::simple_builder;
use lyon::math::{point, Point};
// use lyon::path::builder::*;
use lyon::path::Path;
use lyon::tessellation::*;
// use lyon_tessellation::geometry_builder::simple_builder;
use lyon::tessellation::geometry_builder::simple_builder;

pub mod camera;
pub mod colors;
pub mod lumps;
pub mod map_svg;
pub mod maps;
pub mod png_dump;
pub mod wad_graphics;

// TODO: Read this https://fasterthanli.me/blog/2020/a-half-hour-to-learn-rust/

// TODO: Convert at least some of the many Vec::new to Vec::with_capacity
fn main() {
  let mut f = File::open("doom.wad").unwrap();
  let mut wad_file = Vec::new();
  f.read_to_end(&mut wad_file).unwrap();

  let lumps = lumps::load(&wad_file);

  let current_map = maps::load("^E1M1$", &wad_file, &lumps);
  // map_svg::draw_map_svg(current_map);

  let wall_texture_names_to_gl_textures: HashMap<std::string::String, glium::texture::SrgbTexture2d> = HashMap::new();

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
    wall_texture_names_to_gl_textures: wall_texture_names_to_gl_textures,
  };

  render(scene);
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
  wall_texture_names_to_gl_textures: HashMap<std::string::String, glium::texture::SrgbTexture2d>,
}

// ---------- OpenGL-specific structs
#[derive(Copy, Clone)]
struct GLVertex {
  position: [f32; 3],
  normal: [f32; 3],
  tex_coords: [f32; 2],
}

struct GLTexturedWall {
  gl_vertices: glium::VertexBuffer<GLVertex>,
  texture_name: Option<String>,
}

use glium::glutin;

fn build_wall_quad(
  vertex_1: &MapVertex,
  vertex_2: &MapVertex,
  wall_height_bottom: f32,
  wall_height_top: f32,
) -> [GLVertex; 4] {
  // C *------* D
  //   | \  2 |
  //   |  \   |
  //   | 1 \  |
  //   |    \ |
  // A *------* B

  // https://en.wikipedia.org/wiki/Triangle_strip -- only 4 verts needed to draw two triangles.

  // TODO: Calculate actual quad normals... right now, they're all negative in the z direction
  [
    GLVertex {
      // A
      position: [vertex_1.x as f32, wall_height_bottom as f32, vertex_1.y as f32],
      normal: [0.0, 0.0, -1.0],
      tex_coords: [0.0, 0.0],
    },
    GLVertex {
      // B
      position: [vertex_2.x as f32, wall_height_bottom as f32, vertex_2.y as f32],
      normal: [0.0, 0.0, -1.0],
      tex_coords: [1.0, 0.0],
    },
    GLVertex {
      // C
      position: [vertex_1.x as f32, wall_height_top as f32, vertex_1.y as f32],
      normal: [0.0, 0.0, -1.0],
      tex_coords: [0.0, 1.0],
    },
    GLVertex {
      // D
      position: [vertex_2.x as f32, wall_height_top as f32, vertex_2.y as f32],
      normal: [0.0, 0.0, -1.0],
      tex_coords: [1.0, 1.0],
    },
  ]
}

fn build_floor(
  verts_for_floor: &Vec<(f32, f32)>,
  sector: &Sector,
  display: &glium::Display,
) -> (glium::VertexBuffer<GLVertex>, glium::IndexBuffer<u16>) {
  // TODO: Try https://github.com/nical/lyon/ to tessellate floors, ceilings
  // ### https://github.com/nical/lyon/tree/master/examples/wgpu ###
  // -- main shape (drawn with paths, then 9 holes, drawn as paths within)
  // https://nical.github.io/lyon-doc/src/lyon_extra/extra/src/rust_logo.rs.html#4-269
  //
  //
  // How GzDoom seems to do it.
  // https://github.com/coelckers/gzdoom/blob/76db26ee0be6ab74d468d11bc9de9dfde6f5ed28/src/common/thirdparty/earcut.hpp

  let mut path_builder = Path::builder();
  let mut vert_index = 0;

  for floor_vert in verts_for_floor {
    let x = floor_vert.0;
    let z = floor_vert.1;

    if vert_index == 0 {
      path_builder.move_to(point(x, z));
    } else {
      path_builder.line_to(point(x, z));
    }

    vert_index += 1;
  }
  path_builder.close();
  let path = path_builder.build();

  let mut buffers: VertexBuffers<Point, u16> = VertexBuffers::new();
  let mut vertex_builder = simple_builder(&mut buffers);
  let mut tessellator = FillTessellator::new();

  let tolerance = 0.02;

  // Compute the tessellation.
  // let result = tessellator.tessellate_path(&path, &FillOptions::default(), &mut vertex_builder);

  // https://docs.rs/lyon_tessellation/0.15.6/lyon_tessellation/
  let result = tessellator.tessellate_path(
    &path,
    &FillOptions::tolerance(tolerance).with_fill_rule(lyon::tessellation::FillRule::NonZero),
    &mut vertex_builder,
  );

  assert!(result.is_ok());

  // println!("The generated vertices are: {:?}.", &buffers.vertices[..]);
  // println!("The generated indices are: {:?}.", &buffers.indices[..]);

  let mut floor_gl_verts = Vec::new();

  for floor_vert in &buffers.vertices {
    let new_vert = GLVertex {
      position: [floor_vert.x, sector.floor_height as f32, floor_vert.y],
      normal: [0.0, 1.0, 0.0],
      tex_coords: [0.0, 0.0],
    };

    floor_gl_verts.push(new_vert);
  }

  let new_floor_vbo = glium::vertex::VertexBuffer::new(display, &floor_gl_verts).unwrap();

  // https://docs.rs/glium/0.27.0/glium/index/enum.PrimitiveType.html

  let new_floor_ibo =
    glium::IndexBuffer::new(display, glium::index::PrimitiveType::TriangleStrip, &buffers.indices).unwrap();

  (new_floor_vbo, new_floor_ibo)
}

impl Scene {
  fn texture_to_gl_texture(&self, texture_name: &str, display: &glium::Display) -> glium::texture::SrgbTexture2d {
    println!("[texture_to_gl_texture] Loading texture: {}", texture_name);

    let texture = self.textures.iter().find(|&t| t.name == texture_name).unwrap();
    let mut imgbuf = image::ImageBuffer::<image::Rgba<u8>, Vec<u8>>::new(texture.width as u32, texture.height as u32);

    for patch in &texture.patches {
      let fetched_patch = &self.patch_names[patch.patch_number];

      let patch_picture = wad_graphics::load_picture_from_wad(&self.wad_file, &self.lumps, &fetched_patch.name);

      let patch_bytes = wad_graphics::picture_to_rgba_bytes(&patch_picture, &self.palette);

      let mut patch_x: i32 = 0;
      let mut patch_y: i32 = 0;

      for raw_pixel in patch_bytes.chunks(4) {
        // println!("patch_x: {}, patch_y: {}", patch_x, patch_y);

        let target_x = patch_x + patch.originx as i32;
        let target_y = patch_y + patch.originy as i32;

        // TODO: Deal with negative patch offsets??

        if target_x < texture.width as i32 && target_y < texture.height as i32 && target_x >= 0 && target_y >= 0 {
          imgbuf.put_pixel(
            target_x as u32,
            target_y as u32,
            image::Rgba([raw_pixel[0], raw_pixel[1], raw_pixel[2], raw_pixel[3]]),
          );
        }
        patch_x += 1;
        if patch_x >= patch_picture.width as i32 {
          patch_x = 0;
          patch_y += 1;
        }
      }
    }

    let gl_image = glium::texture::RawImage2d::from_raw_rgba_reversed(
      &imgbuf.into_raw(),
      (texture.width as u32, texture.height as u32),
    );
    let gl_texture = glium::texture::SrgbTexture2d::new(display, gl_image).unwrap();

    gl_texture
  }
}

fn render(mut scene: Scene) {
  // let lost_soul_sprite = wad_graphics::load_picture_from_wad(&self.wad_file, &self.lumps, "SKULA1");
  // let title_screen = wad_graphics::load_picture_from_wad(&self.wad_file, &self.lumps, "TITLEPIC");
  // let some_flat = wad_graphics::load_flat_from_wad(&self.wad_file, &self.lumps, "NUKAGE1");

  // png_dump::dump_picture(&title_screen, &self.palette);
  // png_dump::dump_picture(&lost_soul_sprite, &self.palette);
  // WAD SETUP END

  #[allow(unused_imports)]
  use glium::{glutin, Surface};

  let event_loop = glutin::event_loop::EventLoop::new();
  let window_builder = glutin::window::WindowBuilder::new();

  let context_builder = glutin::ContextBuilder::new().with_depth_buffer(24);
  let display = glium::Display::new(window_builder, context_builder, &event_loop).unwrap();

  implement_vertex!(GLVertex, position, normal, tex_coords);

  // DEFAULT GL TEXTURE (start)
  let image = image::load(
    Cursor::new(&include_bytes!("../tuto-14-diffuse.jpg")[..]),
    image::ImageFormat::Jpeg,
  )
  .unwrap()
  .to_rgba();

  let image_dimensions = image.dimensions();
  let image = glium::texture::RawImage2d::from_raw_rgba_reversed(&image.into_raw(), image_dimensions);
  let diffuse_texture = glium::texture::SrgbTexture2d::new(&display, image).unwrap();

  let image = image::load(
    Cursor::new(&include_bytes!("../tuto-14-normal.png")[..]),
    image::ImageFormat::Png,
  )
  .unwrap()
  .to_rgba();
  let image_dimensions = image.dimensions();
  let image = glium::texture::RawImage2d::from_raw_rgba_reversed(&image.into_raw(), image_dimensions);
  let normal_map = glium::texture::Texture2d::new(&display, image).unwrap();

  let vertex_shader_src = r#"
    #version 150
    
    in vec3 position;
    in vec3 normal;
    in vec2 tex_coords;
    
    out vec3 v_normal;
    out vec3 v_position;
    out vec2 v_tex_coords;
    
    uniform mat4 perspective;
    uniform mat4 view;
    uniform mat4 model;

    void main() {
      // passes through tex_coords unaffacted
      v_tex_coords = tex_coords;

      // compute modelView for easier math below
      mat4 modelview = view * model;

      // corrects normals after non-uniform scaling (different scales in different axes)
      v_normal = transpose(inverse(mat3(modelview))) * normal; 

      // all of the matrix multiplies. vertex position is mulitplied to model * world (camera) position.
      // then multiplied by camera perspective (for FOV, clipping, znear/zfar) etc.
      gl_Position = perspective * modelview * vec4(position, 1.0);

      // pass v_position to fragment shader
      v_position = gl_Position.xyz / gl_Position.w;
    }
  "#;

  let fragment_shader_src = r#"
    #version 140
    
    in vec3 v_normal;
    in vec3 v_position;
    in vec2 v_tex_coords;
    
    out vec4 color;
    
    uniform vec3 u_light;
    uniform sampler2D diffuse_tex;
    uniform sampler2D normal_tex;
    
    const vec3 specular_color = vec3(1.0, 1.0, 1.0);
    
    // original source? http://www.thetenthplanet.de/archives/1180
    mat3 cotangent_frame(vec3 normal, vec3 pos, vec2 uv) {
      vec3 dp1 = dFdx(pos);
      vec3 dp2 = dFdy(pos);
      vec2 duv1 = dFdx(uv);
      vec2 duv2 = dFdy(uv);
      vec3 dp2perp = cross(dp2, normal);
      vec3 dp1perp = cross(normal, dp1);
      vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
      vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;
      float invmax = inversesqrt(max(dot(T, T), dot(B, B)));
      return mat3(T * invmax, B * invmax, normal);
    }

    void main() {
      vec3 diffuse_color = texture(diffuse_tex, v_tex_coords).rgb;
      vec3 ambient_color = diffuse_color * 0.1;
      vec3 normal_map = texture(normal_tex, v_tex_coords).rgb;
      mat3 tbn = cotangent_frame(v_normal, v_position, v_tex_coords);
      vec3 real_normal = normalize(tbn * -(normal_map * 2.0 - 1.0));
      float diffuse = max(dot(real_normal, normalize(u_light)), 0.0);
      vec3 camera_dir = normalize(-v_position);
      vec3 half_direction = normalize(normalize(u_light) + camera_dir);
      float specular = pow(max(dot(half_direction, real_normal), 0.0), 16.0);
      color = vec4(ambient_color + diffuse * diffuse_color + specular * specular_color, 1.0);
    }
  "#;

  let program = glium::Program::from_source(&display, vertex_shader_src, fragment_shader_src, None).unwrap();
  // DEFAULT GL TEXTURE (end)

  let fragment_shader_src_2 = r#"
    #version 140
    
    in vec3 v_normal;
    in vec3 v_position;
    in vec2 v_tex_coords;
    
    out vec4 color;
    
    uniform sampler2D diffuse_tex;
    
    void main() {
      color = texture(diffuse_tex, v_tex_coords);
    }
  "#;
  let program_2 = glium::Program::from_source(&display, vertex_shader_src, fragment_shader_src_2, None).unwrap();

  let mut walls: Vec<GLTexturedWall> = Vec::new();
  let mut floors: Vec<(glium::VertexBuffer<GLVertex>, glium::IndexBuffer<u16>)> = Vec::new();

  let mut vert_tuples_by_sector_id = HashMap::new();

  let mut linedef_index = 0;

  for line in &scene.map.linedefs {
    // if linedef_index != 6 {
    //   continue;
    // }

    let start_vertex_index = line.start_vertex;
    let end_vertex_index = line.end_vertex;

    let start_vertex = &scene.map.vertexes[start_vertex_index];
    let end_vertex = &scene.map.vertexes[end_vertex_index];

    let mut front_sector_floor_height: f32 = -1.0;
    let mut front_sector_ceiling_height: f32 = -1.0;
    let mut back_sector_floor_height: f32 = -1.0;
    let mut back_sector_ceiling_height: f32 = -1.0;

    let front_sidedef = if let Some(front_sidedef_index) = line.front_sidedef_index {
      let front_sidedef = &scene.map.sidedefs[front_sidedef_index];
      let front_sector = &scene.map.sectors[front_sidedef.sector_facing];
      front_sector_floor_height = front_sector.floor_height as f32;
      front_sector_ceiling_height = front_sector.ceiling_height as f32;

      let keyed_vertex_vector = vert_tuples_by_sector_id
        .entry(front_sidedef.sector_facing)
        .or_insert(Vec::<(f32, f32)>::new());
      keyed_vertex_vector.push((start_vertex.x as f32, start_vertex.y as f32));
      keyed_vertex_vector.push((end_vertex.x as f32, end_vertex.y as f32));

      Some(front_sidedef)
    } else {
      None
    };

    let back_sidedef = if let Some(back_sidedef_index) = line.back_sidedef_index {
      let back_sidedef = &scene.map.sidedefs[back_sidedef_index];
      let back_sector = &scene.map.sectors[back_sidedef.sector_facing];
      back_sector_floor_height = back_sector.floor_height as f32;
      back_sector_ceiling_height = back_sector.ceiling_height as f32;

      Some(back_sidedef)
    } else {
      None
    };

    // Simple walls: front
    if let Some(fside) = front_sidedef {
      if fside.name_of_middle_texture.is_some()
        && fside.name_of_upper_texture.is_none()
        && fside.name_of_lower_texture.is_none()
      {
        let texture_name = fside.name_of_middle_texture.clone().unwrap();

        let new_simple_wall = build_wall_quad(
          start_vertex,
          end_vertex,
          front_sector_floor_height,
          front_sector_ceiling_height,
        );
        let new_simple_wall_vertex_buffer = glium::vertex::VertexBuffer::new(&display, &new_simple_wall).unwrap();

        let new_gl_textured_wall = GLTexturedWall {
          gl_vertices: new_simple_wall_vertex_buffer,
          texture_name: Some(texture_name),
        };
        walls.push(new_gl_textured_wall);
      }
    }

    // Simple walls: back
    if let Some(bside) = back_sidedef {
      if bside.name_of_middle_texture.is_some()
        && bside.name_of_upper_texture.is_none()
        && bside.name_of_lower_texture.is_none()
      {
        let texture_name = bside.name_of_middle_texture.clone().unwrap();

        let new_simple_wall = build_wall_quad(
          end_vertex,
          start_vertex,
          back_sector_floor_height,
          back_sector_ceiling_height,
        );
        let new_simple_wall_vertex_buffer = glium::vertex::VertexBuffer::new(&display, &new_simple_wall).unwrap();

        let new_gl_textured_wall = GLTexturedWall {
          gl_vertices: new_simple_wall_vertex_buffer,
          texture_name: Some(texture_name),
        };
        walls.push(new_gl_textured_wall);
      }
    }

    // lower walls: front (nearly the same as ... back below)
    if let Some(fside) = front_sidedef {
      if fside.name_of_middle_texture.is_none()
        && (fside.name_of_upper_texture.is_some() // NOTE some vs. none here
        || fside.name_of_lower_texture.is_some())
      {
        /////////// UP STEP
        let low_y: f32;
        let high_y: f32;

        if front_sector_floor_height < back_sector_floor_height {
          low_y = front_sector_floor_height;
          high_y = back_sector_floor_height;
        } else {
          low_y = back_sector_floor_height;
          high_y = front_sector_floor_height;
        }

        if low_y != high_y {
          let front_up_step = build_wall_quad(start_vertex, end_vertex, low_y, high_y);
          let front_up_step_vertex_buffer = glium::vertex::VertexBuffer::new(&display, &front_up_step).unwrap();

          let texture_name = if fside.name_of_lower_texture.is_some() {
            fside.name_of_lower_texture.clone()
          } else {
            None
          };

          let new_gl_textured_wall = GLTexturedWall {
            gl_vertices: front_up_step_vertex_buffer,
            texture_name: texture_name,
          };
          walls.push(new_gl_textured_wall);
        }

        /////////// DOWN STEP
        let low_y: f32;
        let high_y: f32;

        if front_sector_ceiling_height < back_sector_ceiling_height {
          low_y = front_sector_ceiling_height;
          high_y = back_sector_ceiling_height;
        } else {
          low_y = back_sector_ceiling_height;
          high_y = front_sector_ceiling_height;
        }

        if low_y != high_y {
          let front_down_step = build_wall_quad(start_vertex, end_vertex, low_y, high_y);
          let front_down_step_vertex_buffer = glium::vertex::VertexBuffer::new(&display, &front_down_step).unwrap();

          let texture_name = if fside.name_of_upper_texture.is_some() {
            fside.name_of_upper_texture.clone()
          } else {
            None
          };

          let new_gl_textured_wall = GLTexturedWall {
            gl_vertices: front_down_step_vertex_buffer,
            texture_name: texture_name,
          };
          walls.push(new_gl_textured_wall);
        }
      }
    }

    // lower walls: back (nearly the same as ... front above)
    if let Some(bside) = back_sidedef {
      if bside.name_of_middle_texture.is_none()
        && (bside.name_of_upper_texture.is_some() // NOTE some vs. none here
        || bside.name_of_lower_texture.is_some())
      {
        /////////// UP STEP
        let low_y: f32;
        let high_y: f32;

        if front_sector_floor_height < back_sector_floor_height {
          low_y = front_sector_floor_height;
          high_y = back_sector_floor_height;
        } else {
          low_y = back_sector_floor_height;
          high_y = front_sector_floor_height;
        }

        if low_y != high_y {
          let front_up_step = build_wall_quad(end_vertex, start_vertex, low_y, high_y);
          let front_up_step_vertex_buffer = glium::vertex::VertexBuffer::new(&display, &front_up_step).unwrap();

          let texture_name = if bside.name_of_lower_texture.is_some() {
            bside.name_of_lower_texture.clone()
          } else {
            None
          };

          let new_gl_textured_wall = GLTexturedWall {
            gl_vertices: front_up_step_vertex_buffer,
            texture_name: texture_name,
          };
          walls.push(new_gl_textured_wall);
        }

        /////////// DOWN STEP
        let low_y: f32;
        let high_y: f32;

        if front_sector_ceiling_height < back_sector_ceiling_height {
          low_y = front_sector_ceiling_height;
          high_y = back_sector_ceiling_height;
        } else {
          low_y = back_sector_ceiling_height;
          high_y = front_sector_ceiling_height;
        }

        if low_y != high_y {
          let front_down_step = build_wall_quad(end_vertex, start_vertex, low_y, high_y);
          let front_down_step_vertex_buffer = glium::vertex::VertexBuffer::new(&display, &front_down_step).unwrap();

          let texture_name = if bside.name_of_upper_texture.is_some() {
            bside.name_of_upper_texture.clone()
          } else {
            None
          };

          let new_gl_textured_wall = GLTexturedWall {
            gl_vertices: front_down_step_vertex_buffer,
            texture_name: texture_name,
          };
          walls.push(new_gl_textured_wall);
        }
      }
    }

    linedef_index += 1;
  }

  // Pre-caching all wall textures
  for wall in &walls {
    match &wall.texture_name {
      Some(texture_name) => {
        if !scene.wall_texture_names_to_gl_textures.contains_key(texture_name) {
          let gl_texture = scene.texture_to_gl_texture(&texture_name, &display);
          scene
            .wall_texture_names_to_gl_textures
            .insert(texture_name.clone(), gl_texture);
        }
      }
      _ => {}
    };
  }

  if true {
    // TODO: Floor rendering: disabled for now.
    let mut sector_index = 0;
    for sector in &scene.map.sectors {
      // TODO: Temporary: if sector_index == 1 {
      if sector_index == 1 {
        // TODO: Need to pass in verts for any overlapping sectors, too, to specify where holes are.
        if let Some(verts_for_floor) = vert_tuples_by_sector_id.get(&sector_index) {
          let (new_floor_vbo, new_floor_ibo) = build_floor(&verts_for_floor, &sector, &display);
          floors.push((new_floor_vbo, new_floor_ibo));
        }
      }

      sector_index += 1;
    }
  }

  ////////

  let mut camera = camera::Camera::new([1031.2369, 66.481995, -3472.9282], -2460.6008, 17.500008);

  event_loop.run(move |event, _, control_flow| {
    let next_frame_time = std::time::Instant::now() + std::time::Duration::from_nanos(16_666_667);
    *control_flow = glutin::event_loop::ControlFlow::WaitUntil(next_frame_time);
    use glutin::event::ElementState;

    match event {
      glutin::event::Event::WindowEvent { event, .. } => match event {
        glutin::event::WindowEvent::CloseRequested => {
          *control_flow = glutin::event_loop::ControlFlow::Exit;
          return;
        }
        glutin::event::WindowEvent::KeyboardInput { input, .. } => match (input.virtual_keycode, input.state) {
          (Some(keycode), ElementState::Pressed) => camera.handle_keypress(keycode),
          _ => (),
        },
        glutin::event::WindowEvent::CursorMoved { position, .. } => camera.handle_mouse_move(position),
        _ => return,
      },
      glutin::event::Event::NewEvents(cause) => match cause {
        glutin::event::StartCause::ResumeTimeReached { .. } => (),
        glutin::event::StartCause::Init => (),
        _ => return,
      },
      _ => return,
    }

    let mut target = display.draw();
    target.clear_color_and_depth((0.0, 0.0, 1.0, 1.0), 1.0);

    // The root transform of the world (all walls)
    let model = [
      [1.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 1.0f32],
    ];

    // The camera transform (view)
    let view = camera.get_world_to_view_matrix();
    let view: [[f32; 4]; 4] = view.into();

    let perspective = {
      let (width, height) = target.get_dimensions();
      let aspect_ratio = height as f32 / width as f32;

      let fov: f32 = 3.141592 / 2.0; // NOTE: 90* FOV, like the original doom
      let zfar = 100_000.0;
      let znear = 0.001;

      // focal length?
      let f = 1.0 / (fov / 2.0).tan();

      [
        [f * aspect_ratio, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (zfar + znear) / (zfar - znear), 1.0],
        [0.0, 0.0, -(2.0 * zfar * znear) / (zfar - znear), 0.0],
      ]
    };

    let light = [1.4, 0.4, 0.7f32];

    let params = glium::DrawParameters {
      depth: glium::Depth {
        test: glium::draw_parameters::DepthTest::IfLess,
        write: true,
        ..Default::default()
      },
      // backface_culling: glium::draw_parameters::BackfaceCullingMode::CullClockwise, // TODO: turn this back on
      ..Default::default()
    };

    // Check Doom Builder on windows,
    // gzdoom

    for wall in &walls {
      match &wall.texture_name {
        Some(texture_name) => {
          let fetched_diffuse_tex = scene
            .wall_texture_names_to_gl_textures
            .get(texture_name)
            .unwrap()
            .sampled()
            .magnify_filter(glium::uniforms::MagnifySamplerFilter::Nearest);

          target
            .draw(
              &wall.gl_vertices,
              glium::index::NoIndices(glium::index::PrimitiveType::TriangleStrip),
              &program_2,
              &uniform! {
                model: model,
                view: view,
                perspective: perspective,
                diffuse_tex: fetched_diffuse_tex,
              },
              &params,
            )
            .unwrap();
        }
        _ => {
          // target
          //   .draw(
          //     &wall.gl_vertices,
          //     glium::index::NoIndices(glium::index::PrimitiveType::TriangleStrip),
          //     &program,
          //     &uniform! {
          //       model: model,
          //       view: view,
          //       perspective: perspective,
          //       u_light: light,
          //       diffuse_tex: &diffuse_texture,
          //       normal_tex: &normal_map
          //     },
          //     &params,
          //   )
          //   .unwrap();
        }
      };
    }

    for floor in &floors {
      let floor_vbo = &floor.0;
      let floor_ibo = &floor.1;

      target
        .draw(
          floor_vbo,
          floor_ibo,
          &program,
          &uniform! {
            model: model,
            view: view,
            perspective: perspective,
            u_light: light,
            diffuse_tex: &diffuse_texture,
            normal_tex: &normal_map
          },
          &params,
        )
        .unwrap();
    }

    target.finish().unwrap();
  });
}
