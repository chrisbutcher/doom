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
use lyon::path::builder::*;
use lyon::path::Path;
use lyon::tessellation::*;
// use lyon_tessellation::geometry_builder::simple_builder;
use lyon::tessellation::geometry_builder::simple_builder;

pub mod camera;
pub mod lumps;
pub mod map_svg;
pub mod maps;
pub mod palette;
pub mod wad_graphics;

// TODO: Read this https://fasterthanli.me/blog/2020/a-half-hour-to-learn-rust/

fn load_picture_from_wad(wad_file: &Vec<u8>, lumps: &Vec<Lump>, lump_name: &str) -> Picture {
  let picture_lump = lumps.iter().find(|&l| l.name == lump_name).unwrap();

  let lump_offset = picture_lump.filepos;

  let width = u16::from_le_bytes([wad_file[lump_offset], wad_file[lump_offset + 1]]);
  let height = u16::from_le_bytes([wad_file[lump_offset + 2], wad_file[lump_offset + 3]]);
  let leftoffset = u16::from_le_bytes([wad_file[lump_offset + 4], wad_file[lump_offset + 5]]);
  let topoffset = u16::from_le_bytes([wad_file[lump_offset + 6], wad_file[lump_offset + 7]]);

  println!("image width: {}", width);
  println!("image height: {}", height);

  // panic!("boom");

  let post_column_num = width * 8;

  let mut posts = Vec::new();

  let mut post_column_i = 0;
  for _ in 0..post_column_num - 1 {
    let post_column_offset = u32::from_le_bytes([
      wad_file[lump_offset + (post_column_i * 4) + 8],
      wad_file[lump_offset + (post_column_i * 4) + 9],
      wad_file[lump_offset + (post_column_i * 4) + 10],
      wad_file[lump_offset + (post_column_i * 4) + 11],
    ]) as usize;

    println!("width: {}", width);
    println!("width * 8: {}", width * 8);
    println!("height: {}", height);
    println!("post_column_i: {}", post_column_i);
    println!("lump_offset: {}", lump_offset);
    println!("post_column_offset: {}", post_column_offset);

    let topdelta = u8::from_le_bytes([wad_file[lump_offset + post_column_offset]]);
    let length = u8::from_le_bytes([wad_file[lump_offset + post_column_offset + 1]]);
    let _unused = u8::from_le_bytes([wad_file[lump_offset + post_column_offset + 2]]);

    let pixels_offset = lump_offset + post_column_offset + 3;

    let mut pixels = Vec::new();

    let mut post_i = 0;
    for j in 0..length {
      let pixel = u8::from_le_bytes([wad_file[pixels_offset + post_i]]) as usize;

      pixels.push(pixel);
      post_i += 1;
    }

    posts.push(PicturePost {
      topdelta: topdelta,
      length: length,
      pixels: pixels.to_owned(),
    });

    post_column_i += 1;
  }

  Picture {
    width: width,
    height: height,
    leftoffset: leftoffset,
    topoffset: topoffset,
    posts: posts,
  }
}

fn main() {
  let mut f = File::open("doom.wad").unwrap();
  let mut wad_file = Vec::new();
  f.read_to_end(&mut wad_file).unwrap();

  let lumps = lumps::load(&wad_file);
  let palette = palette::load_palette(&wad_file, &lumps);
  let textures = wad_graphics::load_textures(&wad_file, &lumps);
  let patch_names = wad_graphics::load_patch_names(&wad_file, &lumps);

  let title_pic = load_picture_from_wad(&wad_file, &lumps, "TITLEPIC");

  println!("{:?}", title_pic);

  panic!("boom");

  let maps = maps::load(&wad_file, &lumps);

  let current_map = &maps[0];

  // for map in maps {
  //   draw_map_svg(&map);
  // }
  // map_svg::draw_map_svg(current_map);

  // println!("{:?}", current_map);

  render_scene(current_map, &palette);
}

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
pub struct Picture {
  width: u16,
  height: u16,
  leftoffset: u16,
  topoffset: u16,
  posts: Vec<PicturePost>,
}

#[derive(Debug, Clone)]
pub struct PicturePost {
  topdelta: u8,
  length: u8,
  pixels: Vec<usize>,
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

// ---------- OpenGL-specific structs
#[derive(Copy, Clone)]
struct GLVertex {
  position: [f32; 3],
  normal: [f32; 3],
  tex_coords: [f32; 2],
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

  println!("The generated vertices are: {:?}.", &buffers.vertices[..]);
  println!("The generated indices are: {:?}.", &buffers.indices[..]);

  // panic!("boom");

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

fn render_scene(map: &Map, palette: &Vec<palette::PaletteColor>) {
  #[allow(unused_imports)]
  use glium::{glutin, Surface};

  let event_loop = glutin::event_loop::EventLoop::new();
  let window_builder = glutin::window::WindowBuilder::new();

  let context_builder = glutin::ContextBuilder::new().with_depth_buffer(24);
  let display = glium::Display::new(window_builder, context_builder, &event_loop).unwrap();

  implement_vertex!(GLVertex, position, normal, tex_coords);

  let mut walls: Vec<glium::VertexBuffer<GLVertex>> = Vec::new();
  let mut floors: Vec<(glium::VertexBuffer<GLVertex>, glium::IndexBuffer<u16>)> = Vec::new();

  let mut linedef_num = 0;

  let mut vert_tuples_by_sector_id = HashMap::new();

  for line in &map.linedefs {
    let start_vertex_index = line.start_vertex;
    let end_vertex_index = line.end_vertex;

    let start_vertex = &map.vertexes[start_vertex_index];
    let end_vertex = &map.vertexes[end_vertex_index];

    let mut front_sector_floor_height: f32 = -1.0;
    let mut front_sector_ceiling_height: f32 = -1.0;
    let mut back_sector_floor_height: f32 = -1.0;
    let mut back_sector_ceiling_height: f32 = -1.0;

    let front_sidedef = if let Some(front_sidedef_index) = line.front_sidedef_index {
      let front_sidedef = &map.sidedefs[front_sidedef_index];
      let front_sector = &map.sectors[front_sidedef.sector_facing];
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
      let back_sidedef = &map.sidedefs[back_sidedef_index];
      let back_sector = &map.sectors[back_sidedef.sector_facing];
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
        let new_simple_wall = build_wall_quad(
          start_vertex,
          end_vertex,
          front_sector_floor_height,
          front_sector_ceiling_height,
        );
        let new_simple_wall_vertex_buffer = glium::vertex::VertexBuffer::new(&display, &new_simple_wall).unwrap();
        walls.push(new_simple_wall_vertex_buffer);
      }
    }

    // Simple walls: back
    if let Some(bside) = back_sidedef {
      if bside.name_of_middle_texture.is_some()
        && bside.name_of_upper_texture.is_none()
        && bside.name_of_lower_texture.is_none()
      {
        let new_simple_wall = build_wall_quad(
          end_vertex,
          start_vertex,
          back_sector_floor_height,
          back_sector_ceiling_height,
        );
        let new_simple_wall_vertex_buffer = glium::vertex::VertexBuffer::new(&display, &new_simple_wall).unwrap();
        walls.push(new_simple_wall_vertex_buffer);
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
          walls.push(front_up_step_vertex_buffer);
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
          walls.push(front_down_step_vertex_buffer);
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
          walls.push(front_up_step_vertex_buffer);
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
          walls.push(front_down_step_vertex_buffer);
        }
      }
    }

    linedef_num += 1;
  }

  if true {
    // TODO: Floor rendering: disabled for now.
    let mut sector_index = 0;
    for sector in &map.sectors {
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

  let mut camera = camera::Camera::new([3927.552, 1258.45, -2268.088], -1043.7999, 35.100002);

  let program = glium::Program::from_source(&display, vertex_shader_src, fragment_shader_src, None).unwrap();

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
      target
        .draw(
          wall,
          glium::index::NoIndices(glium::index::PrimitiveType::TriangleStrip),
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
