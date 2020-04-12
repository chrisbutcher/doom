#[macro_use]
extern crate glium;
extern crate image;

extern crate regex;

extern crate nalgebra_glm as glm;

use std::fs::File;
use std::io::prelude::*;
use std::io::Cursor;

pub mod camera;
pub mod lumps;
pub mod map_svg;
pub mod maps;

// TODO: Read this https://fasterthanli.me/blog/2020/a-half-hour-to-learn-rust/

fn main() {
  let mut f = File::open("doom.wad").unwrap();
  let mut wad_file = Vec::new();
  f.read_to_end(&mut wad_file).unwrap();

  let lumps = lumps::load(&wad_file);
  let maps = maps::load(&wad_file, lumps);

  let current_map = &maps[0];

  // for map in maps {
  //   draw_map_svg(&map);
  // }
  // map_svg::draw_map_svg(current_map);

  println!("{:?}", current_map);

  render_scene(current_map);
}

#[derive(Debug)]
pub struct Lump {
  filepos: usize,
  size: usize,
  name: String,
}

// ORIGIN is top-left. y axis grows downward (as in, subtract to go up).

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
  front_sidedef: Option<usize>,
  back_sidedef: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct SideDef {
  x_offset: i16,
  y_offset: i16,
  name_of_upper_texture: Option<String>,
  name_of_lower_texture: Option<String>,
  name_of_middle_texture: Option<String>,
  sector_facing: usize,
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

use glium::glutin;

fn render_scene(map: &Map) {
  #[allow(unused_imports)]
  use glium::{glutin, Surface};

  let event_loop = glutin::event_loop::EventLoop::new();
  let window_builder = glutin::window::WindowBuilder::new();

  let context_builder = glutin::ContextBuilder::new().with_depth_buffer(24);
  let display = glium::Display::new(window_builder, context_builder, &event_loop).unwrap();

  #[derive(Copy, Clone)]
  struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    tex_coords: [f32; 2],
  }

  implement_vertex!(Vertex, position, normal, tex_coords);

  let mut walls = Vec::new();

  const WALL_HEIGHT: f32 = 1.0;

  for line in &map.linedefs {
    let start_vertex_index = line.start_vertex;
    let end_vertex_index = line.end_vertex;

    let start_vertex = &map.vertexes[start_vertex_index];
    let end_vertex = &map.vertexes[end_vertex_index];

    let front_side = if let Some(front_sidedef) = line.front_sidedef {
      Some(&map.sidedefs[front_sidedef])
    } else {
      None
    };

    let back_side = if let Some(back_sidedef) = line.back_sidedef {
      Some(&map.sidedefs[back_sidedef])
    } else {
      None
    };

    let front_sector = match front_side {
      Some(front_sidedef) => Some(&map.sectors[front_sidedef.sector_facing]),
      None => (None),
    };

    // TODO use this
    let back_sector = match back_side {
      Some(back_sidedef) => Some(&map.sectors[back_sidedef.sector_facing]),
      None => (None),
    };

    // C *------* D
    //   | \  2 |
    //   |  \   |
    //   | 1 \  |
    //   |    \ |
    // A *------* B

    match front_sector {
      Some(front_sector) => {
        let front_wall = [
          Vertex {
            // A1
            position: [
              start_vertex.x as f32,
              front_sector.floor_height as f32 * WALL_HEIGHT,
              start_vertex.y as f32,
            ],
            normal: [0.0, 0.0, -1.0],
            tex_coords: [0.0, 0.0],
          },
          Vertex {
            // B1
            position: [
              end_vertex.x as f32,
              front_sector.floor_height as f32 * WALL_HEIGHT,
              end_vertex.y as f32,
            ],
            normal: [0.0, 0.0, -1.0],
            tex_coords: [1.0, 0.0],
          },
          Vertex {
            // C1
            position: [
              start_vertex.x as f32,
              front_sector.ceiling_height as f32 * WALL_HEIGHT,
              start_vertex.y as f32,
            ],
            normal: [0.0, 0.0, -1.0],
            tex_coords: [0.0, 1.0],
          },
          Vertex {
            // B2
            position: [
              end_vertex.x as f32,
              front_sector.floor_height as f32 * WALL_HEIGHT,
              end_vertex.y as f32,
            ],
            normal: [0.0, 0.0, -1.0],
            tex_coords: [1.0, 0.0],
          },
          Vertex {
            // D2
            position: [
              end_vertex.x as f32,
              front_sector.ceiling_height as f32 * WALL_HEIGHT,
              end_vertex.y as f32,
            ],
            normal: [0.0, 0.0, -1.0],
            tex_coords: [1.0, 1.0],
          },
          Vertex {
            // C2
            position: [
              start_vertex.x as f32,
              front_sector.ceiling_height as f32 * WALL_HEIGHT,
              start_vertex.y as f32,
            ],
            normal: [0.0, 0.0, -1.0],
            tex_coords: [0.0, 1.0],
          },
        ];

        let new_front_wall = glium::vertex::VertexBuffer::new(&display, &front_wall).unwrap();
        walls.push(new_front_wall);
      }
      None => (),
    };
  }

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

  let mut camera = camera::Camera::new([860.4145, 250.0, -3634.698], -34.0);

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
          (Some(keycode), ElementState::Pressed) => camera.update_camera(keycode),
          _ => (),
        },
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

      let fov: f32 = 3.141592 / 3.0;
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
      ..Default::default()
    };

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
    target.finish().unwrap();
  });
}
