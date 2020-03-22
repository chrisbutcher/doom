#[macro_use]
extern crate glium;
extern crate image;

extern crate regex;

use std::fs::File;
use std::io::prelude::*;
use std::io::Cursor;

pub mod lumps;
pub mod map_svg;
pub mod maps;

// TODO: Read this https://fasterthanli.me/blog/2020/a-half-hour-to-learn-rust/

fn main() {
  let mut f = File::open("doomu.wad").unwrap();
  let mut wad_file = Vec::new();
  f.read_to_end(&mut wad_file).unwrap();

  let lumps = lumps::load(&wad_file);
  let maps = maps::load(&wad_file, lumps);

  let current_map = &maps[0];

  // for map in maps {
  //   draw_map_svg(&map);
  // }
  map_svg::draw_map_svg(current_map);

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
  map_centerer: map_svg::MapCenterer,
  sectors: Vec<Sector>,
}

#[derive(Debug, Clone)]
pub struct LineDef {
  // TODO: Flags, special type, sector tag: https://doomwiki.org/wiki/Linedef
  start_vertex: usize,
  end_vertex: usize,
  front_sidedef: usize,
  back_sidedef: usize,
}

#[derive(Debug, Clone)]
pub struct SideDef {
  x_offset: i16,
  y_offset: i16,
  name_of_upper_texture: String,
  name_of_lower_texture: String,
  name_of_middle_texture: String,
  sector_facing: usize,
}

#[derive(Debug, Clone)]
pub struct Sector {
  floor_height: i16,
  ceiling_height: i16,
  name_of_floor_texture: String,
  name_of_ceiling_texture: String,
  light_level: i16,
  sector_type: i16,
  tag_number: i16,
}

use glium::glutin;
fn update_camera(
  current_position: [f32; 3],
  current_rotation: [f32; 3],
  key_code: glutin::event::VirtualKeyCode,
) -> ([f32; 3], [f32; 3]) {
  let mut new_position = current_position;
  let mut new_rotation = current_rotation;

  use glutin::event::VirtualKeyCode;

  const MOVE_SPEED: f32 = 50.0;
  const ROTATE_SPEED: f32 = 0.1;

  match key_code {
    VirtualKeyCode::W => {
      new_position[2] += MOVE_SPEED;
    }
    VirtualKeyCode::A => {
      new_position[0] -= MOVE_SPEED;
    }
    VirtualKeyCode::S => {
      new_position[2] -= MOVE_SPEED;
    }
    VirtualKeyCode::D => {
      new_position[0] += MOVE_SPEED;
    }
    VirtualKeyCode::Up => {
      new_position[1] += MOVE_SPEED;
    }
    VirtualKeyCode::Down => {
      new_position[1] -= MOVE_SPEED;
    }
    VirtualKeyCode::Q => {
      new_rotation[0] -= ROTATE_SPEED;
    }
    VirtualKeyCode::E => {
      new_rotation[0] += ROTATE_SPEED;
    }
    _ => (),
  }

  (new_position, new_rotation)
}

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

  let mut shapes = Vec::new();

  let bar = &map.vertexes[400];
  let first_vertex = bar.clone();

  const WALL_HEIGHT: f32 = 50.0;

  for line in &map.linedefs {
    let start_vertex_index = line.start_vertex;
    let end_vertex_index = line.end_vertex;

    let start_vertex = &map.vertexes[start_vertex_index];
    let end_vertex = &map.vertexes[end_vertex_index];

    let foo = [
      Vertex {
        position: [start_vertex.x as f32, 0.0, start_vertex.y as f32],
        normal: [0.0, 0.0, -1.0],
        tex_coords: [0.0, 0.0],
      },
      Vertex {
        position: [end_vertex.x as f32, 0.0, end_vertex.y as f32],
        normal: [0.0, 0.0, -1.0],
        tex_coords: [1.0, 0.0],
      },
      Vertex {
        position: [start_vertex.x as f32, WALL_HEIGHT, start_vertex.y as f32],
        normal: [0.0, 0.0, -1.0],
        tex_coords: [0.0, 1.0],
      },
      Vertex {
        position: [end_vertex.x as f32, 0.0, end_vertex.y as f32],
        normal: [0.0, 0.0, -1.0],
        tex_coords: [1.0, 0.0],
      },
      Vertex {
        position: [end_vertex.x as f32, WALL_HEIGHT, end_vertex.y as f32],
        normal: [0.0, 0.0, -1.0],
        tex_coords: [1.0, 1.0],
      },
      Vertex {
        position: [start_vertex.x as f32, WALL_HEIGHT, start_vertex.y as f32],
        normal: [0.0, 0.0, -1.0],
        tex_coords: [0.0, 1.0],
      },
    ];

    let shape2 = glium::vertex::VertexBuffer::new(&display, &foo).unwrap();
    shapes.push(shape2);
  }

  let image = image::load(
    Cursor::new(&include_bytes!("../tuto-14-diffuse.jpg")[..]),
    image::ImageFormat::Jpeg,
  )
  .unwrap()
  .to_rgba();

  let image_dimensions = image.dimensions();
  let image =
    glium::texture::RawImage2d::from_raw_rgba_reversed(&image.into_raw(), image_dimensions);
  let diffuse_texture = glium::texture::SrgbTexture2d::new(&display, image).unwrap();

  let image = image::load(
    Cursor::new(&include_bytes!("../tuto-14-normal.png")[..]),
    image::ImageFormat::Png,
  )
  .unwrap()
  .to_rgba();
  let image_dimensions = image.dimensions();
  let image =
    glium::texture::RawImage2d::from_raw_rgba_reversed(&image.into_raw(), image_dimensions);
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
      v_tex_coords = tex_coords;
      mat4 modelview = view * model;
      v_normal = transpose(inverse(mat3(modelview))) * normal;
      gl_Position = perspective * modelview * vec4(position, 1.0);
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

  // let camera_position = [0.5, 0.0, -3.0];
  let mut camera_position = [
    first_vertex.x as f32,
    700.0 as f32,
    (first_vertex.y - 4000) as f32,
  ];

  let mut camera_rotation = [-0.5, -0.6, 4.0];

  let program =
    glium::Program::from_source(&display, vertex_shader_src, fragment_shader_src, None).unwrap();

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
        glutin::event::WindowEvent::KeyboardInput { input, .. } => {
          match (input.virtual_keycode, input.state) {
            (Some(keycode), ElementState::Pressed) => {
              println!("{:?}", keycode);
              let result = update_camera(camera_position, camera_rotation, keycode);
              camera_position = result.0;
              camera_rotation = result.1;
            }
            _ => (),
          }
        }
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

    let model = [
      [1.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 1.0f32],
    ];

    let view = view_matrix(&camera_position, &camera_rotation, &[0.0, 1.0, 0.0]);

    let perspective = {
      let (width, height) = target.get_dimensions();
      let aspect_ratio = height as f32 / width as f32;

      let fov: f32 = 3.141592 / 3.0;
      let zfar = 100_000.0;
      let znear = 0.1;

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

    for shape in &shapes {
      target
        .draw(
          shape,
          glium::index::NoIndices(glium::index::PrimitiveType::TriangleStrip),
          &program,
          &uniform! { model: model, view: view, perspective: perspective,
          u_light: light, diffuse_tex: &diffuse_texture, normal_tex: &normal_map },
          &params,
        )
        .unwrap();
    }
    target.finish().unwrap();
  });
}

fn view_matrix(position: &[f32; 3], direction: &[f32; 3], up: &[f32; 3]) -> [[f32; 4]; 4] {
  let f = {
    let f = direction;
    let len = f[0] * f[0] + f[1] * f[1] + f[2] * f[2];
    let len = len.sqrt();
    [f[0] / len, f[1] / len, f[2] / len]
  };

  let s = [
    up[1] * f[2] - up[2] * f[1],
    up[2] * f[0] - up[0] * f[2],
    up[0] * f[1] - up[1] * f[0],
  ];

  let s_norm = {
    let len = s[0] * s[0] + s[1] * s[1] + s[2] * s[2];
    let len = len.sqrt();
    [s[0] / len, s[1] / len, s[2] / len]
  };

  let u = [
    f[1] * s_norm[2] - f[2] * s_norm[1],
    f[2] * s_norm[0] - f[0] * s_norm[2],
    f[0] * s_norm[1] - f[1] * s_norm[0],
  ];

  let p = [
    -position[0] * s_norm[0] - position[1] * s_norm[1] - position[2] * s_norm[2],
    -position[0] * u[0] - position[1] * u[1] - position[2] * u[2],
    -position[0] * f[0] - position[1] * f[1] - position[2] * f[2],
  ];

  [
    [s_norm[0], u[0], f[0], 0.0],
    [s_norm[1], u[1], f[1], 0.0],
    [s_norm[2], u[2], f[2], 0.0],
    [p[0], p[1], p[2], 1.0],
  ]
}
