#[macro_use]
extern crate glium;
extern crate image;

extern crate regex;

use std::fs::File;
use std::io::prelude::*;
use std::io::Cursor;

use image::{GenericImage, GenericImageView, ImageBuffer, Pixel, RgbImage};
use regex::Regex;

fn main() {
  let mut f = File::open("doomu.wad").unwrap();
  let mut wad_file = Vec::new();
  f.read_to_end(&mut wad_file).unwrap();

  let lumps = load_lumps(&wad_file);
  let maps = load_maps(&wad_file, lumps);

  println!("{:?}", maps[0]);

  generate_image();
}

fn generate_image() {
  let mut img: RgbImage = ImageBuffer::new(512, 512);
  let pixel = image::Rgb([255, 0, 0]);
  img.put_pixel(100, 100, pixel);
  img.save("testimage.png").unwrap();
}

#[derive(Debug, Clone)]
struct MapVertex {
  x: i16,
  y: i16,
}

#[derive(Debug)]
struct Map {
  name: String,
  vertexes: Vec<MapVertex>,
  linedefs: Vec<LineDef>,
}

#[derive(Debug, Clone)]
struct LineDef {
  // TODO: Flags, special type, sector tag: https://doomwiki.org/wiki/Linedef
  start_vertex: usize,
  end_vertex: usize,
  front_sidedef: usize,
  back_sidedef: usize,
}

#[derive(Debug, Clone)]
struct SideDef {
  // TODO: Sector facing: https://doomwiki.org/wiki/Sidedef
  x_offset: i16,
  y_offset: i16,
  name_of_upper_texture: String,
  name_of_lower_texture: String,
  name_of_middle_texture: String,
  sector_facing: usize, // TODO
}

fn load_maps(wad_file: &Vec<u8>, lumps: Vec<Lump>) -> Vec<Map> {
  let map_name_pattern = Regex::new("E[0-9]+M[0-9]+").unwrap();

  let mut maps = Vec::new();
  let mut current_map_name: Option<String> = None;
  let mut current_map_vertexes = Vec::new();
  let mut current_map_linedefs = Vec::new();
  let mut current_map_sidedefs = Vec::new();

  for lump in &lumps {
    if map_name_pattern.is_match(&lump.name) {
      if current_map_name.is_some() {
        maps.push(Map {
          name: current_map_name.unwrap(),
          vertexes: current_map_vertexes.to_owned(),
          linedefs: current_map_linedefs.to_owned(),
        })
      }

      current_map_name = Some(lump.name.clone());
    }

    if lump.name == "VERTEXES" {
      let mut vertex_i = lump.filepos;
      let vertex_count = lump.size / 4; // each vertex is 4 bytes, 2x 16-bit (or 2 byte) signed integers

      // current_map_vertexes = vec![];

      for _ in 0..vertex_count {
        let x = i16::from_le_bytes([wad_file[vertex_i], wad_file[vertex_i + 1]]);
        let y = i16::from_le_bytes([wad_file[vertex_i + 2], wad_file[vertex_i + 3]]);

        current_map_vertexes.push(MapVertex { x: x, y: y });

        vertex_i += 4;
      }
    }

    if lump.name == "LINEDEFS" {
      let mut line_i = lump.filepos;
      let line_count = lump.size / 14; // each line is 14 bytes, 7x 16-bit (or 2 byte) signed integers

      // current_map_linedefs = vec![];

      for _ in 0..line_count {
        let start_vertex = i16::from_le_bytes([wad_file[line_i], wad_file[line_i + 1]]);
        let end_vertex = i16::from_le_bytes([wad_file[line_i + 2], wad_file[line_i + 3]]);
        let front_sidedef = i16::from_le_bytes([wad_file[line_i + 10], wad_file[line_i + 11]]);
        let back_sidedef = i16::from_le_bytes([wad_file[line_i + 12], wad_file[line_i + 13]]);

        current_map_linedefs.push(LineDef {
          start_vertex: start_vertex as usize,
          end_vertex: end_vertex as usize,
          front_sidedef: front_sidedef as usize,
          back_sidedef: back_sidedef as usize,
        });

        line_i += 14;
      }
    }

    if lump.name == "SIDEDEFS" {
      let mut sidedef_i = lump.filepos;
      let sidedef_count = lump.size / 30; // each sidedef is 30 bytes

      // current_map_sidedefs = vec![];

      for _ in 0..sidedef_count {
        let x_offset = i16::from_le_bytes([wad_file[sidedef_i], wad_file[sidedef_i + 1]]);
        let y_offset = i16::from_le_bytes([wad_file[sidedef_i + 2], wad_file[sidedef_i + 3]]);

        let name_of_upper_texture: String = format!(
          "{}{}{}{}{}{}{}{}",
          wad_file[sidedef_i + 4] as char,
          wad_file[sidedef_i + 5] as char,
          wad_file[sidedef_i + 6] as char,
          wad_file[sidedef_i + 7] as char,
          wad_file[sidedef_i + 8] as char,
          wad_file[sidedef_i + 9] as char,
          wad_file[sidedef_i + 10] as char,
          wad_file[sidedef_i + 11] as char,
        );

        let name_of_lower_texture: String = format!(
          "{}{}{}{}{}{}{}{}",
          wad_file[sidedef_i + 12] as char,
          wad_file[sidedef_i + 13] as char,
          wad_file[sidedef_i + 14] as char,
          wad_file[sidedef_i + 15] as char,
          wad_file[sidedef_i + 16] as char,
          wad_file[sidedef_i + 17] as char,
          wad_file[sidedef_i + 18] as char,
          wad_file[sidedef_i + 19] as char,
        );

        let name_of_middle_texture: String = format!(
          "{}{}{}{}{}{}{}{}",
          wad_file[sidedef_i + 20] as char,
          wad_file[sidedef_i + 21] as char,
          wad_file[sidedef_i + 22] as char,
          wad_file[sidedef_i + 23] as char,
          wad_file[sidedef_i + 24] as char,
          wad_file[sidedef_i + 25] as char,
          wad_file[sidedef_i + 26] as char,
          wad_file[sidedef_i + 27] as char,
        );

        current_map_sidedefs.push(SideDef {
          x_offset: x_offset,
          y_offset: y_offset,
          name_of_upper_texture: name_of_upper_texture,
          name_of_lower_texture: name_of_lower_texture,
          name_of_middle_texture: name_of_middle_texture,
          sector_facing: 0,
        });

        sidedef_i += 30;
      }
    }
  }

  maps
}

#[derive(Debug)]
struct Lump {
  filepos: usize,
  size: usize,
  name: String,
}

fn load_lumps(wad_file: &Vec<u8>) -> Vec<Lump> {
  println!("Read WAD. File size in bytes: {}", wad_file.len());

  let wad_type: String = format!(
    "{}{}{}{}",
    wad_file[0] as char, wad_file[1] as char, wad_file[2] as char, wad_file[3] as char
  );
  println!("WAD type: {}", wad_type);

  let lump_num = u32::from_le_bytes([wad_file[4], wad_file[5], wad_file[6], wad_file[7]]);

  let directory_offset = u32::from_le_bytes([wad_file[8], wad_file[9], wad_file[10], wad_file[11]]);

  let mut current_lump_offset = directory_offset as usize;
  let mut lumps = Vec::new();

  for _ in 0..lump_num {
    let filepos = u32::from_le_bytes([
      wad_file[current_lump_offset],
      wad_file[current_lump_offset + 1],
      wad_file[current_lump_offset + 2],
      wad_file[current_lump_offset + 3],
    ]);

    let size = u32::from_le_bytes([
      wad_file[current_lump_offset + 4],
      wad_file[current_lump_offset + 5],
      wad_file[current_lump_offset + 6],
      wad_file[current_lump_offset + 7],
    ]);

    let lump_name: String = format!(
      "{}{}{}{}{}{}{}{}",
      wad_file[current_lump_offset + 8] as char,
      wad_file[current_lump_offset + 9] as char,
      wad_file[current_lump_offset + 10] as char,
      wad_file[current_lump_offset + 11] as char,
      wad_file[current_lump_offset + 12] as char,
      wad_file[current_lump_offset + 13] as char,
      wad_file[current_lump_offset + 14] as char,
      wad_file[current_lump_offset + 15] as char,
    );

    lumps.push(Lump {
      filepos: filepos as usize,
      size: size as usize,
      name: lump_name,
    });

    current_lump_offset += 16;
  }

  for lump in &lumps {
    println!("{:?}", lump);
  }

  lumps
}

fn render_scene() {
  #[allow(unused_imports)]
  use glium::{glutin, Surface};

  let event_loop = glutin::event_loop::EventLoop::new();
  let wb = glutin::window::WindowBuilder::new();
  let cb = glutin::ContextBuilder::new().with_depth_buffer(24);
  let display = glium::Display::new(wb, cb, &event_loop).unwrap();

  #[derive(Copy, Clone)]
  struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    tex_coords: [f32; 2],
  }

  implement_vertex!(Vertex, position, normal, tex_coords);

  let shape = glium::vertex::VertexBuffer::new(
    &display,
    &[
      Vertex {
        position: [-1.0, 1.0, 0.0],
        normal: [0.0, 0.0, -1.0],
        tex_coords: [0.0, 1.0],
      },
      Vertex {
        position: [1.0, 1.0, 0.0],
        normal: [0.0, 0.0, -1.0],
        tex_coords: [1.0, 1.0],
      },
      Vertex {
        position: [-1.0, -1.0, 0.0],
        normal: [0.0, 0.0, -1.0],
        tex_coords: [0.0, 0.0],
      },
      Vertex {
        position: [1.0, -1.0, 0.0],
        normal: [0.0, 0.0, -1.0],
        tex_coords: [1.0, 0.0],
      },
    ],
  )
  .unwrap();

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

  let program =
    glium::Program::from_source(&display, vertex_shader_src, fragment_shader_src, None).unwrap();

  event_loop.run(move |event, _, control_flow| {
    let next_frame_time = std::time::Instant::now() + std::time::Duration::from_nanos(16_666_667);
    *control_flow = glutin::event_loop::ControlFlow::WaitUntil(next_frame_time);

    match event {
      glutin::event::Event::WindowEvent { event, .. } => match event {
        glutin::event::WindowEvent::CloseRequested => {
          *control_flow = glutin::event_loop::ControlFlow::Exit;
          return;
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

    let view = view_matrix(&[0.5, 0.2, -3.0], &[-0.5, -0.2, 3.0], &[0.0, 1.0, 0.0]);

    let perspective = {
      let (width, height) = target.get_dimensions();
      let aspect_ratio = height as f32 / width as f32;

      let fov: f32 = 3.141592 / 3.0;
      let zfar = 1024.0;
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

    target
      .draw(
        &shape,
        glium::index::NoIndices(glium::index::PrimitiveType::TriangleStrip),
        &program,
        &uniform! { model: model, view: view, perspective: perspective,
        u_light: light, diffuse_tex: &diffuse_texture, normal_tex: &normal_map },
        &params,
      )
      .unwrap();
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
