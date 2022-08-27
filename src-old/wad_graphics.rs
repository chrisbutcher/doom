pub use super::*;

use image::imageops::FilterType;
use image::DynamicImage;

#[derive(Debug, Clone)]
pub struct Colormap {
  pub palette_indexes: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct PaletteColor {
  pub r: u8,
  pub g: u8,
  pub b: u8,
}

#[derive(Debug, Clone)]
pub struct Flat {
  pub pixels: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct Picture {
  pub width: u16,
  pub height: u16,
  pub leftoffset: u16,
  pub topoffset: u16,
  pub posts: Vec<PicturePost>,
  pub lump_name: String,
}

#[derive(Debug, Clone)]
pub struct PictureSpan {
  pub topdelta: usize,
  pub length: u8,
  pub pixels: Vec<usize>,
  pub last_span_pixel_count: usize,
  pub last_span_topdelta: usize,
}

#[derive(Debug, Clone)]
pub struct PicturePost {
  pub pixel_spans: Vec<PictureSpan>,
}

#[derive(Debug, Clone)]
pub struct Patch {
  pub name: String,
}

#[derive(Debug, Clone)]
pub struct WallPatch {
  pub originx: i16,
  pub originy: i16,
  pub patch_number: usize,
}

#[derive(Debug, Clone)]
pub struct WallTexture {
  pub name: String,
  pub masked: bool,
  pub width: i16,
  pub height: i16,
  pub patches: Vec<WallPatch>,
}

pub fn texture_to_gl_texture(scene: &Scene, texture_name: &str) -> (image::DynamicImage, (i16, i16)) {
  println!("[texture_to_gl_texture] Loading texture: {}", texture_name);

  let texture = scene.textures.iter().find(|&t| t.name == texture_name).unwrap();
  let mut imgbuf = image::ImageBuffer::<image::Rgba<u8>, Vec<u8>>::new(texture.width as u32, texture.height as u32);

  for patch in &texture.patches {
    let fetched_patch = &scene.patch_names[patch.patch_number];

    let patch_picture = wad_graphics::load_picture_from_wad(&scene.wad_file, &scene.lumps, &fetched_patch.name);

    let patch_bytes = wad_graphics::picture_to_rgba_bytes(&patch_picture, &scene.palette);

    let mut patch_x: i32 = 0;
    let mut patch_y: i32 = 0;

    for raw_pixel in patch_bytes.chunks(4) {
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

  let image = DynamicImage::ImageRgba8(imgbuf);
  image.resize(texture.width as u32, texture.height as u32, FilterType::Nearest);

  (image, (texture.width, texture.height))
}

pub fn picture_to_rgba_bytes(picture: &Picture, palette: &std::vec::Vec<PaletteColor>) -> Vec<u8> {
  let mut data = vec![0u8; picture.width as usize * picture.height as usize * 4];
  for (i, post) in picture.posts.iter().enumerate() {
    let mut y = 0;

    for span in &post.pixel_spans {
      for _ in 0..span.topdelta - span.last_span_pixel_count - span.last_span_topdelta {
        let index = (i + y * picture.width as usize) * 4;

        data[index] = 0;
        data[index + 1] = 0;
        data[index + 2] = 0;
        data[index + 3] = 0; // alpha

        y += 1;
      }
      for pixel_addr in &span.pixels {
        let palette_color = &palette[*pixel_addr];

        let index = (i + y * picture.width as usize) * 4;

        data[index] = palette_color.r;
        data[index + 1] = palette_color.g;
        data[index + 2] = palette_color.b;
        data[index + 3] = 0xFF; // alpha

        y += 1;
      }
    }
    let pixels_to_write = picture.height as usize - y;

    if pixels_to_write > 0 {
      for _ in 0..pixels_to_write {
        let index = (i + y * picture.width as usize) * 4;
        data[index] = 0;
        data[index + 1] = 0;
        data[index + 2] = 0;
        data[index + 3] = 0; // alpha

        y += 1;
      }
    }
  }

  data
}

pub fn load_picture_from_wad(wad_file: &Vec<u8>, lumps: &Vec<lumps::Lump>, lump_name: &str) -> Picture {
  let picture_lump = lumps.iter().find(|&l| l.name == lump_name.to_uppercase()).unwrap();

  let lump_offset = picture_lump.filepos;

  let width = u16::from_le_bytes([wad_file[lump_offset], wad_file[lump_offset + 1]]);
  let height = u16::from_le_bytes([wad_file[lump_offset + 2], wad_file[lump_offset + 3]]);
  let leftoffset = u16::from_le_bytes([wad_file[lump_offset + 4], wad_file[lump_offset + 5]]);
  let topoffset = u16::from_le_bytes([wad_file[lump_offset + 6], wad_file[lump_offset + 7]]);

  let mut column_array = Vec::new();
  for i in 0..width as usize {
    let column_offset = u32::from_le_bytes([
      wad_file[lump_offset + (i * 4) + 8],
      wad_file[lump_offset + (i * 4) + 9],
      wad_file[lump_offset + (i * 4) + 10],
      wad_file[lump_offset + (i * 4) + 11],
    ]) as usize;

    column_array.push(column_offset);
  }

  let mut posts = Vec::new();
  let mut post_spans = Vec::new();
  for i in 0..width as usize {
    let mut filepos_with_offset = lump_offset + &column_array[i];
    let mut last_span_pixel_count = 0;
    let mut last_span_topdelta = 0;
    let mut finished_post = false;

    while finished_post == false {
      let topdelta = wad_file[filepos_with_offset] as usize;

      let pixel_count = wad_file[filepos_with_offset + 1] as usize;
      let mut span_pixels = Vec::with_capacity(pixel_count);

      let _dummy_value = wad_file[filepos_with_offset + 2];

      for j in 0..pixel_count as usize {
        let pixel_addr = filepos_with_offset + 3 + j;
        let pixel_palette_addr = wad_file[pixel_addr] as usize;
        span_pixels.push(pixel_palette_addr);
      }

      post_spans.push(PictureSpan {
        topdelta: topdelta,
        length: pixel_count as u8,
        pixels: span_pixels.to_owned(),
        last_span_pixel_count: last_span_pixel_count,
        last_span_topdelta: last_span_topdelta,
      });

      last_span_pixel_count = pixel_count;
      last_span_topdelta = topdelta;

      span_pixels.clear();

      let second_dummy_value_addr = filepos_with_offset + 3 + pixel_count as usize;

      filepos_with_offset = second_dummy_value_addr + 1;

      let end_of_post_or_start_of_next_span = wad_file[filepos_with_offset];

      if end_of_post_or_start_of_next_span == 255 {
        finished_post = true;
      }
    }

    let new_post = PicturePost {
      pixel_spans: post_spans.to_owned(),
    };
    post_spans.clear();

    posts.push(new_post);
  }

  Picture {
    width: width,
    height: height,
    leftoffset: leftoffset,
    topoffset: topoffset,
    posts: posts,
    lump_name: lump_name.to_string(),
  }
}

pub fn load_flat_from_wad(wad_file: &Vec<u8>, lumps: &Vec<lumps::Lump>, lump_name: &str) -> Flat {
  let flat_lump = lumps.iter().find(|&l| l.name == lump_name).unwrap();

  let mut pixels = Vec::with_capacity(4096);

  for i in 0..4096 {
    let pixel = u8::from_le_bytes([wad_file[flat_lump.filepos + i]]) as usize;
    pixels.push(pixel);
  }

  Flat { pixels: pixels }
}

pub fn load_textures(wad_file: &Vec<u8>, lumps: &Vec<lumps::Lump>) -> Vec<WallTexture> {
  let mut wall_textures = Vec::new();

  let texture_lumps: Vec<&lumps::Lump> = lumps
    .iter()
    .filter(|&l| l.name.starts_with("TEXTURE"))
    .collect::<Vec<&lumps::Lump>>();

  for texture_lump in texture_lumps {
    let lump_offset = texture_lump.filepos;

    let num_textures = i32::from_le_bytes([
      wad_file[lump_offset],
      wad_file[lump_offset + 1],
      wad_file[lump_offset + 2],
      wad_file[lump_offset + 3],
    ]) as usize;

    let mut texture_offsets = Vec::new();

    for i in 0..num_textures {
      let texture_struct_offset = i32::from_le_bytes([
        wad_file[lump_offset + (i * 4) + 4],
        wad_file[lump_offset + (i * 4) + 5],
        wad_file[lump_offset + (i * 4) + 6],
        wad_file[lump_offset + (i * 4) + 7],
      ]) as usize;

      texture_offsets.push(texture_struct_offset);
    }

    for this_texture_offset in texture_offsets {
      let offset = lump_offset + this_texture_offset;

      let texture_name: String = format!(
        "{}{}{}{}{}{}{}{}",
        wad_file[offset] as char,
        wad_file[offset + 1] as char,
        wad_file[offset + 2] as char,
        wad_file[offset + 3] as char,
        wad_file[offset + 4] as char,
        wad_file[offset + 5] as char,
        wad_file[offset + 6] as char,
        wad_file[offset + 7] as char,
      )
      .trim_matches(char::from(0))
      .to_owned();

      // TODO: Convert to boolean
      let masked = i32::from_le_bytes([
        wad_file[offset + 8],
        wad_file[offset + 9],
        wad_file[offset + 10],
        wad_file[offset + 11],
      ]);

      let width = i16::from_le_bytes([wad_file[offset + 12], wad_file[offset + 13]]);
      let height = i16::from_le_bytes([wad_file[offset + 14], wad_file[offset + 15]]);

      let _column_directory = i32::from_le_bytes([
        wad_file[offset + 16],
        wad_file[offset + 17],
        wad_file[offset + 18],
        wad_file[offset + 19],
      ]);

      let patch_count = i16::from_le_bytes([wad_file[offset + 20], wad_file[offset + 21]]);

      let mut patches = Vec::new();

      let mut patch_index = 0;
      for _i in 0..patch_count {
        let originx = i16::from_le_bytes([wad_file[offset + patch_index + 22], wad_file[offset + patch_index + 23]]);

        let originy = i16::from_le_bytes([wad_file[offset + patch_index + 24], wad_file[offset + patch_index + 25]]);

        let patch =
          i16::from_le_bytes([wad_file[offset + patch_index + 26], wad_file[offset + patch_index + 27]]) as usize;

        patches.push(WallPatch {
          originx: originx,
          originy: originy,
          patch_number: patch,
        });

        patch_index += 10;
      }

      wall_textures.push(WallTexture {
        name: texture_name,
        masked: masked != 0,
        width: width,
        height: height,
        patches: patches,
      });
    }
  }

  wall_textures
}

pub fn load_patch_names(wad_file: &Vec<u8>, lumps: &Vec<lumps::Lump>) -> Vec<Patch> {
  let mut patches = Vec::new();

  let pnames_lump = lumps.iter().find(|&l| l.name.starts_with("PNAMES")).unwrap();

  let num_patches = i32::from_le_bytes([
    wad_file[pnames_lump.filepos],
    wad_file[pnames_lump.filepos + 1],
    wad_file[pnames_lump.filepos + 2],
    wad_file[pnames_lump.filepos + 3],
  ]) as usize;

  let patches_starting_offset = pnames_lump.filepos + 4;

  for i in 0..num_patches {
    let patch_name: String = format!(
      "{}{}{}{}{}{}{}{}",
      wad_file[patches_starting_offset + (i * 8)] as char,
      wad_file[patches_starting_offset + (i * 8) + 1] as char,
      wad_file[patches_starting_offset + (i * 8) + 2] as char,
      wad_file[patches_starting_offset + (i * 8) + 3] as char,
      wad_file[patches_starting_offset + (i * 8) + 4] as char,
      wad_file[patches_starting_offset + (i * 8) + 5] as char,
      wad_file[patches_starting_offset + (i * 8) + 6] as char,
      wad_file[patches_starting_offset + (i * 8) + 7] as char,
    )
    .trim_matches(char::from(0))
    .to_owned();

    patches.push(Patch { name: patch_name });
  }

  patches
}
