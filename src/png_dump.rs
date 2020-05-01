pub use super::*;
use std::io::BufWriter;
use std::path::Path as std_path;

pub fn dump_picture(picture: &Picture, palette: &std::vec::Vec<PaletteColor>) {
  let output_filename = format!("{}.png", picture.lump_name);
  let path = std_path::new(&output_filename);
  let file = File::create(path).unwrap();
  let ref mut w = BufWriter::new(file);

  let mut encoder = png::Encoder::new(w, picture.width as u32, picture.height as u32); // Width is 2 pixels and height is 1.
  encoder.set_color(png::ColorType::RGBA);
  encoder.set_depth(png::BitDepth::Eight);
  let mut writer = encoder.write_header().unwrap();

  let mut data = vec![0u8; picture.width as usize * picture.height as usize * 4];
  for (i, post) in picture.posts.iter().enumerate() {
    let mut y = 0;

    for span in &post.pixel_spans {
      for _ in 0..span.blank_vertical_space_preceding {
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

  writer.write_image_data(&data).unwrap(); // Save

  // png end
}
