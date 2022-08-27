pub use super::*;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path as std_path;

pub fn dump_picture(picture: &wad_graphics::Picture, palette: &std::vec::Vec<wad_graphics::PaletteColor>) {
    let output_filename = format!("{}.png", picture.lump_name);
    let path = std_path::new(&output_filename);
    let file = File::create(path).unwrap();
    let ref mut w = BufWriter::new(file);

    let mut encoder = png::Encoder::new(w, picture.width as u32, picture.height as u32); // Width is 2 pixels and height is 1.
    encoder.set_color(png::ColorType::RGBA);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().unwrap();

    let data = wad_graphics::picture_to_rgba_bytes(picture, palette);
    writer.write_image_data(&data).unwrap(); // Save
}
