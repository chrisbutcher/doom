pub use super::*;

pub fn load_picture_from_wad_2(wad_file: &Vec<u8>, lumps: &Vec<Lump>, lump_name: &str) {
  let picture_lump = lumps.iter().find(|&l| l.name == lump_name).unwrap();

  let lump_offset = picture_lump.filepos;

  let width = u16::from_le_bytes([wad_file[lump_offset], wad_file[lump_offset + 1]]);
  let height = u16::from_le_bytes([wad_file[lump_offset + 2], wad_file[lump_offset + 3]]);
  let leftoffset = u16::from_le_bytes([wad_file[lump_offset + 4], wad_file[lump_offset + 5]]);
  let topoffset = u16::from_le_bytes([wad_file[lump_offset + 6], wad_file[lump_offset + 7]]);

  // aka posts
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

  for i in 0..width as usize {
    let mut filepos_with_offset = lump_offset + &column_array[i];

    let mut topdelta = 0;

    while topdelta != 255 {
      let topdelta = wad_file[filepos_with_offset];

      if topdelta == 255 {
        break;
      }

      let pixel_count = wad_file[filepos_with_offset + 1];
      let _dummy_value = wad_file[filepos_with_offset + 2];

      for j in 0..pixel_count as usize {
        let pixel = wad_file[filepos_with_offset + 3 + j];
      }
    }

    // println!("{}", &column_array[i]);
  }

  panic!("boom!");

  // Byte = 0 - 255
  // Word = 0 - 65535
  // DWord = 0 - 4294967295

  // dummy_value = 	Byte, those unused bytes in the file (excerpt from UDS: "..left overs from NeXT machines?..")
  // picture_* = 	Word, the maximum width for an image in doom picture format is 256 pixels
  // pixel_count = 	Byte, the number of pixels in a post
  // Pixel = 	Byte, the pixel colour
  // column_array =	array of DWord, this holds all the post start offsets for each column

  // doom image = could be a file or memory stream

  // Algorithm:
  // ----------

  // create a image with a pixel format of 8bit and the doom palette, set the background colour to a contrasting colour (cyan).

  // read width from doom image (word)
  // read height from doom image (word)
  // read left from doom image (word)
  // read top from doom image (word)

  // create column_array with width number of elements

  // for loop, i = 0, break on i = width - 1
  // 	column_array[i] = read from doom image, 4 bytes
  // end block

  // for loop, i = 0, break on i = width - 1
  // 	seek doom image to column_array[i] from beginning of doom image

  // 	rowstart = 0

  // 	while loop, rowstart != 255
  // 		read rowstart from doom image, 1 byte

  // 		if rowstart = 255, break from this loop

  // 		read pixel_count from doom image, 1 byte

  // 		read dummy_value from doom image, 1 byte

  // 		for loop, j = 0, break on j = pixel_count - 1
  // 			read Pixel from doom image, 1 byte

  // 			write Pixel to image, j + rowstart = row, i = column
  // 		end block

  // 		read dummy_value from doom image, 1 byte
  // 	end block
  // end block
}

pub fn load_picture_from_wad(wad_file: &Vec<u8>, lumps: &Vec<Lump>, lump_name: &str) -> Picture {
  let picture_lump = lumps.iter().find(|&l| l.name == lump_name).unwrap();

  let lump_offset = picture_lump.filepos;

  let width = u16::from_le_bytes([wad_file[lump_offset], wad_file[lump_offset + 1]]);
  let height = u16::from_le_bytes([wad_file[lump_offset + 2], wad_file[lump_offset + 3]]);
  let leftoffset = u16::from_le_bytes([wad_file[lump_offset + 4], wad_file[lump_offset + 5]]);
  let topoffset = u16::from_le_bytes([wad_file[lump_offset + 6], wad_file[lump_offset + 7]]);

  let mut posts = Vec::new();

  let mut post_column_i = 0;
  for _ in 0..width {
    let post_column_offset = u32::from_le_bytes([
      wad_file[lump_offset + (post_column_i * 4) + 8],
      wad_file[lump_offset + (post_column_i * 4) + 9],
      wad_file[lump_offset + (post_column_i * 4) + 10],
      wad_file[lump_offset + (post_column_i * 4) + 11],
    ]) as usize;

    let topdelta = u8::from_le_bytes([wad_file[lump_offset + post_column_offset]]);
    let length = u8::from_le_bytes([wad_file[lump_offset + post_column_offset + 1]]);
    let _unused = u8::from_le_bytes([wad_file[lump_offset + post_column_offset + 2]]);

    let pixels_offset = lump_offset + post_column_offset;

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

pub fn load_flat_from_wad(wad_file: &Vec<u8>, lumps: &Vec<Lump>, lump_name: &str) -> Flat {
  let flat_lump = lumps.iter().find(|&l| l.name == lump_name).unwrap();

  let mut pixels = Vec::with_capacity(4096);

  for i in 0..4096 {
    let pixel = u8::from_le_bytes([wad_file[flat_lump.filepos + i]]) as usize;
    pixels.push(pixel);
  }

  Flat { pixels: pixels }
}

pub fn load_textures(wad_file: &Vec<u8>, lumps: &Vec<Lump>) -> Vec<WallTexture> {
  let mut wall_textures = Vec::new();

  let texture_lumps: Vec<&Lump> = lumps
    .iter()
    .filter(|&l| l.name.starts_with("TEXTURE"))
    .collect::<Vec<&Lump>>();

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

pub fn load_patch_names(wad_file: &Vec<u8>, lumps: &Vec<Lump>) -> Vec<Patch> {
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
