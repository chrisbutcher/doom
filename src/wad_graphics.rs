pub use super::{Lump, Patch, WallPatch, WallTexture};

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
    println!("patches_starting_offset: {}", patches_starting_offset + (i * 8));
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

    println!("{}", patch_name);

    patches.push(Patch { name: patch_name });
  }

  patches
}