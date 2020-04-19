pub use super::Lump;

pub fn load(wad_file: &Vec<u8>) -> Vec<Lump> {
  println!("Read WAD. File size in bytes: {}", wad_file.len());

  let _wad_type: String = format!(
    "{}{}{}{}",
    wad_file[0] as char, wad_file[1] as char, wad_file[2] as char, wad_file[3] as char
  );

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
    )
    .trim_matches(char::from(0))
    .to_owned();

    lumps.push(Lump {
      filepos: filepos as usize,
      size: size as usize,
      name: lump_name,
    });

    current_lump_offset += 16;
  }

  // for lump in &lumps {
  //   println!("{:?}", lump);
  // }

  lumps
}
