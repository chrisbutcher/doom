pub use super::{map_svg, LineDef, Lump, Map, MapVertex, Sector, SideDef};
use regex::Regex;

pub fn load(map_name_pattern: &str, wad_file: &Vec<u8>, lumps: &Vec<Lump>) -> Map {
  let any_map_name_regex = Regex::new("E[0-9]+M[0-9]+").unwrap();
  let this_map_name_regex = Regex::new(map_name_pattern).unwrap();

  let mut current_map_name: Option<String> = None;
  let mut current_map_vertexes = Vec::new();
  let mut current_map_linedefs = Vec::new();
  let mut current_map_sidedefs = Vec::new();
  let mut current_map_sectors = Vec::new();
  let mut current_map_centerer = map_svg::MapCenterer::new();

  for lump in lumps {
    if this_map_name_regex.is_match(&lump.name) {
      current_map_name = Some(lump.name.clone());
    } else if any_map_name_regex.is_match(&lump.name) && current_map_name.is_some() {
      // If we've started another map, and we've found the one we're looking for, return early.
      return Map {
        name: current_map_name.unwrap(),
        vertexes: current_map_vertexes.to_owned(),
        linedefs: current_map_linedefs.to_owned(),
        sidedefs: current_map_sidedefs.to_owned(),
        map_centerer: current_map_centerer.to_owned(),
        sectors: current_map_sectors.to_owned(),
      };
    }

    if current_map_name.is_some() {
      if lump.name == "VERTEXES" {
        let mut vertex_i = lump.filepos;
        let vertex_count = lump.size / 4; // each vertex is 4 bytes, 2x 16-bit (or 2 byte) signed integers

        for _ in 0..vertex_count {
          let x = i16::from_le_bytes([wad_file[vertex_i], wad_file[vertex_i + 1]]);
          let y = i16::from_le_bytes([wad_file[vertex_i + 2], wad_file[vertex_i + 3]]);

          current_map_centerer.record_x(x);
          current_map_centerer.record_y(y);

          current_map_vertexes.push(MapVertex { x: x, y: y });

          vertex_i += 4;
        }
      }

      if lump.name == "LINEDEFS" {
        let mut line_i = lump.filepos;
        let line_count = lump.size / 14; // each line is 14 bytes, 7x 16-bit (or 2 byte) signed integers

        for _ in 0..line_count {
          let start_vertex = i16::from_le_bytes([wad_file[line_i], wad_file[line_i + 1]]);
          let end_vertex = i16::from_le_bytes([wad_file[line_i + 2], wad_file[line_i + 3]]);
          let front_sidedef_index =
            option_wrap_sidedef(i16::from_le_bytes([wad_file[line_i + 10], wad_file[line_i + 11]]));
          let back_sidedef_index =
            option_wrap_sidedef(i16::from_le_bytes([wad_file[line_i + 12], wad_file[line_i + 13]]));

          // NOTE: If front_sidedef_index or back_sidedef_index are std::usize::MAX, they are actually -1, meaning ignore them.

          current_map_linedefs.push(LineDef {
            start_vertex: start_vertex as usize,
            end_vertex: end_vertex as usize,
            front_sidedef_index: front_sidedef_index,
            back_sidedef_index: back_sidedef_index,
          });

          line_i += 14;
        }
      }

      if lump.name == "SIDEDEFS" {
        let mut sidedef_i = lump.filepos;
        let sidedef_count = lump.size / 30; // each sidedef is 30 bytes

        for _ in 0..sidedef_count {
          let x_offset = i16::from_le_bytes([wad_file[sidedef_i], wad_file[sidedef_i + 1]]);
          let y_offset = i16::from_le_bytes([wad_file[sidedef_i + 2], wad_file[sidedef_i + 3]]);

          let name_of_upper_texture = option_wrap_sidedef_texture_name(
            format!(
              "{}{}{}{}{}{}{}{}",
              wad_file[sidedef_i + 4] as char,
              wad_file[sidedef_i + 5] as char,
              wad_file[sidedef_i + 6] as char,
              wad_file[sidedef_i + 7] as char,
              wad_file[sidedef_i + 8] as char,
              wad_file[sidedef_i + 9] as char,
              wad_file[sidedef_i + 10] as char,
              wad_file[sidedef_i + 11] as char,
            )
            .trim_matches(char::from(0))
            .to_owned(),
          );

          let name_of_lower_texture = option_wrap_sidedef_texture_name(
            format!(
              "{}{}{}{}{}{}{}{}",
              wad_file[sidedef_i + 12] as char,
              wad_file[sidedef_i + 13] as char,
              wad_file[sidedef_i + 14] as char,
              wad_file[sidedef_i + 15] as char,
              wad_file[sidedef_i + 16] as char,
              wad_file[sidedef_i + 17] as char,
              wad_file[sidedef_i + 18] as char,
              wad_file[sidedef_i + 19] as char,
            )
            .trim_matches(char::from(0))
            .to_owned(),
          );

          let name_of_middle_texture = option_wrap_sidedef_texture_name(
            format!(
              "{}{}{}{}{}{}{}{}",
              wad_file[sidedef_i + 20] as char,
              wad_file[sidedef_i + 21] as char,
              wad_file[sidedef_i + 22] as char,
              wad_file[sidedef_i + 23] as char,
              wad_file[sidedef_i + 24] as char,
              wad_file[sidedef_i + 25] as char,
              wad_file[sidedef_i + 26] as char,
              wad_file[sidedef_i + 27] as char,
            )
            .trim_matches(char::from(0))
            .to_owned(),
          );

          let sector_facing = i16::from_le_bytes([wad_file[sidedef_i + 28], wad_file[sidedef_i + 29]]) as usize;

          current_map_sidedefs.push(SideDef {
            x_offset: x_offset,
            y_offset: y_offset,
            name_of_upper_texture: name_of_upper_texture, // TODO, sometimes return None, if texture is blank
            name_of_lower_texture: name_of_lower_texture,
            name_of_middle_texture: name_of_middle_texture,
            sector_facing: sector_facing,
          });

          sidedef_i += 30;
        }
      }

      if lump.name == "SECTORS" {
        let mut sector_i = lump.filepos;
        let sector_count = lump.size / 26; // each sector is 26 bytes

        for _ in 0..sector_count {
          let floor_height = i16::from_le_bytes([wad_file[sector_i], wad_file[sector_i + 1]]);
          let ceiling_height = i16::from_le_bytes([wad_file[sector_i + 2], wad_file[sector_i + 3]]);

          let name_of_floor_texture: String = format!(
            "{}{}{}{}{}{}{}{}",
            wad_file[sector_i + 4] as char,
            wad_file[sector_i + 5] as char,
            wad_file[sector_i + 6] as char,
            wad_file[sector_i + 7] as char,
            wad_file[sector_i + 8] as char,
            wad_file[sector_i + 9] as char,
            wad_file[sector_i + 10] as char,
            wad_file[sector_i + 11] as char,
          )
          .trim_matches(char::from(0))
          .to_owned();

          let name_of_ceiling_texture: String = format!(
            "{}{}{}{}{}{}{}{}",
            wad_file[sector_i + 12] as char,
            wad_file[sector_i + 13] as char,
            wad_file[sector_i + 14] as char,
            wad_file[sector_i + 15] as char,
            wad_file[sector_i + 16] as char,
            wad_file[sector_i + 17] as char,
            wad_file[sector_i + 18] as char,
            wad_file[sector_i + 19] as char,
          )
          .trim_matches(char::from(0))
          .to_owned();

          let light_level = i16::from_le_bytes([wad_file[sector_i + 20], wad_file[sector_i + 21]]);
          let sector_type = i16::from_le_bytes([wad_file[sector_i + 22], wad_file[sector_i + 23]]);
          let tag_number = i16::from_le_bytes([wad_file[sector_i + 24], wad_file[sector_i + 25]]);

          current_map_sectors.push(Sector {
            floor_height: floor_height,
            ceiling_height: ceiling_height,
            name_of_floor_texture: name_of_floor_texture,
            name_of_ceiling_texture: name_of_ceiling_texture,
            light_level: light_level,
            sector_type: sector_type,
            tag_number: tag_number,
          });

          sector_i += 26;
        }
      }
    }
  }
  // Or if it's the last map in the wad
  Map {
    name: current_map_name.unwrap(),
    vertexes: current_map_vertexes.to_owned(),
    linedefs: current_map_linedefs.to_owned(),
    sidedefs: current_map_sidedefs.to_owned(),
    map_centerer: current_map_centerer.to_owned(),
    sectors: current_map_sectors.to_owned(),
  }
}

fn option_wrap_sidedef(sidedef_index: i16) -> Option<usize> {
  if sidedef_index == -1 {
    None
  } else {
    Some(sidedef_index as usize)
  }
}

fn option_wrap_sidedef_texture_name(texture_name: String) -> Option<String> {
  if texture_name == "-" {
    None
  } else {
    Some(texture_name)
  }
}
