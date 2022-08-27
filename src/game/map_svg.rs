use super::*;
use crate::model::ModelVertex;

use std::cmp;
use std::collections::HashMap;
use svg::node::element::path::Data;
use svg::node::element::Path;
use svg::Document;

// ORIGIN is top-left. y axis grows downward (as in, subtract to go up).

#[derive(Debug, Copy, Clone)]
pub struct MapCenterer {
    pub left_most_x: i16,
    pub right_most_x: i16,
    pub lower_most_y: i16,
    pub upper_most_y: i16,
}

impl MapCenterer {
    pub fn new() -> MapCenterer {
        MapCenterer {
            left_most_x: i16::max_value(),
            right_most_x: i16::min_value(),
            lower_most_y: i16::max_value(),
            upper_most_y: i16::min_value(),
        }
    }

    pub fn record_x(&mut self, x: i16) {
        self.left_most_x = cmp::min(self.left_most_x, x);
        self.right_most_x = cmp::max(self.right_most_x, x);
    }

    pub fn record_y(&mut self, y: i16) {
        self.lower_most_y = cmp::min(self.lower_most_y, y);
        self.upper_most_y = cmp::max(self.upper_most_y, y);
    }
}

pub fn draw_map_svg_with_floors(
    sector_id_to_floor_vertices_with_indices: HashMap<usize, (Vec<ModelVertex>, Vec<u32>)>,
    map: &maps::Map,
) {
    let mut document = Document::new();

    let map_x_offset = 0 - map.map_centerer.left_most_x;
    let map_y_offset = 0 - map.map_centerer.upper_most_y;

    for line in &map.linedefs {
        let v1_index = line.start_vertex;
        let v2_index = line.end_vertex;

        let v1 = &map.vertexes[v1_index];
        let v2 = &map.vertexes[v2_index];

        let v1_x = v1.x + map_x_offset;
        let v2_x = v2.x + map_x_offset;
        let v1_y = v1.y + map_y_offset;
        let v2_y = v2.y + map_y_offset;

        let path = Path::new()
            .set("fill", "none")
            .set("stroke", "black")
            .set("stroke-width", 10)
            .set(
                "d",
                Data::new()
                    .move_to((v1_x, -v1_y)) // flipping y axis at the last moment to account for SVG convention
                    .line_to((v2_x, -v2_y))
                    .close(),
            );

        document = document.clone().add(path);
    }

    let mut vertex_pairs: Vec<[ModelVertex; 2]> = Vec::new();

    for (sector_id, vertices_with_indices) in sector_id_to_floor_vertices_with_indices {
        let verts = vertices_with_indices.0.clone();
        let inds = vertices_with_indices.1.clone();

        println!("Drawing debug SVG vertices for sector: {}", sector_id);
        println!("verts.len(): {}, inds.len(): {}", verts.len(), inds.len());

        println!("{:?}", vertices_with_indices);

        for ind_chunk in inds.chunks(3) {
            let v0 = verts[ind_chunk[0] as usize];
            let v1 = verts[ind_chunk[1] as usize];
            let v2 = verts[ind_chunk[2] as usize];

            vertex_pairs.push([v0, v1]);
            vertex_pairs.push([v1, v2]);
            vertex_pairs.push([v2, v0]);
        }
    }

    for pair in &vertex_pairs {
        let v1 = pair[0];
        let v2 = pair[1];

        let v1_x = v1.position[0] as i16 + map_x_offset;
        let v2_x = v2.position[0] as i16 + map_x_offset;
        let v1_y = -v1.position[2] as i16 + map_y_offset;
        let v2_y = -v2.position[2] as i16 + map_y_offset;

        let path = Path::new()
            .set("fill", "none")
            .set("stroke", "red")
            .set("stroke-width", 6)
            .set(
                "d",
                Data::new()
                    .move_to((v1_x, -v1_y)) // flipping y axis at the last moment to account for SVG convention
                    .line_to((v2_x, -v2_y))
                    .close(),
            );

        document = document.clone().add(path);
    }

    let filename = format!(
        "{}{}{}{}.svg",
        map.name.chars().nth(0).unwrap(),
        map.name.chars().nth(1).unwrap(),
        map.name.chars().nth(2).unwrap(),
        map.name.chars().nth(3).unwrap(),
    );

    let width = map.map_centerer.right_most_x - map.map_centerer.left_most_x;
    let height = map.map_centerer.upper_most_y - map.map_centerer.lower_most_y;
    document = document
        .clone()
        .set("viewBox", (-10, -10, width as i32 * 5, height as i32 * 5))
        .set("width", width)
        .set("height", height);
    svg::save(filename.trim(), &document).unwrap();
}

pub fn draw_map_svg(map: &maps::Map) {
    let mut document = Document::new();

    let map_x_offset = 0 - map.map_centerer.left_most_x;
    let map_y_offset = 0 - map.map_centerer.upper_most_y;

    for line in &map.linedefs {
        let v1_index = line.start_vertex;
        let v2_index = line.end_vertex;

        let v1 = &map.vertexes[v1_index];
        let v2 = &map.vertexes[v2_index];

        let v1_x = v1.x + map_x_offset;
        let v2_x = v2.x + map_x_offset;
        let v1_y = v1.y + map_y_offset;
        let v2_y = v2.y + map_y_offset;

        let path = Path::new()
            .set("fill", "none")
            .set("stroke", "black")
            .set("stroke-width", 10)
            .set(
                "d",
                Data::new()
                    .move_to((v1_x, -v1_y)) // flipping y axis at the last moment to account for SVG convention
                    .line_to((v2_x, -v2_y))
                    .close(),
            );

        document = document.clone().add(path);
    }

    let filename = format!(
        "{}{}{}{} (without floors).svg",
        map.name.chars().nth(0).unwrap(),
        map.name.chars().nth(1).unwrap(),
        map.name.chars().nth(2).unwrap(),
        map.name.chars().nth(3).unwrap(),
    );

    let width = map.map_centerer.right_most_x - map.map_centerer.left_most_x;
    let height = map.map_centerer.upper_most_y - map.map_centerer.lower_most_y;
    document = document
        .clone()
        .set("viewBox", (-10, -10, width as i32 * 5, height as i32 * 5))
        .set("width", width)
        .set("height", height);
    svg::save(filename.trim(), &document).unwrap();
}
