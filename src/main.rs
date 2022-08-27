use doomlib::game::scene::Scene;
use doomlib::run;

fn main() {
    let scene = Scene::load("doom.wad", "^E1M1$");

    pollster::block_on(run(scene));
}
