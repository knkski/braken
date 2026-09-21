#[cfg(target_arch = "wasm32")]
fn main() {
    braken_web::run();
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    eprintln!("braken-web is a browser application; use Trunk to run it");
}
