use std::fs;
use serde::Deserialize;

#[derive(Deserialize)]
struct Config {
    paths: Paths,
}

#[derive(Deserialize)]
struct Paths {
    python_lib_path: String,
    python_lib_name: String,
}

fn main() {
    // Read the configuration file
    let config_contents = fs::read_to_string("config.yaml").expect("Failed to read config.yaml");
    let config: Config = serde_yaml::from_str(&config_contents).expect("Failed to parse config.yaml");

    // Extract the paths from the configuration
    let lib_path = config.paths.python_lib_path;
    let lib_name = config.paths.python_lib_name;

    // Remove the "lib" prefix and ".so" suffix for the linker
    let lib_name_stripped = lib_name.trim_start_matches("lib").trim_end_matches(".so");

    // Print the cargo instructions to link the library
    println!("cargo:rustc-link-search=native={}", lib_path);
    println!("cargo:rustc-link-lib={}", lib_name_stripped);
}
