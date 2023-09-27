use std::path::PathBuf;

fn main() {
    let frida_root = std::env::var("FRIDA_ROOT")
        .unwrap_or("../../build/FridaCore-prefix/src/FridaCore".to_string());

    println!("cargo:rustc-link-search={}", frida_root);

    // Tell cargo to tell rustc to link the system bzip2
    // shared library.
    println!("cargo:rustc-link-lib=static=frida-core");

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed={}/frida-core.h", frida_root);
    let bindings = bindgen::Builder::default()
        .header(format!("{}/frida-core.h", frida_root))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
