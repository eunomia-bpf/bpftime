fn main() {
    let _ = cxx_build::bridges(["src/lib.rs", "src/to_cxx.rs"])
        .include("cxx")
        .file("cxx/to_cxx.cc")
        .compile("patcher");

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/to_cxx.rs");
    println!("cargo:rerun-if-changed=cxx/to_cxx.h");
    println!("cargo:rerun-if-changed=cxx/to_cxx.cc");
}
