use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(rusty_metal)");

    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set"));
    let obj = out_dir.join("metal_backend.o");
    let lib = out_dir.join("librusty_metal_backend.a");

    let clang_status = Command::new("xcrun")
        .args([
            "clang",
            "-x",
            "objective-c",
            "-fobjc-arc",
            "-O3",
            "-c",
            "src/metal_backend.m",
            "-o",
        ])
        .arg(&obj)
        .status();

    let Ok(status) = clang_status else {
        println!("cargo:warning=Metal backend disabled: xcrun clang was not available");
        return;
    };
    if !status.success() {
        println!("cargo:warning=Metal backend disabled: Objective-C shim did not compile");
        return;
    }

    let ar_status = Command::new("ar").arg("crs").arg(&lib).arg(&obj).status();
    if !matches!(ar_status, Ok(status) if status.success()) {
        println!("cargo:warning=Metal backend disabled: static library creation failed");
        return;
    }

    println!("cargo:rustc-cfg=rusty_metal");
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=rusty_metal_backend");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=CoreGraphics");
    println!("cargo:rerun-if-changed=src/metal_backend.m");
}
