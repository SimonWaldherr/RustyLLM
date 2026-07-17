use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn macos_sdk_candidates() -> impl Iterator<Item = PathBuf> {
    let mut candidates = Vec::new();

    if let Some(sdkroot) = env::var_os("SDKROOT").filter(|value| !value.is_empty()) {
        candidates.push(PathBuf::from(sdkroot));
    }
    if let Some(developer_dir) = env::var_os("DEVELOPER_DIR").filter(|value| !value.is_empty()) {
        candidates.push(
            PathBuf::from(developer_dir)
                .join("Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"),
        );
    }
    candidates.push(
        PathBuf::from("/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"),
    );
    candidates.push(PathBuf::from(
        "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk",
    ));

    candidates.into_iter()
}

fn find_macos_sdk() -> Option<PathBuf> {
    macos_sdk_candidates().find(|path| Path::new(path).exists())
}

/// Finds Clang inside Xcode or the Command Line Tools. `/usr/bin/clang` is an
/// xcrun wrapper on macOS; invoking the real compiler avoids an unnecessary
/// xcrun SDK/cache lookup during every Cargo build.
fn find_macos_clang() -> PathBuf {
    let mut candidates = Vec::new();
    if let Some(developer_dir) = env::var_os("DEVELOPER_DIR").filter(|value| !value.is_empty()) {
        let developer_dir = PathBuf::from(developer_dir);
        candidates.push(developer_dir.join("Toolchains/XcodeDefault.xctoolchain/usr/bin/clang"));
        candidates.push(developer_dir.join("usr/bin/clang"));
    }
    candidates.push(PathBuf::from(
        "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang",
    ));
    candidates.push(PathBuf::from(
        "/Library/Developer/CommandLineTools/usr/bin/clang",
    ));

    candidates
        .into_iter()
        .find(|path| path.is_file())
        .unwrap_or_else(|| PathBuf::from("clang"))
}

fn main() {
    println!("cargo:rustc-check-cfg=cfg(rusty_metal)");

    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    if env::var_os("CARGO_FEATURE_METAL").is_none() {
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set"));
    let obj = out_dir.join("metal_backend.o");
    let lib = out_dir.join("librusty_metal_backend.a");
    let tmp_dir = out_dir.join("xcrun-tmp");
    let _ = fs::create_dir_all(&tmp_dir);

    let mut clang = Command::new(find_macos_clang());
    clang
        .env("TMPDIR", &tmp_dir)
        .env("TMP", &tmp_dir)
        .env("TEMP", &tmp_dir)
        .stderr(Stdio::piped())
        .args(["-x", "objective-c", "-fobjc-arc", "-O3"]);
    if let Some(sdk) = find_macos_sdk() {
        clang.arg("-isysroot").arg(sdk);
    }
    let clang_output = clang
        .args(["-c", "src/metal_backend.m", "-o"])
        .arg(&obj)
        .output();

    let Ok(output) = clang_output else {
        println!("cargo:warning=Metal backend disabled: clang was not available");
        return;
    };
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let detail = stderr
            .lines()
            .find(|line| !line.trim().is_empty())
            .unwrap_or("unknown clang error");
        println!(
            "cargo:warning=Metal backend disabled: Objective-C shim did not compile ({detail})"
        );
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
