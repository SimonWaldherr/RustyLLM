use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn decode_c_string(text: &str) -> Option<String> {
    let mut decoded = String::with_capacity(text.len());
    let mut chars = text.chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            decoded.push(ch);
            continue;
        }
        decoded.push(match chars.next()? {
            'n' => '\n',
            'r' => '\r',
            't' => '\t',
            '\\' => '\\',
            '"' => '"',
            other => other,
        });
    }
    Some(decoded)
}

/// Extracts the embedded kernel strings into one normal source file. The
/// Objective-C fallback and the build-time compiler therefore consume the
/// exact same independently maintained kernels.
fn extract_metal_source(objective_c: &str) -> Option<String> {
    let mut source = String::new();
    let mut in_kernel = false;
    for line in objective_c.lines() {
        if line.starts_with("static NSString *const k") && line.contains("Source") {
            in_kernel = true;
            continue;
        }
        if !in_kernel {
            continue;
        }
        let ends_kernel = line.trim_end_matches('\r').trim_end().ends_with(';');
        let start = line.find('"')?;
        let end = line.rfind('"')?;
        if end <= start {
            return None;
        }
        source.push_str(&decode_c_string(&line[start + 1..end])?);
        if ends_kernel {
            in_kernel = false;
            source.push('\n');
        }
    }
    (!source.is_empty()).then_some(source)
}

fn write_metallib_header(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut header = String::from(
        "#pragma once\n#include <stddef.h>\nstatic const unsigned char rusty_precompiled_metallib[] = {\n",
    );
    if bytes.is_empty() {
        header.push_str("0\n");
    } else {
        for chunk in bytes.chunks(16) {
            for byte in chunk {
                header.push_str(&format!("0x{byte:02x},"));
            }
            header.push('\n');
        }
    }
    header.push_str(&format!(
        "}};\nstatic const size_t rusty_precompiled_metallib_len = {};\n",
        bytes.len()
    ));
    fs::write(path, header)
}

fn precompile_metal_library(out_dir: &Path, tmp_dir: &Path) -> Vec<u8> {
    let Ok(objective_c) = fs::read_to_string("src/metal_backend.m") else {
        return Vec::new();
    };
    let Some(source) = extract_metal_source(&objective_c) else {
        return Vec::new();
    };
    let source_path = out_dir.join("rusty_kernels.metal");
    let air_path = out_dir.join("rusty_kernels.air");
    let library_path = out_dir.join("rusty_kernels.metallib");
    if fs::write(&source_path, source).is_err() {
        return Vec::new();
    }

    let compile = Command::new("xcrun")
        .env("TMPDIR", tmp_dir)
        .env("TMP", tmp_dir)
        .env("TEMP", tmp_dir)
        .args(["-sdk", "macosx", "metal", "-ffast-math", "-c"])
        .arg(&source_path)
        .arg("-o")
        .arg(&air_path)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    if !matches!(compile, Ok(status) if status.success()) {
        return Vec::new();
    }
    let link = Command::new("xcrun")
        .env("TMPDIR", tmp_dir)
        .env("TMP", tmp_dir)
        .env("TEMP", tmp_dir)
        .args(["-sdk", "macosx", "metallib"])
        .arg(&air_path)
        .arg("-o")
        .arg(&library_path)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    if !matches!(link, Ok(status) if status.success()) {
        return Vec::new();
    }
    fs::read(library_path).unwrap_or_default()
}

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
    let metallib = precompile_metal_library(&out_dir, &tmp_dir);
    let metallib_header = out_dir.join("rusty_metallib.h");
    if write_metallib_header(&metallib_header, &metallib).is_err() {
        println!("cargo:warning=Metal backend disabled: could not generate kernel header");
        return;
    }

    let mut clang = Command::new(find_macos_clang());
    clang
        .env("TMPDIR", &tmp_dir)
        .env("TMP", &tmp_dir)
        .env("TEMP", &tmp_dir)
        .stderr(Stdio::piped())
        .arg("-I")
        .arg(&out_dir)
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
