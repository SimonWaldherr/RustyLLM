#requires -Version 5.1
# RustyLLM.Build.psm1 - shared helpers for make.ps1 and bench_models.ps1.
#
# Written for Windows PowerShell 5.1 (no ternary/null-coalescing/null-conditional
# operators, no PS7-only cmdlets) so it runs on a stock Windows install with no
# extra components.

Set-StrictMode -Version Latest

function Get-RepoRoot {
    <#
    .SYNOPSIS
        Returns the repository root (the directory containing this module's
        parent 'scripts' folder), independent of the caller's working directory.
    #>
    return (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
}

function Assert-CargoOnPath {
    <#
    .SYNOPSIS
        Ensures `cargo`/`rustup` are reachable, adding %USERPROFILE%\.cargo\bin
        to PATH for this process if rustup installed there but never added it
        to the user's PATH (a common state on fresh rustup installs launched
        from a non-login shell).
    #>
    if (Get-Command cargo -ErrorAction SilentlyContinue) {
        return
    }
    if ($env:USERPROFILE) {
        $candidate = Join-Path $env:USERPROFILE '.cargo\bin'
        if (Test-Path (Join-Path $candidate 'cargo.exe')) {
            $env:PATH = "$candidate;$env:PATH"
        }
    }
    if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
        throw "cargo not found on PATH. Install Rust from https://rustup.rs, then re-run, or add `%USERPROFILE%\.cargo\bin` to PATH."
    }
}

function Get-PinnedRustChannel {
    <#
    .SYNOPSIS
        Reads the `channel = "..."` value pinned in rust-toolchain.toml.
    #>
    param([string]$RepoRoot = (Get-RepoRoot))
    $tomlPath = Join-Path $RepoRoot 'rust-toolchain.toml'
    if (Test-Path -LiteralPath $tomlPath) {
        $content = Get-Content -Raw -LiteralPath $tomlPath
        $match = [regex]::Match($content, 'channel\s*=\s*"([^"]+)"')
        if ($match.Success) {
            return $match.Groups[1].Value
        }
    }
    return 'stable'
}

function Find-MsvcBuildTools {
    <#
    .SYNOPSIS
        Returns the Visual Studio installation path that has the C++ (VC.Tools)
        workload installed, or $null when no usable MSVC linker is present.
    #>
    $vswhereCandidates = @(
        (Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'),
        (Join-Path $env:ProgramFiles 'Microsoft Visual Studio\Installer\vswhere.exe')
    )
    $vswhere = $null
    foreach ($candidate in $vswhereCandidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) {
            $vswhere = $candidate
            break
        }
    }
    if (-not $vswhere) {
        return $null
    }

    # NOTE: no `2>` redirect on this native call -- see Invoke-Checked's
    # scoping note. Under $ErrorActionPreference = 'Stop' (set by make.ps1),
    # redirecting a native command's stderr (even to $null) wraps each stderr
    # line as a terminating NativeCommandError instead of just discarding it.
    # Letting stderr print through untouched is harmless here and avoids that.
    $installPath = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
    if ([string]::IsNullOrWhiteSpace($installPath)) {
        return $null
    }
    return $installPath.Trim()
}

function Test-RustupToolchainInstalled {
    param([Parameter(Mandatory)][string]$Toolchain)
    $lines = & rustup toolchain list
    foreach ($line in $lines) {
        if ($line -like "$Toolchain*") {
            return $true
        }
    }
    return $false
}

function Resolve-RustEnv {
    <#
    .SYNOPSIS
        Decides how to build on this machine: plain MSVC if the C++ Build
        Tools are installed, otherwise the GNU host toolchain linked with
        rust-lld (which needs no external linker/SDK at all).
    .DESCRIPTION
        Many Windows dev machines have Visual Studio installed WITHOUT the
        "Desktop development with C++" workload, which leaves rustc unable to
        link the default *-pc-windows-msvc target. Detect that up front rather
        than failing deep inside a cargo build with a cryptic linker error.
    .OUTPUTS
        PSCustomObject with RustupToolchain (env RUSTUP_TOOLCHAIN value, or
        $null to leave the default active toolchain alone), RustFlags (final
        RUSTFLAGS value to export), and Reason (human-readable explanation,
        printed once by callers).
    #>
    param([string]$OptFlags = '-C target-cpu=native')

    $msvc = Find-MsvcBuildTools
    if ($msvc) {
        return [PSCustomObject]@{
            RustupToolchain = $null
            RustFlags       = $OptFlags
            UsesGnuFallback = $false
            Reason          = "MSVC Build Tools found at $msvc"
        }
    }

    $channel = Get-PinnedRustChannel
    $toolchain = "$channel-x86_64-pc-windows-gnu"

    if (-not (Test-RustupToolchainInstalled -Toolchain $toolchain)) {
        Write-Host "make.ps1: no MSVC Build Tools found; installing $toolchain (one-time)..." -ForegroundColor Yellow
        & rustup toolchain install $toolchain --profile minimal
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install Rust toolchain '$toolchain'. Either install the 'Desktop development with C++' Visual Studio workload, or fix rustup and retry."
        }
    }
    # `--profile minimal` deliberately skips rustfmt/clippy (rust-toolchain.toml
    # pins both), and forcing RUSTUP_TOOLCHAIN like this bypasses rustup's
    # usual "auto-install pinned components" convenience (that only fires
    # through its own rust-toolchain.toml file resolution, not an env
    # override) -- so `cargo clippy`/`cargo fmt` under the GNU fallback would
    # otherwise fail with "cargo-clippy is not installed" even though the
    # toolchain itself is present. `component add` is a fast no-op when
    # already installed, so just ensure both unconditionally.
    & rustup component add --toolchain $toolchain rustfmt clippy
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "make.ps1: could not ensure rustfmt/clippy for $toolchain; 'fmt'/'vet' may fail."
    }

    $linkerFlags = '-C linker=rust-lld -C linker-flavor=ld.lld -C link-self-contained=yes'
    $rustFlags = $OptFlags
    if ($rustFlags -notmatch 'linker=rust-lld') {
        $rustFlags = "$rustFlags $linkerFlags".Trim()
    }

    return [PSCustomObject]@{
        RustupToolchain = $toolchain
        RustFlags       = $rustFlags
        UsesGnuFallback = $true
        Reason          = "No MSVC Build Tools detected; using $toolchain + rust-lld (self-contained linker, no Windows SDK needed)"
    }
}

function Invoke-CargoWithEnv {
    <#
    .SYNOPSIS
        Runs `cargo <ArgumentList>` with RUSTFLAGS/RUSTUP_TOOLCHAIN set per
        Resolve-RustEnv, restoring the previous environment afterward.
    #>
    param(
        [Parameter(Mandatory)][string[]]$ArgumentList,
        [string]$OptFlags = '-C target-cpu=native',
        [PSCustomObject]$RustEnv
    )
    if (-not $RustEnv) {
        $RustEnv = Resolve-RustEnv -OptFlags $OptFlags
        Write-Host "make.ps1: $($RustEnv.Reason)" -ForegroundColor DarkGray
    }

    $envOverrides = @{ RUSTFLAGS = $RustEnv.RustFlags }
    if ($RustEnv.RustupToolchain) {
        $envOverrides['RUSTUP_TOOLCHAIN'] = $RustEnv.RustupToolchain
    }
    Invoke-Checked -FilePath 'cargo' -ArgumentList $ArgumentList -EnvOverrides $envOverrides
}

function Get-CargoFeatureArgs {
    <#
    .SYNOPSIS
        Picks the --no-default-features/--features flags needed to build on
        this machine.
    .DESCRIPTION
        The default feature set (`full`) pulls in the `tls` feature, whose
        `ring` dependency needs a real C compiler to build its assembly/C
        sources. rustup's GNU host toolchain provides rustc/cargo/rust-lld
        but NOT a standalone MinGW-w64 GCC, so `ring`'s build script fails
        with "gcc.exe not found" the moment a build actually needs it (test,
        clippy, or a default build) -- exactly what building this repo hit
        the first time, by hand, on this exact class of machine.

        Default: drop `tls` (keep cli/server/metal) only when the GNU
        fallback is active; MSVC builds `ring` fine via cl.exe, so they get
        cargo's own defaults untouched, matching the Unix Makefile exactly.
        FEATURES=full forces the default set back on (needs a separate
        MinGW-w64 GCC on PATH); FEATURES=<list> picks an explicit set.
    #>
    param(
        [Parameter(Mandatory)][hashtable]$Overrides,
        [Parameter(Mandatory)][PSCustomObject]$RustEnv
    )
    $features = Get-Var -Overrides $Overrides -Name 'FEATURES' -Default ''
    if ($features -eq 'full') { return @() }
    if ($features) { return @('--no-default-features', '--features', $features) }
    if ($RustEnv.UsesGnuFallback) {
        return @('--no-default-features', '--features', 'cli,server,metal')
    }
    return @()
}

function Invoke-WithEnvOverrides {
    <#
    .SYNOPSIS
        Runs $Body with the given environment variables temporarily set,
        restoring (or removing) each one afterward regardless of success.
    #>
    param(
        [hashtable]$Overrides = @{},
        [Parameter(Mandatory)][scriptblock]$Body
    )
    $saved = @{}
    foreach ($key in $Overrides.Keys) {
        $existing = Get-Item -Path "Env:$key" -ErrorAction SilentlyContinue
        if ($existing) { $saved[$key] = $existing.Value } else { $saved[$key] = $null }
        Set-Item -Path "Env:$key" -Value ([string]$Overrides[$key])
    }
    try {
        & $Body
    } finally {
        foreach ($key in $Overrides.Keys) {
            if ($null -ne $saved[$key]) {
                Set-Item -Path "Env:$key" -Value $saved[$key]
            } else {
                Remove-Item -Path "Env:$key" -ErrorAction SilentlyContinue
            }
        }
    }
}

function Invoke-Checked {
    <#
    .SYNOPSIS
        Runs a native executable with scoped environment-variable overrides
        and throws (stopping the script) on a non-zero exit code, matching
        Make's default fail-fast behavior.
    .NOTES
        The check-and-throw happens INSIDE the scriptblock passed to
        Invoke-WithEnvOverrides, not after it returns: a scriptblock invoked
        via `&` runs in a new child scope of wherever it is *executed*
        (Invoke-WithEnvOverrides's frame), not of wherever it was *written*
        (this function), so writing to a plain local variable here and
        reading it back after the call would silently read the wrong (unset)
        variable. `.GetNewClosure()` binds $FilePath/$ArgumentList's current
        values into the scriptblock itself, sidestepping that entirely;
        $LASTEXITCODE is a genuine engine-global and needs no such binding.
    #>
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [string[]]$ArgumentList = @(),
        [hashtable]$EnvOverrides = @{}
    )
    $body = {
        & $FilePath @ArgumentList
        if ($LASTEXITCODE -ne 0) {
            throw "$FilePath $($ArgumentList -join ' ') exited with code $LASTEXITCODE"
        }
    }.GetNewClosure()
    Invoke-WithEnvOverrides -Overrides $EnvOverrides -Body $body
}

function Test-Truthy {
    <#
    .SYNOPSIS
        Mirrors the Makefile's `$(filter 1 true yes on,...)` boolean parsing.
    #>
    param([string]$Value)
    if ([string]::IsNullOrWhiteSpace($Value)) { return $false }
    return @('1', 'true', 'yes', 'on') -contains $Value.Trim().ToLowerInvariant()
}

function Get-Var {
    <#
    .SYNOPSIS
        Resolves a configuration value with Make-like precedence: an explicit
        CLI override (Key=Value passed to make.ps1), then an environment
        variable of the same name, then the caller-supplied default.
    .PARAMETER NoEnvFallback
        Skips the environment-variable lookup. Required for names that
        collide with a Windows-predefined variable of unrelated meaning --
        e.g. TEMP/TMP always point at the temp *directory*, so a naive env
        fallback for the Makefile's sampling-temperature TEMP variable would
        silently pick up "C:\Users\...\AppData\Local\Temp" instead of "0".
    #>
    param(
        [Parameter(Mandatory)][hashtable]$Overrides,
        [Parameter(Mandatory)][string]$Name,
        [string]$Default = '',
        [switch]$NoEnvFallback
    )
    if ($Overrides.ContainsKey($Name)) { return $Overrides[$Name] }
    if (-not $NoEnvFallback) {
        $envValue = [System.Environment]::GetEnvironmentVariable($Name)
        if ($envValue) { return $envValue }
    }
    return $Default
}

function Test-GgufFile {
    <#
    .SYNOPSIS
        Reports whether a file starts with the 4-byte "GGUF" magic, for
        extensionless files such as Ollama's content-addressed blob store.
    #>
    param([Parameter(Mandatory)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    try {
        $stream = [System.IO.File]::OpenRead($Path)
        try {
            $buffer = New-Object byte[] 4
            $read = $stream.Read($buffer, 0, 4)
            if ($read -lt 4) { return $false }
            return ([System.Text.Encoding]::ASCII.GetString($buffer) -eq 'GGUF')
        } finally {
            $stream.Dispose()
        }
    } catch {
        return $false
    }
}

function Get-GgufModelFiles {
    <#
    .SYNOPSIS
        Recursively lists GGUF model files under $Directory, skipping mmproj
        (vision projector) files, which are not standalone language models.
    #>
    param([Parameter(Mandatory)][string]$Directory)
    if (-not (Test-Path -LiteralPath $Directory -PathType Container)) {
        # `,@()` (not `@()`): an empty array returned bare across a function
        # boundary collapses to $null for the caller, not an empty array --
        # the same enumeration behavior the single-item case hits below.
        return , @()
    }
    $items = Get-ChildItem -LiteralPath $Directory -Recurse -File -ErrorAction SilentlyContinue
    $models = foreach ($item in $items) {
        if ($item.Name -match '(?i)mmproj') { continue }
        if ($item.Extension -ieq '.gguf') {
            $item
        } elseif ($item.FullName -match '[\\/]blobs[\\/]sha256-[0-9a-f]+$' -and (Test-GgufFile $item.FullName)) {
            $item
        }
    }
    # The unary comma forces the pipeline to hand back an array even when
    # there is exactly one match; without it PowerShell silently unwraps a
    # single-item array to a bare scalar on return, which breaks `.Count`
    # (and any strict-mode caller) for the common "one model in this folder" case.
    return , @($models | Sort-Object FullName)
}

function Test-HasModelFiles {
    param([Parameter(Mandatory)][string]$Directory)
    return ((Get-GgufModelFiles -Directory $Directory).Count -gt 0)
}

function Get-ModelDirCandidates {
    <#
    .SYNOPSIS
        Windows-first search order for local GGUF model caches (LM Studio,
        Ollama, GPT4All, Jan), mirroring bench_models.sh's model_dir_candidates
        so both scripts find the same models on the same machine.
    #>
    $candidates = New-Object System.Collections.Generic.List[string]
    if ($env:RUSTY_LLM_MODEL_DIR) { [void]$candidates.Add($env:RUSTY_LLM_MODEL_DIR) }
    if ($env:OLLAMA_MODELS) { [void]$candidates.Add($env:OLLAMA_MODELS) }

    if ($env:USERPROFILE) {
        [void]$candidates.Add((Join-Path $env:USERPROFILE '.lmstudio\models\lmstudio-community'))
        [void]$candidates.Add((Join-Path $env:USERPROFILE '.lmstudio\models'))
        [void]$candidates.Add((Join-Path $env:USERPROFILE '.ollama\models'))
        [void]$candidates.Add((Join-Path $env:USERPROFILE 'models'))
    }
    if ($env:LOCALAPPDATA) {
        [void]$candidates.Add((Join-Path $env:LOCALAPPDATA 'LM Studio\models'))
        [void]$candidates.Add((Join-Path $env:LOCALAPPDATA 'Ollama\models'))
        [void]$candidates.Add((Join-Path $env:LOCALAPPDATA 'nomic.ai\GPT4All'))
        [void]$candidates.Add((Join-Path $env:LOCALAPPDATA 'Jan\models'))
    }
    [void]$candidates.Add('.\models')
    return $candidates
}

function Resolve-ModelDir {
    <#
    .SYNOPSIS
        Finds the first candidate directory that actually contains GGUF
        models. Throws (matching bench_models.sh's `resolve_model_dir` exit 1)
        when $Preferred is set but missing, or when nothing is found.
    #>
    param([string]$Preferred)

    if ($Preferred) {
        if (Test-Path -LiteralPath $Preferred -PathType Container) {
            return (Resolve-Path -LiteralPath $Preferred).Path
        }
        throw "MODEL_DIR does not exist: $Preferred"
    }

    $seen = New-Object System.Collections.Generic.HashSet[string]
    foreach ($candidate in (Get-ModelDirCandidates)) {
        if ([string]::IsNullOrWhiteSpace($candidate)) { continue }
        if (-not $seen.Add($candidate)) { continue }
        if (Test-HasModelFiles -Directory $candidate) {
            return $candidate
        }
    }

    $checked = (Get-ModelDirCandidates | ForEach-Object { "  - $_" }) -join "`n"
    throw "No GGUF text models found in known model directories.`nSet MODEL_DIR=<path> or `$env:RUSTY_LLM_MODEL_DIR=<path>.`nChecked:`n$checked"
}

function Get-DefaultModelDir {
    <#
    .SYNOPSIS
        Best-effort MODEL_DIR default for commands that pass --model-dir
        through to rusty-llm.exe: the first real match, or an unvalidated
        LM Studio-shaped fallback path so the binary's own "no models found"
        error (with its own search list) is what the user actually sees.
    #>
    try {
        return Resolve-ModelDir
    } catch {
        if ($env:USERPROFILE) {
            return (Join-Path $env:USERPROFILE '.lmstudio\models\lmstudio-community')
        }
        return '.\models'
    }
}

function Get-RustyLlmBinaryPath {
    param(
        [string]$RepoRoot = (Get-RepoRoot),
        [string]$Profile = 'release',
        [string]$AppName = 'rusty-llm'
    )
    return (Join-Path $RepoRoot "target\$Profile\$AppName.exe")
}

Export-ModuleMember -Function *
