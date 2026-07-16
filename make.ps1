#requires -Version 5.1
<#
.SYNOPSIS
    Windows-native equivalent of the repository's Makefile.

.DESCRIPTION
    RustyLLM's Makefile assumes a Unix shell (Git Bash, WSL, macOS/Linux) and
    a `make` binary, neither of which a stock Windows install has. This script
    covers the same targets using only PowerShell 5.1 (present on every
    supported Windows release) plus `cargo`/`rustup`.

    On top of a straight port it auto-detects whether MSVC Build Tools are
    installed; if not (a common state: Visual Studio present but without the
    "Desktop development with C++" workload), it transparently falls back to
    the GNU host toolchain linked with rust-lld, which needs no external
    linker or Windows SDK at all. See scripts/RustyLLM.Build.psm1.

.PARAMETER Target
    The target to run (default: all). See '.\make.ps1 help' for the full list.

.PARAMETER RemainingArgs
    Variable overrides in `Key=Value` form, mirroring `make target KEY=value`,
    e.g. `.\make.ps1 run MODEL=C:\models\foo.gguf PROMPT="Hello"`.

.EXAMPLE
    .\make.ps1 release

.EXAMPLE
    .\make.ps1 run MODEL=C:\models\Ministral-3-3B.gguf PROMPT="Explain GGUF briefly"

.EXAMPLE
    .\make.ps1 serve ADDR=127.0.0.1:8080 CHAT=1
#>
param(
    [Parameter(Position = 0)]
    [string]$Target = 'all',

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Import-Module (Join-Path $PSScriptRoot 'scripts\RustyLLM.Build.psm1') -Force
Assert-CargoOnPath

$Overrides = @{}
foreach ($rawArg in $RemainingArgs) {
    $match = [regex]::Match($rawArg, '^([A-Za-z_][A-Za-z0-9_]*)=(.*)$')
    if ($match.Success) {
        $Overrides[$match.Groups[1].Value] = $match.Groups[2].Value
    } else {
        Write-Warning "make.ps1: ignoring unrecognized argument '$rawArg' (expected KEY=value)"
    }
}

# ─── Shared argument builders ────────────────────────────────────────────────

function Get-ModelDirArg {
    param([hashtable]$Overrides)
    $preferred = Get-Var -Overrides $Overrides -Name 'MODEL_DIR' -Default ''
    if ($preferred) {
        return (Resolve-ModelDir -Preferred $preferred)
    }
    return (Get-DefaultModelDir)
}

function Get-ModelArgs {
    <#
    .SYNOPSIS
        Builds the shared `--model-dir <dir> [--model <name>]` argument pair.
        MODEL_DIR is resolved lazily, right here, only when a target that
        actually needs it runs -- the same "don't scan the filesystem for
        make fmt" fix applied to the Unix Makefile.
    #>
    param([hashtable]$Overrides, [switch]$UseBenchModelFallback)
    $modelDir = Get-ModelDirArg -Overrides $Overrides
    $result = @('--model-dir', $modelDir)
    $model = Get-Var -Overrides $Overrides -Name 'MODEL' -Default ''
    if (-not $model -and $UseBenchModelFallback) {
        $model = Get-Var -Overrides $Overrides -Name 'BENCH_MODEL' -Default 'Ministral-3-3B-Instruct-2512-Q4_K_M.gguf'
    }
    if ($model) {
        $result += @('--model', $model)
    }
    return $result
}

function Write-MetalNote {
    Write-Host 'make.ps1: RUSTY_LLM_METAL=1 has no effect on Windows (the Metal backend is macOS-only); running the CPU path.' -ForegroundColor DarkYellow
}

# ─── Build targets ────────────────────────────────────────────────────────────

function Invoke-TargetBuild {
    param([hashtable]$Overrides)
    $opt = Get-Var -Overrides $Overrides -Name 'RUSTFLAGS' -Default ''
    $rustEnv = Resolve-RustEnv -OptFlags $opt
    Write-Host "make.ps1: $($rustEnv.Reason)" -ForegroundColor DarkGray
    $featureArgs = Get-CargoFeatureArgs -Overrides $Overrides -RustEnv $rustEnv
    Invoke-CargoWithEnv -ArgumentList (@('build') + $featureArgs) -RustEnv $rustEnv
}

function Invoke-TargetRelease {
    param([hashtable]$Overrides)
    $opt = Get-Var -Overrides $Overrides -Name 'RUSTFLAGS' -Default '-C target-cpu=native'
    $rustEnv = Resolve-RustEnv -OptFlags $opt
    Write-Host "make.ps1: $($rustEnv.Reason)" -ForegroundColor DarkGray
    $featureArgs = Get-CargoFeatureArgs -Overrides $Overrides -RustEnv $rustEnv
    Invoke-CargoWithEnv -ArgumentList (@('build', '--release') + $featureArgs) -RustEnv $rustEnv
}

function Invoke-TargetReleaseMax {
    param([hashtable]$Overrides)
    $opt = Get-Var -Overrides $Overrides -Name 'RUSTFLAGS' -Default '-C target-cpu=native'
    $rustEnv = Resolve-RustEnv -OptFlags $opt
    Write-Host "make.ps1: $($rustEnv.Reason)" -ForegroundColor DarkGray
    $featureArgs = Get-CargoFeatureArgs -Overrides $Overrides -RustEnv $rustEnv
    Invoke-CargoWithEnv -ArgumentList (@('build', '--profile', 'release-max') + $featureArgs) -RustEnv $rustEnv
}

# ─── Run / serve targets ──────────────────────────────────────────────────────

function Invoke-TargetRun {
    param([hashtable]$Overrides)
    Invoke-TargetRelease -Overrides $Overrides
    $bin = Get-RustyLlmBinaryPath
    $fullArgs = (Get-ModelArgs -Overrides $Overrides) + @(
        '--profile', (Get-Var -Overrides $Overrides -Name 'PROFILE' -Default 'auto'),
        '--prompt', (Get-Var -Overrides $Overrides -Name 'PROMPT' -Default 'Wer war Albert Einstein?'),
        '--max-tokens', (Get-Var -Overrides $Overrides -Name 'MAX_TOKENS' -Default '32'),
        '--temp', (Get-Var -Overrides $Overrides -Name 'TEMP' -Default '0' -NoEnvFallback),
        '--top-p', (Get-Var -Overrides $Overrides -Name 'TOP_P' -Default '0.9'),
        '--top-k', (Get-Var -Overrides $Overrides -Name 'TOP_K' -Default '40')
    )
    Invoke-Checked -FilePath $bin -ArgumentList $fullArgs
}

function Invoke-TargetRepl {
    param([hashtable]$Overrides)
    Invoke-TargetRelease -Overrides $Overrides
    $bin = Get-RustyLlmBinaryPath
    Invoke-Checked -FilePath $bin -ArgumentList ((Get-ModelArgs -Overrides $Overrides) + @('--repl'))
}

function Invoke-TargetServe {
    param([hashtable]$Overrides, [switch]$ForceMetal, [string]$ProfileOverride = '')
    Invoke-TargetRelease -Overrides $Overrides
    $bin = Get-RustyLlmBinaryPath
    $addr = Get-Var -Overrides $Overrides -Name 'ADDR' -Default '127.0.0.1:8080'
    $serveAddr = Get-Var -Overrides $Overrides -Name 'SERVE_ADDR' -Default $addr
    $chat = Get-Var -Overrides $Overrides -Name 'CHAT' -Default '1'

    $fullArgs = (Get-ModelArgs -Overrides $Overrides) + @('--serve', $serveAddr)
    if ($ProfileOverride) { $fullArgs += @('--profile', $ProfileOverride) }
    if (Test-Truthy $chat) { $fullArgs += '--chat' }

    $envOverrides = @{}
    if ($ForceMetal) {
        $envOverrides['RUSTY_LLM_METAL'] = '1'
        Write-MetalNote
    }
    Invoke-Checked -FilePath $bin -ArgumentList $fullArgs -EnvOverrides $envOverrides
}

function Invoke-TargetHttps {
    param([hashtable]$Overrides)
    Invoke-TargetRelease -Overrides $Overrides
    $bin = Get-RustyLlmBinaryPath
    $addr = Get-Var -Overrides $Overrides -Name 'ADDR' -Default '127.0.0.1:8080'
    $serveAddr = Get-Var -Overrides $Overrides -Name 'SERVE_ADDR' -Default $addr
    $chat = Get-Var -Overrides $Overrides -Name 'CHAT' -Default '1'
    $tlsCert = Get-Var -Overrides $Overrides -Name 'TLS_CERT' -Default 'cert.pem'
    $tlsKey = Get-Var -Overrides $Overrides -Name 'TLS_KEY' -Default 'key.pem'

    $fullArgs = (Get-ModelArgs -Overrides $Overrides) + @('--serve', $serveAddr, '--tls-cert', $tlsCert, '--tls-key', $tlsKey)
    if (Test-Truthy $chat) { $fullArgs += '--chat' }
    Invoke-Checked -FilePath $bin -ArgumentList $fullArgs
}

# ─── Model discovery / inspection targets ─────────────────────────────────────

function Invoke-TargetFindModelDir {
    param([hashtable]$Overrides)
    $preferred = Get-Var -Overrides $Overrides -Name 'MODEL_DIR' -Default ''
    Write-Output (Resolve-ModelDir -Preferred $preferred)
}

function Invoke-TargetListModels {
    param([hashtable]$Overrides)
    Invoke-TargetRelease -Overrides $Overrides
    $bin = Get-RustyLlmBinaryPath
    $modelDir = Get-ModelDirArg -Overrides $Overrides
    Invoke-Checked -FilePath $bin -ArgumentList @('--model-dir', $modelDir, '--list-models')
}

function Invoke-TargetInspect {
    param([hashtable]$Overrides)
    Invoke-TargetRelease -Overrides $Overrides
    $bin = Get-RustyLlmBinaryPath
    Invoke-Checked -FilePath $bin -ArgumentList ((Get-ModelArgs -Overrides $Overrides) + @('--inspect'))
}

function Invoke-TargetListTensors {
    param([hashtable]$Overrides)
    Invoke-TargetRelease -Overrides $Overrides
    $bin = Get-RustyLlmBinaryPath
    Invoke-Checked -FilePath $bin -ArgumentList ((Get-ModelArgs -Overrides $Overrides) + @('--list-tensors'))
}

# ─── Benchmarking targets ─────────────────────────────────────────────────────

function Invoke-TargetCargoBench {
    param([hashtable]$Overrides)
    $opt = Get-Var -Overrides $Overrides -Name 'RUSTFLAGS' -Default ''
    $rustEnv = Resolve-RustEnv -OptFlags $opt
    Write-Host "make.ps1: $($rustEnv.Reason)" -ForegroundColor DarkGray
    $featureArgs = Get-CargoFeatureArgs -Overrides $Overrides -RustEnv $rustEnv
    Invoke-CargoWithEnv -ArgumentList (@('bench') + $featureArgs) -RustEnv $rustEnv
}

function Invoke-GenerationBenchmark {
    <#
    .SYNOPSIS
        Shared `--bench --bench-json` invocation used by bench-model[-metal|
        -ultra], synonym-bench, and nato-bench[-metal].
    #>
    param(
        [hashtable]$Overrides,
        [Parameter(Mandatory)][string]$Prompt,
        [Parameter(Mandatory)][string]$MaxTokens,
        [string]$Temp = '0',
        [string]$ProfileOverride = '',
        [string[]]$ExtraArgs = @(),
        [switch]$ForceMetal
    )
    Invoke-TargetRelease -Overrides $Overrides
    $bin = Get-RustyLlmBinaryPath
    $profileValue = $ProfileOverride
    if (-not $profileValue) { $profileValue = Get-Var -Overrides $Overrides -Name 'PROFILE' -Default 'auto' }
    $benchRuns = Get-Var -Overrides $Overrides -Name 'BENCH_RUNS' -Default '3'

    $fullArgs = (Get-ModelArgs -Overrides $Overrides -UseBenchModelFallback) + @(
        '--profile', $profileValue, '--prompt', $Prompt, '--max-tokens', $MaxTokens, '--temp', $Temp
    ) + $ExtraArgs + @('--bench', '--bench-json', '--bench-runs', $benchRuns)

    $envOverrides = @{}
    if ($ForceMetal) {
        $envOverrides['RUSTY_LLM_METAL'] = '1'
        Write-MetalNote
    }
    Invoke-Checked -FilePath $bin -ArgumentList $fullArgs -EnvOverrides $envOverrides
}

function Invoke-TargetBenchModel {
    param([hashtable]$Overrides, [switch]$ForceMetal, [string]$ProfileOverride = '')
    $prompt = Get-Var -Overrides $Overrides -Name 'PROMPT' -Default 'Wer war Albert Einstein?'
    $maxTokens = Get-Var -Overrides $Overrides -Name 'MAX_TOKENS' -Default '32'
    $temp = Get-Var -Overrides $Overrides -Name 'TEMP' -Default '0' -NoEnvFallback
    Invoke-GenerationBenchmark -Overrides $Overrides -Prompt $prompt -MaxTokens $maxTokens -Temp $temp `
        -ForceMetal:$ForceMetal -ProfileOverride $ProfileOverride
}

function Invoke-TargetSynonymBench {
    param([hashtable]$Overrides)
    $prompt = Get-Var -Overrides $Overrides -Name 'SYNONYM_PROMPT' -Default 'Nenne ein Synonym für Synonym und antworte nur mit diesem einen Wort.'
    $topP = Get-Var -Overrides $Overrides -Name 'TOP_P' -Default '0.9'
    $topK = Get-Var -Overrides $Overrides -Name 'TOP_K' -Default '40'
    Invoke-GenerationBenchmark -Overrides $Overrides -Prompt $prompt -MaxTokens '8' -Temp '0' `
        -ExtraArgs @('--top-p', $topP, '--top-k', $topK)
}

function Invoke-TargetNatoBench {
    param([hashtable]$Overrides, [switch]$ForceMetal)
    $prompt = Get-Var -Overrides $Overrides -Name 'NATO_PROMPT' -Default 'Output exactly the 26 NATO phonetic alphabet code words from A to Z, one word per line. No letters, numbers, punctuation, parentheses, or explanation.'
    $topP = Get-Var -Overrides $Overrides -Name 'TOP_P' -Default '0.9'
    $topK = Get-Var -Overrides $Overrides -Name 'TOP_K' -Default '40'
    Invoke-GenerationBenchmark -Overrides $Overrides -Prompt $prompt -MaxTokens '128' -Temp '0' `
        -ExtraArgs @('--top-p', $topP, '--top-k', $topK, '--repeat-penalty', '1') -ForceMetal:$ForceMetal
}

function Invoke-KernelBenchmark {
    param([hashtable]$Overrides, [switch]$ForceMetal, [string]$ProfileOverride = '')
    Invoke-TargetRelease -Overrides $Overrides
    $bin = Get-RustyLlmBinaryPath
    $profileValue = $ProfileOverride
    if (-not $profileValue) { $profileValue = Get-Var -Overrides $Overrides -Name 'PROFILE' -Default 'auto' }
    $runs = Get-Var -Overrides $Overrides -Name 'KERNEL_BENCH_RUNS' -Default '25'
    $layer = Get-Var -Overrides $Overrides -Name 'KERNEL_BENCH_LAYER' -Default '0'

    $fullArgs = (Get-ModelArgs -Overrides $Overrides) + @(
        '--profile', $profileValue, '--kernel-bench-json', '--kernel-bench-runs', $runs, '--kernel-bench-layer', $layer
    )
    $envOverrides = @{}
    if ($ForceMetal) {
        $envOverrides['RUSTY_LLM_METAL'] = '1'
        Write-MetalNote
    }
    Invoke-Checked -FilePath $bin -ArgumentList $fullArgs -EnvOverrides $envOverrides
}

function Invoke-TargetBenchModels {
    param([hashtable]$Overrides)
    Invoke-TargetRelease -Overrides $Overrides
    $scriptPath = Join-Path (Get-RepoRoot) 'bench_models.ps1'
    Invoke-WithEnvOverrides -Overrides $Overrides -Body {
        & $scriptPath
        if ($LASTEXITCODE -ne 0) { throw "bench_models.ps1 exited with code $LASTEXITCODE" }
    }
}

function Invoke-TargetBenchmarkReport {
    param([hashtable]$Overrides)
    $scriptPath = Join-Path (Get-RepoRoot) 'bench_models.ps1'
    $withReportOnly = $Overrides.Clone()
    $withReportOnly['REPORT_ONLY'] = '1'
    Invoke-WithEnvOverrides -Overrides $withReportOnly -Body {
        & $scriptPath
        if ($LASTEXITCODE -ne 0) { throw "bench_models.ps1 exited with code $LASTEXITCODE" }
    }
}

# ─── Quality gate targets ─────────────────────────────────────────────────────

function Invoke-TargetFmt {
    param([hashtable]$Overrides)
    Invoke-Checked -FilePath 'cargo' -ArgumentList @('fmt')
}

function Invoke-TargetTest {
    param([hashtable]$Overrides)
    $opt = Get-Var -Overrides $Overrides -Name 'RUSTFLAGS' -Default ''
    $rustEnv = Resolve-RustEnv -OptFlags $opt
    Write-Host "make.ps1: $($rustEnv.Reason)" -ForegroundColor DarkGray
    $featureArgs = Get-CargoFeatureArgs -Overrides $Overrides -RustEnv $rustEnv
    Invoke-CargoWithEnv -ArgumentList (@('test') + $featureArgs) -RustEnv $rustEnv
}

function Invoke-TargetVet {
    param([hashtable]$Overrides)
    $opt = Get-Var -Overrides $Overrides -Name 'RUSTFLAGS' -Default ''
    $rustEnv = Resolve-RustEnv -OptFlags $opt
    Write-Host "make.ps1: $($rustEnv.Reason)" -ForegroundColor DarkGray
    $featureArgs = Get-CargoFeatureArgs -Overrides $Overrides -RustEnv $rustEnv
    Invoke-CargoWithEnv -ArgumentList (@('clippy', '--all-targets') + $featureArgs + @('--', '-D', 'warnings')) -RustEnv $rustEnv
}

function Invoke-TargetCheck {
    param([hashtable]$Overrides)
    Invoke-TargetFmt -Overrides $Overrides
    Invoke-TargetTest -Overrides $Overrides
    Invoke-TargetVet -Overrides $Overrides
}

function Invoke-TargetAll {
    param([hashtable]$Overrides)
    Invoke-TargetCheck -Overrides $Overrides
    Invoke-TargetRelease -Overrides $Overrides
}

# ─── wasm / clean ─────────────────────────────────────────────────────────────

function Invoke-TargetWasm {
    param([hashtable]$Overrides)
    $wasmTarget = Get-Var -Overrides $Overrides -Name 'WASM_TARGET' -Default 'wasm32-unknown-unknown'
    $wasmOut = Get-Var -Overrides $Overrides -Name 'WASM_OUT' -Default 'demo\wasm\pkg'
    $wasmBindgen = Get-Var -Overrides $Overrides -Name 'WASM_BINDGEN' -Default 'wasm-bindgen'
    $wasmBindgenVersion = Get-Var -Overrides $Overrides -Name 'WASM_BINDGEN_VERSION' -Default '0.2.100'
    $wasmOpt = Get-Var -Overrides $Overrides -Name 'WASM_OPT' -Default 'wasm-opt'
    $wasmOptFlags = Get-Var -Overrides $Overrides -Name 'WASM_OPT_FLAGS' -Default '-Oz'

    # The wasm feature set is explicit here (no tls/ring in it), so this
    # doesn't need Get-CargoFeatureArgs -- just the toolchain/RUSTFLAGS setup
    # from Resolve-RustEnv for machines without MSVC Build Tools. Resolve it
    # FIRST: `rustup target add`/`target list --installed` operate on whatever
    # toolchain RUSTUP_TOOLCHAIN currently points at, and that must be the
    # same one the actual build below runs under, or the wasm32 component
    # ends up installed for the wrong (default) toolchain.
    $rustEnv = Resolve-RustEnv -OptFlags ''
    Write-Host "make.ps1: $($rustEnv.Reason)" -ForegroundColor DarkGray
    $toolchainEnv = @{}
    if ($rustEnv.RustupToolchain) { $toolchainEnv['RUSTUP_TOOLCHAIN'] = $rustEnv.RustupToolchain }

    Invoke-WithEnvOverrides -Overrides $toolchainEnv -Body {
        $installedTargets = & rustup target list --installed
        $hasTarget = $false
        foreach ($line in $installedTargets) {
            if ($line.Trim() -eq $wasmTarget) { $hasTarget = $true; break }
        }
        if (-not $hasTarget) {
            Invoke-Checked -FilePath 'rustup' -ArgumentList @('target', 'add', $wasmTarget)
        }
    }.GetNewClosure()

    Invoke-CargoWithEnv -ArgumentList @(
        'build', '--lib', '--release', '--target', $wasmTarget, '--no-default-features', '--features', 'wasm'
    ) -RustEnv $rustEnv

    $needsInstall = $true
    if (Get-Command $wasmBindgen -ErrorAction SilentlyContinue) {
        $verOutput = (& $wasmBindgen --version) -join ' '
        if ($verOutput -match [regex]::Escape($wasmBindgenVersion)) {
            $needsInstall = $false
        }
    }
    if ($needsInstall) {
        Invoke-Checked -FilePath 'cargo' -ArgumentList @(
            'install', 'wasm-bindgen-cli', '--version', $wasmBindgenVersion, '--locked', '--force'
        )
    }

    if (Test-Path -LiteralPath $wasmOut) {
        Remove-Item -LiteralPath $wasmOut -Recurse -Force
    }
    New-Item -ItemType Directory -Path $wasmOut -Force | Out-Null

    $wasmFile = "target\$wasmTarget\release\rusty_llm.wasm"
    Invoke-Checked -FilePath $wasmBindgen -ArgumentList @(
        '--target', 'web', '--out-dir', $wasmOut, '--out-name', 'rusty_llm', $wasmFile
    )

    if (Get-Command $wasmOpt -ErrorAction SilentlyContinue) {
        $bgWasm = Join-Path $wasmOut 'rusty_llm_bg.wasm'
        $optFlagParts = $wasmOptFlags -split '\s+' | Where-Object { $_ }
        Invoke-Checked -FilePath $wasmOpt -ArgumentList ($optFlagParts + @('-o', $bgWasm, $bgWasm))
    } else {
        Write-Host 'Skipping wasm-opt; install binaryen for optional size optimization.'
    }
}

function Invoke-TargetClean {
    param([hashtable]$Overrides)
    Invoke-Checked -FilePath 'cargo' -ArgumentList @('clean')
    $wasmOut = Get-Var -Overrides $Overrides -Name 'WASM_OUT' -Default 'demo\wasm\pkg'
    if (Test-Path -LiteralPath $wasmOut) {
        Remove-Item -LiteralPath $wasmOut -Recurse -Force
    }
}

# ─── Help ─────────────────────────────────────────────────────────────────────

function Show-Help {
    param([hashtable]$Overrides)
    @'
Windows-native equivalent of the Makefile. Usage:

  .\make.ps1 <target> [KEY=value ...]

Targets:
  all                                  Run check then release build (default)
  build                                 Build debug binary
  release                               Build optimized native binary (auto MSVC/GNU+rust-lld detection)
  release-max                           Build slower FatLTO profile for final benchmarking
  run MODEL=... PROMPT='...'            Generate from a one-shot prompt
  repl MODEL=...                        Start interactive REPL mode
  serve MODEL=... CHAT=1                Start HTTP API / optional web UI
  serve-metal MODEL=...                 Same as serve; RUSTY_LLM_METAL=1 has no effect on Windows
  serve-ultra MODEL=...                 Same with --profile mistral-ultra; Metal env var has no effect here
  https MODEL=...                       Start HTTPS API with TLS_CERT/TLS_KEY
  find-model-dir                        Print the auto-detected GGUF model directory
  list-models                           List GGUFs in MODEL_DIR
  inspect MODEL=...                     Inspect GGUF metadata and compatibility
  list-tensors MODEL=...                Print tensor inventory
  bench [BENCH_MODEL=...]               Alias for bench-model
  cargo-bench                           Run Rust benchmark harness
  bench-model [BENCH_MODEL=...]         Run CLI generation benchmark JSON
  bench-model-metal [BENCH_MODEL=...]   Same; RUSTY_LLM_METAL=1 has no effect on Windows
  bench-model-ultra [BENCH_MODEL=...]   Same with --profile mistral-ultra
  bench-models                          Sweep all local GGUFs and refresh BENCHMARK.md (native PS, no bash/python3/jq)
  benchmark-report                      Rebuild BENCHMARK.md from existing .bench_raw files
  synonym-bench [BENCH_MODEL=...]       Run fixed one-word synonym prompt benchmark
  nato-bench [BENCH_MODEL=...]          Run fixed NATO alphabet prompt benchmark
  nato-bench-metal [BENCH_MODEL=...]    Same; RUSTY_LLM_METAL=1 has no effect on Windows
  kernel-bench MODEL=...                Run isolated kernel benchmark JSON
  kernel-bench-metal MODEL=...          Same; RUSTY_LLM_METAL=1 has no effect on Windows
  kernel-bench-ultra MODEL=...          Same with --profile mistral-ultra
  fmt / test / vet / check              Format, test, lint, or all three
  wasm                                  Build stable web wasm package
  clean                                 Remove build artifacts
  help                                  Show this help

Variables (KEY=value overrides, or set as an environment variable):
'@ | Write-Host

    $modelDirDisplay = 'not found'
    try { $modelDirDisplay = Get-ModelDirArg -Overrides $Overrides } catch { $modelDirDisplay = 'not found' }

    Write-Host "  MODEL_DIR=$modelDirDisplay"
    Write-Host "  MODEL=$(Get-Var -Overrides $Overrides -Name 'MODEL' -Default '')"
    Write-Host "  PROMPT=$(Get-Var -Overrides $Overrides -Name 'PROMPT' -Default 'Wer war Albert Einstein?')"
    Write-Host "  SYNONYM_PROMPT=$(Get-Var -Overrides $Overrides -Name 'SYNONYM_PROMPT' -Default 'Nenne ein Synonym für Synonym und antworte nur mit diesem einen Wort.')"
    Write-Host "  NATO_PROMPT=$(Get-Var -Overrides $Overrides -Name 'NATO_PROMPT' -Default '(26 NATO code words prompt)')"
    Write-Host ("  MAX_TOKENS={0} TEMP={1} TOP_P={2} TOP_K={3}" -f `
        (Get-Var -Overrides $Overrides -Name 'MAX_TOKENS' -Default '32'), `
        (Get-Var -Overrides $Overrides -Name 'TEMP' -Default '0' -NoEnvFallback), `
        (Get-Var -Overrides $Overrides -Name 'TOP_P' -Default '0.9'), `
        (Get-Var -Overrides $Overrides -Name 'TOP_K' -Default '40'))
    Write-Host ("  BENCH_RUNS={0} PROFILE={1} SERVE_ADDR={2} CHAT={3}" -f `
        (Get-Var -Overrides $Overrides -Name 'BENCH_RUNS' -Default '3'), `
        (Get-Var -Overrides $Overrides -Name 'PROFILE' -Default 'auto'), `
        (Get-Var -Overrides $Overrides -Name 'SERVE_ADDR' -Default (Get-Var -Overrides $Overrides -Name 'ADDR' -Default '127.0.0.1:8080')), `
        (Get-Var -Overrides $Overrides -Name 'CHAT' -Default '1'))
    Write-Host ("  KERNEL_BENCH_RUNS={0} KERNEL_BENCH_LAYER={1}" -f `
        (Get-Var -Overrides $Overrides -Name 'KERNEL_BENCH_RUNS' -Default '25'), `
        (Get-Var -Overrides $Overrides -Name 'KERNEL_BENCH_LAYER' -Default '0'))
    Write-Host ("  WASM_OUT={0} WASM_TARGET={1} WASM_BINDGEN_VERSION={2}" -f `
        (Get-Var -Overrides $Overrides -Name 'WASM_OUT' -Default 'demo\wasm\pkg'), `
        (Get-Var -Overrides $Overrides -Name 'WASM_TARGET' -Default 'wasm32-unknown-unknown'), `
        (Get-Var -Overrides $Overrides -Name 'WASM_BINDGEN_VERSION' -Default '0.2.100'))

    $rustEnv = Resolve-RustEnv
    Write-Host ''
    Write-Host "Rust build environment: $($rustEnv.Reason)"
}

# ─── Dispatch ─────────────────────────────────────────────────────────────────

Push-Location (Get-RepoRoot)
try {
    switch ($Target) {
        'all'                { Invoke-TargetAll -Overrides $Overrides }
        'build'               { Invoke-TargetBuild -Overrides $Overrides }
        'release'             { Invoke-TargetRelease -Overrides $Overrides }
        'release-max'         { Invoke-TargetReleaseMax -Overrides $Overrides }
        'run'                 { Invoke-TargetRun -Overrides $Overrides }
        'repl'                { Invoke-TargetRepl -Overrides $Overrides }
        'serve'               { Invoke-TargetServe -Overrides $Overrides }
        'serve-metal'         { Invoke-TargetServe -Overrides $Overrides -ForceMetal }
        'serve-ultra'         { Invoke-TargetServe -Overrides $Overrides -ForceMetal -ProfileOverride 'mistral-ultra' }
        'https'               { Invoke-TargetHttps -Overrides $Overrides }
        'find-model-dir'      { Invoke-TargetFindModelDir -Overrides $Overrides }
        'list-models'         { Invoke-TargetListModels -Overrides $Overrides }
        'inspect'             { Invoke-TargetInspect -Overrides $Overrides }
        'list-tensors'        { Invoke-TargetListTensors -Overrides $Overrides }
        'bench'               { Invoke-TargetBenchModel -Overrides $Overrides }
        'cargo-bench'         { Invoke-TargetCargoBench -Overrides $Overrides }
        'bench-model'         { Invoke-TargetBenchModel -Overrides $Overrides }
        'bench-model-metal'   { Invoke-TargetBenchModel -Overrides $Overrides -ForceMetal }
        'bench-model-ultra'   { Invoke-TargetBenchModel -Overrides $Overrides -ForceMetal -ProfileOverride 'mistral-ultra' }
        'bench-models'        { Invoke-TargetBenchModels -Overrides $Overrides }
        'benchmark-report'    { Invoke-TargetBenchmarkReport -Overrides $Overrides }
        'synonym-bench'       { Invoke-TargetSynonymBench -Overrides $Overrides }
        'nato-bench'          { Invoke-TargetNatoBench -Overrides $Overrides }
        'nato-bench-metal'    { Invoke-TargetNatoBench -Overrides $Overrides -ForceMetal }
        'kernel-bench'        { Invoke-KernelBenchmark -Overrides $Overrides }
        'kernel-bench-metal'  { Invoke-KernelBenchmark -Overrides $Overrides -ForceMetal }
        'kernel-bench-ultra'  { Invoke-KernelBenchmark -Overrides $Overrides -ForceMetal -ProfileOverride 'mistral-ultra' }
        'fmt'                 { Invoke-TargetFmt -Overrides $Overrides }
        'test'                { Invoke-TargetTest -Overrides $Overrides }
        'vet'                 { Invoke-TargetVet -Overrides $Overrides }
        'check'               { Invoke-TargetCheck -Overrides $Overrides }
        'wasm'                { Invoke-TargetWasm -Overrides $Overrides }
        'clean'               { Invoke-TargetClean -Overrides $Overrides }
        'help'                { Show-Help -Overrides $Overrides }
        default {
            Write-Error "Unknown target '$Target'. Run '.\make.ps1 help' to list targets."
            exit 1
        }
    }
} catch {
    Write-Error $_.Exception.Message
    exit 1
} finally {
    Pop-Location
}
