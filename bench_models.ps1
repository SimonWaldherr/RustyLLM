#requires -Version 5.1
<#
.SYNOPSIS
    Windows-native equivalent of bench_models.sh: sweeps every local GGUF
    model and (re)writes BENCHMARK.md.

.DESCRIPTION
    Unlike bench_models.sh, this needs no bash, python3, or jq: JSON parsing
    uses PowerShell's built-in ConvertFrom-Json and the report is generated
    with native string formatting. Only a "cpu" profile exists here since the
    Metal backend is macOS-only -- requesting BENCH_PROFILES=metal on Windows
    prints a warning and skips it rather than silently duplicating the CPU
    numbers under a misleading label.

    Configuration is read from environment variables, matching
    bench_models.sh's documented interface:

      $env:BENCH_PROFILES = 'cpu'
      $env:MODEL_DIR = 'C:\path\to\models'
      $env:MODEL_FILTER = 'phi'; $env:MODEL_LIMIT = '2'; $env:MAX_TOKENS = '16'
      .\bench_models.ps1

      $env:REPORT_ONLY = '1'; .\bench_models.ps1
      $env:FIND_MODEL_DIR_ONLY = '1'; .\bench_models.ps1

    make.ps1's bench-models/benchmark-report targets set these from
    KEY=value CLI overrides and call this script.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Import-Module (Join-Path $PSScriptRoot 'scripts\RustyLLM.Build.psm1') -Force
Assert-CargoOnPath

function Get-Cfg {
    param([string]$Name, [string]$Default = '')
    $value = [System.Environment]::GetEnvironmentVariable($Name)
    if ($value) { return $value }
    return $Default
}

$RepoRoot = Get-RepoRoot
$Binary = Get-Cfg 'BINARY' (Get-RustyLlmBinaryPath -RepoRoot $RepoRoot)
$ModelDirSetting = Get-Cfg 'MODEL_DIR' ''
$BenchProfilesRaw = Get-Cfg 'BENCH_PROFILES' 'cpu'
$ModelFilter = Get-Cfg 'MODEL_FILTER' ''
$ModelLimit = [int](Get-Cfg 'MODEL_LIMIT' '0')
$WaitSecs = [int](Get-Cfg 'WAIT_SECS' '8')
$BenchRuns = [int](Get-Cfg 'BENCH_RUNS' '2')
$MaxTokens = [int](Get-Cfg 'MAX_TOKENS' '64')
$Prompt = Get-Cfg 'PROMPT' 'Explain local LLM inference performance in one concise paragraph.'
$TimeoutSecs = [int](Get-Cfg 'TIMEOUT_SECS' '600')
function Resolve-ConfiguredPath {
    <#
    .SYNOPSIS
        Joins $Value under $RepoRoot unless it's already an absolute path --
        Join-Path (unlike .NET's Path.Combine) doesn't discard the base when
        the second segment is rooted, so passing e.g.
        RAW_DIR=C:\scratch\bench_raw through Join-Path unmodified produces a
        broken concatenated path instead of just using C:\scratch\bench_raw.
    #>
    param([string]$RepoRoot, [string]$Value)
    if ([System.IO.Path]::IsPathRooted($Value)) { return $Value }
    return (Join-Path $RepoRoot $Value)
}

$ReadmePath = Resolve-ConfiguredPath -RepoRoot $RepoRoot -Value (Get-Cfg 'README' 'README.md')
$BenchmarkMdPath = Resolve-ConfiguredPath -RepoRoot $RepoRoot -Value (Get-Cfg 'BENCHMARK_MD' 'BENCHMARK.md')
$RawDir = Resolve-ConfiguredPath -RepoRoot $RepoRoot -Value (Get-Cfg 'RAW_DIR' '.bench_raw')
$ReportOnly = Test-Truthy (Get-Cfg 'REPORT_ONLY' '0')
$FindModelDirOnly = Test-Truthy (Get-Cfg 'FIND_MODEL_DIR_ONLY' '0')

$ResultsCachePath = Join-Path $RawDir 'results.json'
$ProfilesCachePath = Join-Path $RawDir 'profiles.json'

# ─── Small JSON / process helpers ────────────────────────────────────────────

function Get-JsonProp {
    <#
    .SYNOPSIS
        Null-coalescing property lookup on a ConvertFrom-Json object tree
        (like jq's `.a.b // default`), safe under Set-StrictMode: dotting into
        a genuinely absent property throws, so this checks PSObject.Properties
        first instead.
    #>
    param($Object, [string[]]$Path, $Default = $null)
    $current = $Object
    foreach ($segment in $Path) {
        if ($null -eq $current) { return $Default }
        $prop = $current.PSObject.Properties[$segment]
        if (-not $prop) { return $Default }
        $current = $prop.Value
    }
    if ($null -eq $current) { return $Default }
    return $current
}

function Get-EmbeddedJson {
    <#
    .SYNOPSIS
        Extracts the first balanced {...} object from mixed stdout/stderr
        (the CLI prints load/config diagnostics around its JSON payload),
        respecting quoted strings so braces inside string values don't
        confuse the depth count.
    #>
    param([string]$Text)
    $start = $Text.IndexOf('{')
    if ($start -lt 0) { return '{}' }
    $depth = 0
    $inString = $false
    $escape = $false
    for ($i = $start; $i -lt $Text.Length; $i++) {
        $c = $Text[$i]
        if ($inString) {
            if ($escape) { $escape = $false }
            elseif ($c -eq '\') { $escape = $true }
            elseif ($c -eq '"') { $inString = $false }
            continue
        }
        if ($c -eq '"') { $inString = $true }
        elseif ($c -eq '{') { $depth++ }
        elseif ($c -eq '}') {
            $depth--
            if ($depth -eq 0) { return $Text.Substring($start, $i - $start + 1) }
        }
    }
    return '{}'
}

function ConvertTo-ProcessArgumentString {
    param([string[]]$ArgumentList)
    $parts = foreach ($arg in $ArgumentList) {
        if ($arg -match '[\s"]') {
            '"' + ($arg -replace '"', '\"') + '"'
        } else {
            $arg
        }
    }
    return ($parts -join ' ')
}

function Invoke-ProcessCaptured {
    <#
    .SYNOPSIS
        Runs a native process with a hard wall-clock timeout, capturing
        merged-ish stdout/stderr (kept as two streams, concatenated) without
        needing GNU coreutils' `timeout`.
    #>
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [string[]]$ArgumentList = @(),
        [hashtable]$EnvOverrides = @{},
        [int]$TimeoutSeconds = 600
    )
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $FilePath
    $psi.Arguments = ConvertTo-ProcessArgumentString -ArgumentList $ArgumentList
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $psi.WorkingDirectory = (Get-RepoRoot)
    foreach ($key in $EnvOverrides.Keys) {
        $psi.EnvironmentVariables[$key] = [string]$EnvOverrides[$key]
    }

    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $psi
    [void]$proc.Start()
    $stdoutTask = $proc.StandardOutput.ReadToEndAsync()
    $stderrTask = $proc.StandardError.ReadToEndAsync()
    $finished = $proc.WaitForExit($TimeoutSeconds * 1000)

    if (-not $finished) {
        try { $proc.Kill($true) } catch {}
        try { $proc.WaitForExit(5000) } catch {}
        return [PSCustomObject]@{
            ExitCode = -1
            Output   = "$($stdoutTask.Result)`n$($stderrTask.Result)"
            TimedOut = $true
        }
    }
    return [PSCustomObject]@{
        ExitCode = $proc.ExitCode
        Output   = "$($stdoutTask.Result)`n$($stderrTask.Result)"
        TimedOut = $false
    }
}

function Invoke-BinaryJson {
    param(
        [string[]]$ArgumentList,
        [hashtable]$EnvOverrides = @{},
        [int]$TimeoutSeconds = 600
    )
    $proc = Invoke-ProcessCaptured -FilePath $Binary -ArgumentList $ArgumentList -EnvOverrides $EnvOverrides -TimeoutSeconds $TimeoutSeconds
    $jsonText = Get-EmbeddedJson -Text $proc.Output
    $json = $null
    try { $json = $jsonText | ConvertFrom-Json -ErrorAction Stop } catch { $json = $null }
    return [PSCustomObject]@{
        ExitCode  = $proc.ExitCode
        TimedOut  = $proc.TimedOut
        RawOutput = $proc.Output
        Json      = $json
    }
}

function Format-Decimal {
    <#
    .NOTES
        Takes the already-numeric value straight from ConvertFrom-Json (or a
        computed double) and casts it directly with `[double]$Value` rather
        than round-tripping through `[string]` + `[double]::TryParse`: the
        latter's default TryParse overload parses using CurrentCulture, so on
        a non-US-decimal-separator system (e.g. de-DE, comma decimals) it can
        silently misparse an invariant-formatted string like "1.169..." --
        reading the "." as a thousands separator and inflating the value by
        orders of magnitude. `[double]$x` uses PowerShell's own (culture-
        invariant) conversion, sidestepping the mismatch entirely.
    #>
    param($Value)
    if ($null -eq $Value) { return '—' }
    try {
        $number = [double]$Value
    } catch {
        return '—'
    }
    return $number.ToString('F1', [System.Globalization.CultureInfo]::InvariantCulture)
}

function ConvertTo-Slug {
    param([string]$FileName)
    $stem = $FileName -replace '\.gguf$', ''
    return ($stem -replace '[^A-Za-z0-9._-]', '_')
}

function Get-InspectNote {
    <#
    .NOTES
        Every Get-JsonProp call below is wrapped in `@(...)`: a JSON array
        with exactly one (or zero) elements collapses to a bare scalar (or
        $null) when it crosses a PowerShell function-return boundary, which
        would otherwise break `.Count` here for the single-missing-tensor
        case -- the single most common one in practice.
    #>
    param($InspectJson)
    $unsupportedLayouts = @(Get-JsonProp $InspectJson @('gguf', 'unsupported_layouts') @())
    if ($unsupportedLayouts.Count -gt 0) { return [string]$unsupportedLayouts[0] }
    $missingExamples = @(Get-JsonProp $InspectJson @('gguf', 'missing_tensor_examples') @())
    if ($missingExamples.Count -gt 0) { return "missing tensor: $($missingExamples[0])" }
    $unsupportedExamples = @(Get-JsonProp $InspectJson @('gguf', 'unsupported_tensor_examples') @())
    if ($unsupportedExamples.Count -gt 0) { return "unsupported tensor: $($unsupportedExamples[0])" }
    $supportedArch = Get-JsonProp $InspectJson @('model', 'supported_architecture') $true
    if ($supportedArch -eq $false) { return 'unsupported architecture' }
    return 'not loadable'
}

function Get-ProfileConfig {
    param([string]$InputValue)
    $key = $InputValue.Trim().ToLowerInvariant()
    if (@('cpu', 'off', '0') -contains $key) {
        return [PSCustomObject]@{ Key = 'cpu'; Label = 'CPU'; MetalEnv = '0'; Supported = $true }
    }
    if (@('metal', 'gpu', 'on', '1') -contains $key) {
        return [PSCustomObject]@{ Key = 'metal'; Label = 'Metal GPU'; MetalEnv = '1'; Supported = $false }
    }
    throw "Unknown BENCH_PROFILES entry: $InputValue"
}

# ─── find-model-dir / report-only short-circuits ─────────────────────────────

if ($FindModelDirOnly) {
    Write-Output (Resolve-ModelDir -Preferred $ModelDirSetting)
    exit 0
}

if ($ReportOnly -and -not (Test-Path -LiteralPath $ResultsCachePath) ) {
    Write-Host "REPORT_ONLY=1 requires an existing raw cache: $ResultsCachePath"
    exit 1
}

New-Item -ItemType Directory -Path $RawDir -Force | Out-Null

$ResolvedModelDir = $null
if ($ReportOnly) {
    if ($ModelDirSetting) {
        $ResolvedModelDir = $ModelDirSetting
    } else {
        try { $ResolvedModelDir = Resolve-ModelDir } catch { $ResolvedModelDir = 'unknown' }
    }
} else {
    $ResolvedModelDir = Resolve-ModelDir -Preferred $ModelDirSetting
}

$Results = @()
$Profiles = @()

if ($ReportOnly) {
    Write-Host "RustyLLM Model Benchmark (report-only) - $(Get-Date)"
    Write-Host "  Binary   : $Binary"
    Write-Host "  Models   : report-only (from $ResultsCachePath)"
    Write-Host "  Profiles : report-only (from $ProfilesCachePath)"
    Write-Host '================================================================'
    $Results = @(Get-Content -Raw -LiteralPath $ResultsCachePath | ConvertFrom-Json)
    if (Test-Path -LiteralPath $ProfilesCachePath) {
        $Profiles = @(Get-Content -Raw -LiteralPath $ProfilesCachePath | ConvertFrom-Json)
    }
} else {
    if (-not (Test-Path -LiteralPath $Binary)) {
        Write-Host "Binary not found: $Binary"
        Write-Host 'Building release binary first...'
        $rustEnv = Resolve-RustEnv
        Write-Host "make.ps1: $($rustEnv.Reason)" -ForegroundColor DarkGray
        $featureArgs = @()
        if ($rustEnv.UsesGnuFallback) { $featureArgs = @('--no-default-features', '--features', 'cli,server,metal') }
        Invoke-CargoWithEnv -ArgumentList (@('build', '--release') + $featureArgs) -RustEnv $rustEnv
    }

    $profileTokens = $BenchProfilesRaw -split '[,\s]+' | Where-Object { $_ }
    $profileConfigs = @()
    foreach ($token in $profileTokens) {
        $cfg = Get-ProfileConfig -InputValue $token
        if (-not $cfg.Supported) {
            Write-Warning "bench_models.ps1: '$token' requested Metal, but the Metal backend is macOS-only -- RUSTY_LLM_METAL=1 has no effect on Windows and would just duplicate the CPU numbers. Skipping."
            continue
        }
        $profileConfigs += $cfg
    }
    if ($profileConfigs.Count -eq 0) {
        Write-Host 'No runnable profiles requested (only cpu is supported on Windows).'
        exit 1
    }

    $allModelFiles = Get-GgufModelFiles -Directory $ResolvedModelDir
    $models = New-Object System.Collections.Generic.List[string]
    foreach ($file in $allModelFiles) {
        if ($ModelFilter -and $file.FullName -notmatch $ModelFilter) { continue }
        [void]$models.Add($file.FullName)
        if ($ModelLimit -gt 0 -and $models.Count -ge $ModelLimit) { break }
    }
    if ($models.Count -eq 0) {
        Write-Host "No GGUF text models found in $ResolvedModelDir"
        exit 1
    }

    Write-Host "RustyLLM Model Benchmark - $(Get-Date)"
    Write-Host "  Binary   : $Binary"
    Write-Host "  Models   : $($models.Count)  (from $ResolvedModelDir)"
    Write-Host "  Profiles : $(($profileConfigs | ForEach-Object { $_.Key }) -join ' ')"
    Write-Host "  Runs     : $BenchRuns x $MaxTokens tokens each"
    Write-Host "  Pause    : ${WaitSecs}s between models"
    Write-Host '================================================================'

    foreach ($profileCfg in $profileConfigs) {
        $rawProfileDir = Join-Path $RawDir $profileCfg.Key
        New-Item -ItemType Directory -Path $rawProfileDir -Force | Out-Null
        $profileRuntime = 'not observed'

        Write-Host ''
        Write-Host "Profile: $($profileCfg.Label) (RUSTY_LLM_METAL=$($profileCfg.MetalEnv))"
        Write-Host '----------------------------------------------------------------'

        for ($idx = 0; $idx -lt $models.Count; $idx++) {
            $modelPath = $models[$idx]
            $modelFile = Split-Path -Leaf $modelPath
            $rawBase = '{0:D2}_{1}' -f ($idx + 1), (ConvertTo-Slug $modelFile)

            Write-Host ''
            Write-Host "[$($idx + 1)/$($models.Count)][$($profileCfg.Key)] $modelFile"

            $envOverrides = @{ RUSTY_LLM_METAL = $profileCfg.MetalEnv }
            $inspect = Invoke-BinaryJson -ArgumentList @($modelPath, '--inspect') -EnvOverrides $envOverrides -TimeoutSeconds 30
            $inspectJson = $inspect.Json
            if (-not $inspectJson) { $inspectJson = [PSCustomObject]@{} }
            ($inspectJson | ConvertTo-Json -Depth 20) | Set-Content -LiteralPath (Join-Path $rawProfileDir "${rawBase}_inspect.json")

            $metalLine = ($inspect.RawOutput -split "`n" | Where-Object { $_ -match '^Metal:' } | Select-Object -First 1)
            if ($metalLine) { $profileRuntime = $metalLine.Trim() }

            $status = Get-JsonProp $inspectJson @('status') 'unknown'
            $arch = Get-JsonProp $inspectJson @('model', 'architecture') 'unknown'
            $modelName = Get-JsonProp $inspectJson @('model', 'name') ''
            if (-not $modelName) { $modelName = $modelFile }
            $fileBytes = Get-JsonProp $inspectJson @('file_size_bytes') 0
            $fileMb = [Math]::Floor([double]$fileBytes / 1048576)
            $unsupportedCount = Get-JsonProp $inspectJson @('gguf', 'unsupported_tensor_count') 0

            Write-Host "  arch=$arch  status=$status  size=${fileMb}MB  unsupported_tensors=$unsupportedCount"

            if ($status -ne 'supported') {
                $note = Get-InspectNote -InspectJson $inspectJson
                Write-Host "  skip: $note"
                $Results += [PSCustomObject]@{
                    Profile = $profileCfg.Key; ProfileLabel = $profileCfg.Label; Index = $idx + 1
                    Path = $modelPath; File = $modelFile; Name = $modelName; Arch = $arch
                    Status = $status; Mb = $fileMb; LoadMs = $null; Decode = $null; Prefill = $null; Note = $note
                }
                Write-Host "  Waiting ${WaitSecs}s..."
                Start-Sleep -Seconds $WaitSecs
                continue
            }

            Write-Host "  Running $BenchRuns runs x $MaxTokens tokens..."
            $benchArgs = @(
                $modelPath, '--bench-json', '--bench-runs', $BenchRuns, '--max-tokens', $MaxTokens, '--prompt', $Prompt
            )
            $bench = Invoke-BinaryJson -ArgumentList $benchArgs -EnvOverrides $envOverrides -TimeoutSeconds $TimeoutSecs
            $benchJson = $bench.Json
            $benchJsonForSave = $benchJson
            if (-not $benchJsonForSave) { $benchJsonForSave = [PSCustomObject]@{} }
            ($benchJsonForSave | ConvertTo-Json -Depth 20) |
                Set-Content -LiteralPath (Join-Path $rawProfileDir "${rawBase}_bench.json")

            if ($bench.ExitCode -ne 0 -or -not $benchJson) {
                $errLine = ($bench.RawOutput -split "`n" | Where-Object { $_ -match '(?i)error|panic|failed|unsupported' } | Select-Object -First 1)
                if (-not $errLine) {
                    $errLine = if ($bench.TimedOut) { "bench timed out after ${TimeoutSecs}s" } else { "bench failed (exit $($bench.ExitCode))" }
                }
                Write-Host "  failed: $errLine"
                $Results += [PSCustomObject]@{
                    Profile = $profileCfg.Key; ProfileLabel = $profileCfg.Label; Index = $idx + 1
                    Path = $modelPath; File = $modelFile; Name = $modelName; Arch = $arch
                    Status = 'bench_failed'; Mb = $fileMb; LoadMs = $null; Decode = $null; Prefill = $null; Note = $errLine
                }
            } else {
                $loadMs = Get-JsonProp $benchJson @('load_ms') 0
                $decode = Get-JsonProp $benchJson @('summary', 'aggregate_decode_tok_s') 0
                $prefill = Get-JsonProp $benchJson @('summary', 'aggregate_prefill_tok_s') 0
                Write-Host "  ok: load=${loadMs}ms  decode=$(Format-Decimal $decode) tok/s  prefill=$(Format-Decimal $prefill) tok/s"
                $Results += [PSCustomObject]@{
                    Profile = $profileCfg.Key; ProfileLabel = $profileCfg.Label; Index = $idx + 1
                    Path = $modelPath; File = $modelFile; Name = $modelName; Arch = $arch
                    Status = 'supported'; Mb = $fileMb; LoadMs = $loadMs; Decode = $decode; Prefill = $prefill; Note = ''
                }
            }

            Write-Host "  Waiting ${WaitSecs}s to free memory..."
            Start-Sleep -Seconds $WaitSecs
        }

        $Profiles += [PSCustomObject]@{
            Profile = $profileCfg.Key; ProfileLabel = $profileCfg.Label; MetalEnv = $profileCfg.MetalEnv
            RawDir = $rawProfileDir; Runtime = $profileRuntime
        }
    }

    ($Results | ConvertTo-Json -Depth 20) | Set-Content -LiteralPath $ResultsCachePath
    ($Profiles | ConvertTo-Json -Depth 20) | Set-Content -LiteralPath $ProfilesCachePath
}

# ─── Hardware info ────────────────────────────────────────────────────────────

function Get-HardwareInfo {
    $cpu = 'unknown'
    try {
        $cpu = (Get-CimInstance -ClassName Win32_Processor -ErrorAction Stop | Select-Object -First 1 -ExpandProperty Name).Trim()
    } catch {}
    $cores = [System.Environment]::ProcessorCount
    $ramDisplay = 'unknown'
    try {
        $ramBytes = (Get-CimInstance -ClassName Win32_ComputerSystem -ErrorAction Stop).TotalPhysicalMemory
        if ($ramBytes -gt 0) { $ramDisplay = "$([Math]::Round($ramBytes / 1GB)) GB" }
    } catch {}
    $osName = 'Windows'
    $osVer = [System.Environment]::OSVersion.VersionString
    try {
        $os = Get-CimInstance -ClassName Win32_OperatingSystem -ErrorAction Stop
        $osName = $os.Caption.Trim()
        $osVer = $os.Version
    } catch {}
    $rustVer = 'unknown'
    try { $rustVer = ((& rustc --version) -join ' ').Trim() } catch {}
    if (-not $rustVer) { $rustVer = 'unknown' }

    return [PSCustomObject]@{
        Cpu = $cpu; Cores = $cores; RamDisplay = $ramDisplay
        OsName = $osName; OsVer = $osVer; RustVer = $rustVer
        Simd = 'x86 runtime detection (AVX2/FMA when available)'
    }
}

# ─── BENCHMARK.md generation ──────────────────────────────────────────────────

function ConvertTo-MdEscaped {
    param([string]$Text)
    if ($null -eq $Text) { return '' }
    return ($Text -replace '\|', '\|') -replace "`n", ' '
}

function ConvertTo-MdCode {
    param([string]$Text)
    return '`' + ((ConvertTo-MdEscaped $Text) -replace '`', '\`') + '`'
}

function Format-Note {
    param([string]$Note, [int]$Limit = 92)
    $text = ConvertTo-MdEscaped $Note
    if ($text.Length -le $Limit) { return $text }
    return $text.Substring(0, $Limit - 1).TrimEnd() + '…'
}

function Get-StatusText {
    param([string]$Status)
    switch ($Status) {
        'supported' { return 'ok' }
        'bench_failed' { return 'failed' }
        'partially-supported' { return 'partial' }
        default { return 'skip' }
    }
}

function New-BenchmarkMarkdown {
    param($Results, $Profiles, $Hardware)

    $sb = New-Object System.Text.StringBuilder
    $null = $sb.AppendLine('# RustyLLM Benchmark Results')
    $null = $sb.AppendLine()
    $null = $sb.AppendLine("Updated: **$(Get-Date -Format 'yyyy-MM-dd HH:mm')** (local time)")
    $null = $sb.AppendLine()
    $null = $sb.AppendLine('This report was generated on Windows via `bench_models.ps1` (native PowerShell, no bash/python3/jq). Only the CPU backend is covered -- RustyLLM''s Metal path is macOS-only.')
    $null = $sb.AppendLine()
    $null = $sb.AppendLine('## Run Configuration')
    $null = $sb.AppendLine()
    $null = $sb.AppendLine('| Setting | Value |')
    $null = $sb.AppendLine('|---|---|')
    $null = $sb.AppendLine("| Model directory | $(ConvertTo-MdCode $ResolvedModelDir) |")
    $null = $sb.AppendLine("| Prompt | $(ConvertTo-MdEscaped $Prompt) |")
    $null = $sb.AppendLine("| Runs | $BenchRuns x $MaxTokens generated tokens per model |")
    $null = $sb.AppendLine("| Pause | ${WaitSecs} seconds between models |")
    $null = $sb.AppendLine("| Raw JSON | $(ConvertTo-MdCode '.bench_raw/<profile>/') |")
    $null = $sb.AppendLine()
    $null = $sb.AppendLine('## Hardware')
    $null = $sb.AppendLine()
    $null = $sb.AppendLine('| Component | Value |')
    $null = $sb.AppendLine('|---|---|')
    $null = $sb.AppendLine("| CPU | $(ConvertTo-MdEscaped $Hardware.Cpu) |")
    $null = $sb.AppendLine("| Logical cores | $($Hardware.Cores) |")
    $null = $sb.AppendLine("| RAM | $(ConvertTo-MdEscaped $Hardware.RamDisplay) |")
    $null = $sb.AppendLine("| OS | $(ConvertTo-MdEscaped $Hardware.OsName) $(ConvertTo-MdEscaped $Hardware.OsVer) |")
    $null = $sb.AppendLine("| Rust | $(ConvertTo-MdEscaped $Hardware.RustVer) |")
    $null = $sb.AppendLine("| SIMD | $(ConvertTo-MdEscaped $Hardware.Simd) |")
    $null = $sb.AppendLine()

    $resultsByProfile = @{}
    foreach ($row in $Results) {
        if (-not $resultsByProfile.ContainsKey($row.Profile)) { $resultsByProfile[$row.Profile] = @() }
        $resultsByProfile[$row.Profile] += $row
    }

    $null = $sb.AppendLine('## Summary')
    $null = $sb.AppendLine()
    $null = $sb.AppendLine('| Profile | Ok | Failed | Skipped/partial | Best decode | Median decode |')
    $null = $sb.AppendLine('|---|---:|---:|---:|---:|---:|')
    foreach ($profileRow in $Profiles) {
        $rows = @($resultsByProfile[$profileRow.Profile])
        $ok = @($rows | Where-Object { $_.Status -eq 'supported' })
        $failed = @($rows | Where-Object { $_.Status -eq 'bench_failed' })
        $skipped = $rows.Count - $ok.Count - $failed.Count
        $speeds = @($ok | Where-Object { $null -ne $_.Decode } | ForEach-Object { [double]$_.Decode })
        $best = if ($speeds.Count -gt 0) { ($speeds | Measure-Object -Maximum).Maximum } else { $null }
        $median = $null
        if ($speeds.Count -gt 0) {
            $sorted = @($speeds | Sort-Object)
            $mid = [Math]::Floor($sorted.Count / 2)
            $median = if ($sorted.Count % 2 -eq 0) { ($sorted[$mid - 1] + $sorted[$mid]) / 2 } else { $sorted[$mid] }
        }
        $null = $sb.AppendLine("| $(ConvertTo-MdEscaped $profileRow.ProfileLabel) | $($ok.Count) | $($failed.Count) | $skipped | $(Format-Decimal $best) | $(Format-Decimal $median) |")
    }
    $null = $sb.AppendLine()

    $issues = @($Results | Where-Object { $_.Status -ne 'supported' })
    if ($issues.Count -gt 0) {
        $null = $sb.AppendLine('## Support Issues')
        $null = $sb.AppendLine()
        $null = $sb.AppendLine('| Profile | Model | Arch | Status | Reason |')
        $null = $sb.AppendLine('|---|---|:---:|---|---|')
        foreach ($row in $issues) {
            $null = $sb.AppendLine("| $(ConvertTo-MdEscaped $row.ProfileLabel) | $(ConvertTo-MdCode $row.File) | $(ConvertTo-MdEscaped $row.Arch) | $(Get-StatusText $row.Status) | $(Format-Note $row.Note) |")
        }
        $null = $sb.AppendLine()
    }

    $null = $sb.AppendLine('## Profile Details')
    $null = $sb.AppendLine()
    foreach ($profileRow in $Profiles) {
        $rows = @($resultsByProfile[$profileRow.Profile] | Sort-Object Index)
        $null = $sb.AppendLine("### $(ConvertTo-MdEscaped $profileRow.ProfileLabel)")
        $null = $sb.AppendLine()
        $null = $sb.AppendLine('| # | Model | Arch | Status | Size | Load | Decode | Prefill | Note |')
        $null = $sb.AppendLine('|---:|---|:---:|---|---:|---:|---:|---:|---|')
        foreach ($row in $rows) {
            $loadDisplay = if ($null -ne $row.LoadMs) { [string][Math]::Floor([double]$row.LoadMs) } else { '—' }
            $null = $sb.AppendLine("| $($row.Index) | $(ConvertTo-MdCode $row.File) | $(ConvertTo-MdEscaped $row.Arch) | $(Get-StatusText $row.Status) | $($row.Mb) | $loadDisplay | $(Format-Decimal $row.Decode) | $(Format-Decimal $row.Prefill) | $(Format-Note $row.Note) |")
        }
        $null = $sb.AppendLine()

        $ranking = @($rows | Where-Object { $_.Status -eq 'supported' -and $null -ne $_.Decode } | Sort-Object { [double]$_.Decode } -Descending)
        if ($ranking.Count -gt 0) {
            $null = $sb.AppendLine("### $(ConvertTo-MdEscaped $profileRow.ProfileLabel) Decode Ranking")
            $null = $sb.AppendLine()
            $null = $sb.AppendLine('| Rank | Model | Decode | Prefill | Load |')
            $null = $sb.AppendLine('|---:|---|---:|---:|---:|')
            $rank = 1
            foreach ($row in $ranking) {
                $loadDisplay = if ($null -ne $row.LoadMs) { [string][Math]::Floor([double]$row.LoadMs) } else { '—' }
                $null = $sb.AppendLine("| $rank | $(ConvertTo-MdCode $row.File) | $(Format-Decimal $row.Decode) | $(Format-Decimal $row.Prefill) | $loadDisplay |")
                $rank++
            }
            $null = $sb.AppendLine()
        }
    }

    $null = $sb.AppendLine('---')
    $null = $sb.AppendLine()
    $null = $sb.AppendLine('Re-run `bench_models.ps1` (or `.\make.ps1 bench-models`) to refresh this report. Set `$env:MODEL_FILTER`/`$env:MODEL_LIMIT` for a partial sweep.')
    return $sb.ToString()
}

Write-Host ''
Write-Host "Writing $BenchmarkMdPath ..."
$hardware = Get-HardwareInfo
$markdown = New-BenchmarkMarkdown -Results $Results -Profiles $Profiles -Hardware $hardware
Set-Content -LiteralPath $BenchmarkMdPath -Value $markdown -Encoding UTF8
Write-Host "Benchmark report written: $BenchmarkMdPath"

# ─── README link insertion ────────────────────────────────────────────────────

$linkLine = '-> **[Benchmark results](BENCHMARK.md)** - CPU model compatibility/speed for tested models.'
if (Test-Path -LiteralPath $ReadmePath) {
    $readmeText = Get-Content -Raw -LiteralPath $ReadmePath
    if ($readmeText -match [regex]::Escape('BENCHMARK.md')) {
        Write-Host 'README.md already links to BENCHMARK.md.'
    } else {
        # @(...) guards against Get-Content returning a bare string (which the
        # List[string] constructor would then enumerate character-by-character,
        # since string implements IEnumerable<char>) if README.md ever had
        # exactly one line.
        $lines = [System.Collections.Generic.List[string]]@(Get-Content -LiteralPath $ReadmePath)
        $insertAt = $lines.Count
        $sawContent = $false
        for ($i = 0; $i -lt $lines.Count; $i++) {
            $stripped = $lines[$i].Trim()
            if ($stripped -and -not $stripped.StartsWith('#') -and -not $stripped.StartsWith('[![')) {
                $sawContent = $true
            }
            if ($sawContent -and $stripped -eq '') {
                $insertAt = $i + 1
                break
            }
        }
        $lines.Insert($insertAt, '')
        $lines.Insert($insertAt, $linkLine)
        $lines.Insert($insertAt, '')
        Set-Content -LiteralPath $ReadmePath -Value $lines -Encoding UTF8
        Write-Host 'README.md link inserted.'
    }
} else {
    Write-Host 'README.md not found; link insertion skipped.'
}

Write-Host ''
Write-Host '================================================================'
Write-Host 'Done.'
Write-Host "  BENCHMARK.md : $BenchmarkMdPath"
Write-Host "  Raw JSON     : $RawDir\<profile>\"
Write-Host '================================================================'
