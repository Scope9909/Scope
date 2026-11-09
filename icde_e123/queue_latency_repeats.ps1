param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$work = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$out = Join-Path $work 'icde_e123\latency_repeats_v1'
New-Item -ItemType Directory -Path $out -Force | Out-Null
$process = Start-Process -FilePath $Python `
    -ArgumentList '-B', 'icde_e123\run_latency_repeats.py' `
    -WorkingDirectory $work `
    -RedirectStandardOutput (Join-Path $out 'runner.stdout.log') `
    -RedirectStandardError (Join-Path $out 'runner.stderr.log') `
    -WindowStyle Hidden -Wait -PassThru
@{
    completed_at = (Get-Date).ToString('o')
    exit_code = $process.ExitCode
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $out 'exit.json') -Encoding utf8
