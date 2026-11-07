$ErrorActionPreference = 'Stop'
$work = 'D:\lunwen\June\TLAST-main'
$out = Join-Path $work 'icde_e123_20261015\latency_repeats_v1'
$python = Join-Path $work '_runtime_deps\python311_embed\python.exe'
New-Item -ItemType Directory -Path $out -Force | Out-Null
$process = Start-Process -FilePath $python `
    -ArgumentList '-B', 'icde_e123_20261015\run_latency_repeats.py' `
    -WorkingDirectory $work `
    -RedirectStandardOutput (Join-Path $out 'runner.stdout.log') `
    -RedirectStandardError (Join-Path $out 'runner.stderr.log') `
    -WindowStyle Hidden -Wait -PassThru
@{
    completed_at = (Get-Date).ToString('o')
    exit_code = $process.ExitCode
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $out 'exit.json') -Encoding utf8
