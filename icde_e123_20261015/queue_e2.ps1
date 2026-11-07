$ErrorActionPreference = 'Stop'
$work = 'D:\lunwen\June\TLAST-main'
$out = Join-Path $work 'icde_e123_20261015'
$python = Join-Path $work '_runtime_deps\python311_embed\python.exe'
Wait-Process -Id 18000 -ErrorAction SilentlyContinue
$process = Start-Process -FilePath $python `
    -ArgumentList '-B', 'icde_e123_20261015\run_e2.py' `
    -WorkingDirectory $work `
    -RedirectStandardOutput (Join-Path $out 'e2.stdout.log') `
    -RedirectStandardError (Join-Path $out 'e2.stderr.log') `
    -WindowStyle Hidden -Wait -PassThru
@{
    completed_at = (Get-Date).ToString('o')
    exit_code = $process.ExitCode
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $out 'e2_exit.json') -Encoding utf8
