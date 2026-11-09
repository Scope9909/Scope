param(
    [string]$Python = 'python',
    [int]$WaitForPid = 0
)
$ErrorActionPreference = 'Stop'
$work = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$out = Join-Path $work 'icde_e123'
if ($WaitForPid -gt 0) {
    Wait-Process -Id $WaitForPid -ErrorAction SilentlyContinue
}
$process = Start-Process -FilePath $Python `
    -ArgumentList '-B', 'icde_e123\run_e2.py' `
    -WorkingDirectory $work `
    -RedirectStandardOutput (Join-Path $out 'e2.stdout.log') `
    -RedirectStandardError (Join-Path $out 'e2.stderr.log') `
    -WindowStyle Hidden -Wait -PassThru
@{
    completed_at = (Get-Date).ToString('o')
    exit_code = $process.ExitCode
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $out 'e2_exit.json') -Encoding utf8
