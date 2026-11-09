param(
    [ValidateSet('core', 'ablations', 'wo', 'all')]
    [string]$Mode = 'core',
    [int[]]$Seeds = @(1),
    [string]$PythonExe = 'python'
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

function Invoke-PaperStage {
    param(
        [int]$Seed,
        [string]$Stage
    )

    $RunId = "paper_pems04_s${Seed}_${Stage}"
    $Config = "paper/pems04_s${Seed}_${Stage}.json"
    $Best = Join-Path $ProjectRoot (
        "EXP_experiments/TLAST_improved/PEMS04/" +
        "TLAST_improved_PEMS04_12to12_exp${RunId}/best.pth"
    )
    $RunDirectory = Split-Path -Parent $Best
    $Complete = Join-Path $RunDirectory '.pipeline_complete'
    $IsWoStage = $Stage.StartsWith('wo_')
    if ($IsWoStage) {
        if ((Test-Path -LiteralPath $Best) -and
            (Test-Path -LiteralPath $Complete)) {
            Write-Host "[SKIP] completed: $RunId"
            return
        }
        if (Test-Path -LiteralPath $RunDirectory) {
            throw (
                "Incomplete w/o run directory exists: $RunDirectory. " +
                'Archive it before resuming; a partial checkpoint must not be reused.'
            )
        }
    }
    elseif (Test-Path -LiteralPath $Best) {
        Write-Host "[SKIP] completed: $RunId"
        return
    }

    Write-Host "[RUN] $RunId"
    & $PythonExe run.py --config_file $Config --seed $Seed --run_id $RunId
    if ($LASTEXITCODE -ne 0) {
        throw "Experiment failed with exit code ${LASTEXITCODE}: $RunId"
    }
    if (-not (Test-Path -LiteralPath $Best)) {
        throw "Experiment ended without best.pth: $RunId"
    }
    if ($IsWoStage) {
        Set-Content -LiteralPath $Complete -Value (Get-Date -Format 'o')
    }
}

Push-Location $ProjectRoot
try {
    & $PythonExe -c "import torch, einops; print('training_python_ok')"
    if ($LASTEXITCODE -ne 0) {
        throw (
            "The selected Python cannot import torch and einops. " +
            "Pass the same interpreter used for the original TLAST runs " +
            "with -PythonExe 'C:\path\to\python.exe'."
        )
    }

    & $PythonExe make_paper_experiment_configs.py --seeds $Seeds
    if ($LASTEXITCODE -ne 0) {
        throw "Config generation failed with exit code $LASTEXITCODE"
    }

    $CoreStages = @(
        'lowrank',
        'exact',
        'reranker_lowrank',
        'head',
        'confidence',
        'final_joint'
    )
    $AblationStages = @(
        'reranker_only',
        'direct_retrieval',
        'final_direct_joint'
    )
    $WoStages = @(
        'wo_confidence',
        'wo_head_confidence',
        'wo_head',
        'wo_reranker_head',
        'wo_reranker_confidence',
        'wo_reranker'
    )

    foreach ($Seed in $Seeds) {
        if ($Mode -in @('core', 'all')) {
            foreach ($Stage in $CoreStages) {
                Invoke-PaperStage -Seed $Seed -Stage $Stage
            }
        }
        if ($Mode -in @('ablations', 'all')) {
            foreach ($Stage in $AblationStages) {
                Invoke-PaperStage -Seed $Seed -Stage $Stage
            }
        }
        if ($Mode -in @('wo', 'all')) {
            foreach ($Stage in $WoStages) {
                Invoke-PaperStage -Seed $Seed -Stage $Stage
            }
        }
    }

    & $PythonExe collect_paper_results.py --prefix paper_pems04
    if ($LASTEXITCODE -ne 0) {
        throw "Result collection failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
