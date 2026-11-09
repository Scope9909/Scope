param(
    [ValidateSet('PEMS03', 'PEMS07', 'PEMS08', 'PEMSBAY')]
    [string[]]$Datasets = @('PEMS03', 'PEMS07', 'PEMS08'),
    [int]$Seed = 1,
    [string]$PythonExe = 'python'
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Stages = @(
    'backbone',
    'lowrank',
    'exact',
    'reranker_lowrank',
    'head',
    'confidence',
    'final_joint'
)

# Seeds are repeatable: retrieval / candidate / index-scaler caches are built from
# data only (no learned parameters), so they are shared across seeds and the
# cache stage below is a no-op for any seed other than the one that created them.

function Get-RunDirectory {
    param([string]$Dataset, [string]$Stage)
    $RunId = "paper_$($Dataset.ToLower())_s${Seed}_${Stage}"
    return Join-Path $ProjectRoot (
        "EXP_experiments/TLAST_improved/${Dataset}/" +
        "TLAST_improved_${Dataset}_12to12_exp${RunId}"
    )
}

function Invoke-Stage {
    param([string]$Dataset, [string]$Stage)
    $RunId = "paper_$($Dataset.ToLower())_s${Seed}_${Stage}"
    $Config = "cross_dataset/$($Dataset.ToLower())_s${Seed}_${Stage}.json"
    $RunDirectory = Get-RunDirectory -Dataset $Dataset -Stage $Stage
    $Best = Join-Path $RunDirectory 'best.pth'
    $Complete = Join-Path $RunDirectory '.pipeline_complete'
    if ((Test-Path -LiteralPath $Best) -and (Test-Path -LiteralPath $Complete)) {
        Write-Host "[SKIP] completed: $RunId"
        return
    }
    if (Test-Path -LiteralPath $RunDirectory) {
        throw (
            "Incomplete run directory exists: $RunDirectory. " +
            'Archive or remove it before resuming so a partial best.pth is not reused.'
        )
    }
    Write-Host "[RUN] $RunId"
    & $PythonExe run.py --config_file $Config --seed $Seed --run_id $RunId
    if ($LASTEXITCODE -ne 0) {
        throw "Experiment failed with exit code ${LASTEXITCODE}: $RunId"
    }
    if (-not (Test-Path -LiteralPath $Best)) {
        throw "Experiment ended without best.pth: $RunId"
    }
    Set-Content -LiteralPath $Complete -Value (Get-Date -Format 'o')
}

function Ensure-RetrievalCaches {
    param([string]$Dataset)
    $TimeCache = Join-Path $ProjectRoot "datasets/$Dataset/${Dataset}_time.npz"
    $IndexCache = Join-Path $ProjectRoot (
        "datasets/$Dataset/${Dataset}_12to12_index_scaler.npz"
    )
    if (-not (Test-Path -LiteralPath $TimeCache)) {
        throw "Backbone did not create time cache: $TimeCache"
    }
    if (-not (Test-Path -LiteralPath $IndexCache)) {
        throw "Backbone did not create index/scaler cache: $IndexCache"
    }

    $BaseRetrieval = Join-Path $ProjectRoot (
        "datasets/$Dataset/${Dataset}_12to12_analog_k8.npz"
    )
    if (-not (Test-Path -LiteralPath $BaseRetrieval)) {
        Write-Host "[CACHE] base retrieval: $Dataset"
        & $PythonExe compute_analog_retrieval_cache.py `
            --dataset_name $Dataset `
            --input_length 12 `
            --predict_length 12 `
            --k 8 `
            --min_valid_history 6 `
            --aggregation mean `
            --output $BaseRetrieval
        if ($LASTEXITCODE -ne 0) {
            throw "Base retrieval cache failed with exit code ${LASTEXITCODE}: $Dataset"
        }
    } else {
        Write-Host "[SKIP] base retrieval cache exists: $Dataset"
    }

    $Retrieval = Join-Path $ProjectRoot (
        "datasets/$Dataset/${Dataset}_12to12_analog_k8_exact_mean.npz"
    )
    if (-not (Test-Path -LiteralPath $Retrieval)) {
        Write-Host "[CACHE] exact retrieval: $Dataset"
        & $PythonExe compute_analog_exact_match_cache.py `
            --dataset_name $Dataset `
            --input_length 12 `
            --predict_length 12 `
            --min_valid_history 6 `
            --base_cache $BaseRetrieval `
            --output $Retrieval
        if ($LASTEXITCODE -ne 0) {
            throw "Exact retrieval cache failed with exit code ${LASTEXITCODE}: $Dataset"
        }
    } else {
        Write-Host "[SKIP] exact retrieval cache exists: $Dataset"
    }

    $Candidate = Join-Path $ProjectRoot (
        "datasets/$Dataset/${Dataset}_12to12_analog_k8_candidate_indices.npz"
    )
    if (-not (Test-Path -LiteralPath $Candidate)) {
        Write-Host "[CACHE] candidates: $Dataset"
        & $PythonExe compute_analog_candidate_index_cache.py `
            --dataset_name $Dataset `
            --input_length 12 `
            --predict_length 12 `
            --k 8 `
            --min_valid_history 6 `
            --reference_cache $Retrieval `
            --output $Candidate
        if ($LASTEXITCODE -ne 0) {
            throw "Candidate cache failed with exit code ${LASTEXITCODE}: $Dataset"
        }
    } else {
        Write-Host "[SKIP] candidate cache exists: $Dataset"
    }
}

Push-Location $ProjectRoot
try {
    & $PythonExe -c "import torch, einops; assert torch.cuda.is_available(); print('training_python_ok')"
    if ($LASTEXITCODE -ne 0) {
        throw 'Selected Python cannot import dependencies or access CUDA.'
    }
    & $PythonExe make_cross_dataset_experiment_configs.py --datasets $Datasets --seed $Seed
    if ($LASTEXITCODE -ne 0) {
        throw "Config generation failed with exit code $LASTEXITCODE"
    }

    foreach ($Dataset in $Datasets) {
        Invoke-Stage -Dataset $Dataset -Stage 'backbone'
        Ensure-RetrievalCaches -Dataset $Dataset
        foreach ($Stage in $Stages | Where-Object { $_ -ne 'backbone' }) {
            Invoke-Stage -Dataset $Dataset -Stage $Stage
        }
        & $PythonExe collect_paper_results.py --dataset $Dataset
        if ($LASTEXITCODE -ne 0) {
            throw "Result collection failed with exit code ${LASTEXITCODE}: $Dataset"
        }
    }
}
finally {
    Pop-Location
}
