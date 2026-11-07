param([string]$PythonExe = 'python')

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Dataset = 'PEMSBAY'
$IndexCache = Join-Path $ProjectRoot 'datasets/PEMSBAY/PEMSBAY_12to12_index_scaler_712.npz'
$BaseRetrieval = Join-Path $ProjectRoot 'datasets/PEMSBAY/PEMSBAY_12to12_analog_k8_712.npz'
$Retrieval = Join-Path $ProjectRoot 'datasets/PEMSBAY/PEMSBAY_12to12_analog_k8_exact_mean_712.npz'
$Candidate = Join-Path $ProjectRoot 'datasets/PEMSBAY/PEMSBAY_12to12_analog_k8_candidate_indices_712.npz'
$Stages = @('backbone', 'lowrank', 'exact', 'reranker_lowrank', 'head', 'confidence', 'final_joint')

function Invoke-Checked {
    param([scriptblock]$Command, [string]$Failure)
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Failure (exit code $LASTEXITCODE)"
    }
}

function Invoke-Stage {
    param([string]$Stage)
    $RunId = "paper_pemsbay712_s1_$Stage"
    $Config = "cross_dataset_712/pemsbay712_s1_$Stage.json"
    $RunDirectory = Join-Path $ProjectRoot "EXP_experiments/TLAST_improved/PEMSBAY/TLAST_improved_PEMSBAY_12to12_exp$RunId"
    $Best = Join-Path $RunDirectory 'best.pth'
    $Complete = Join-Path $RunDirectory '.pipeline_complete'
    if ((Test-Path -LiteralPath $Best) -and (Test-Path -LiteralPath $Complete)) {
        Write-Host "[SKIP] completed: $RunId"
        return
    }
    if (Test-Path -LiteralPath $RunDirectory) {
        throw "Incomplete run directory exists: $RunDirectory"
    }
    Write-Host "[RUN] $RunId"
    & $PythonExe run.py --config_file $Config --seed 1 --run_id $RunId
    if ($LASTEXITCODE -ne 0) {
        throw "Experiment failed: $RunId (exit code $LASTEXITCODE)"
    }
    if (-not (Test-Path -LiteralPath $Best)) {
        throw "Experiment ended without best.pth: $RunId"
    }
    Set-Content -LiteralPath $Complete -Value (Get-Date -Format 'o')
}

Push-Location $ProjectRoot
try {
    & $PythonExe -c "import torch, einops; assert torch.cuda.is_available(); print('training_python_ok')"
    if ($LASTEXITCODE -ne 0) { throw 'Python dependencies or CUDA unavailable.' }

    Invoke-Checked { & $PythonExe make_pemsbay_712_configs.py } 'Config generation failed'
    if (-not (Test-Path -LiteralPath $IndexCache)) {
        Invoke-Checked {
            & $PythonExe prepare_pemsbay_712_split.py --output $IndexCache
        } '7:1:2 split cache generation failed'
    }

    Invoke-Stage -Stage 'backbone'

    if (-not (Test-Path -LiteralPath $BaseRetrieval)) {
        Invoke-Checked {
            & $PythonExe compute_analog_retrieval_cache.py --dataset_name $Dataset --input_length 12 --predict_length 12 --k 8 --min_valid_history 6 --aggregation mean --index_scaler_path $IndexCache --output $BaseRetrieval
        } 'Base retrieval cache generation failed'
    }
    if (-not (Test-Path -LiteralPath $Retrieval)) {
        Invoke-Checked {
            & $PythonExe compute_analog_exact_match_cache.py --dataset_name $Dataset --input_length 12 --predict_length 12 --min_valid_history 6 --index_scaler_path $IndexCache --base_cache $BaseRetrieval --output $Retrieval
        } 'Exact-match cache generation failed'
    }
    if (-not (Test-Path -LiteralPath $Candidate)) {
        Invoke-Checked {
            & $PythonExe compute_analog_candidate_index_cache.py --dataset_name $Dataset --input_length 12 --predict_length 12 --k 8 --min_valid_history 6 --index_scaler_path $IndexCache --reference_cache $Retrieval --output $Candidate
        } 'Candidate cache generation failed'
    }

    foreach ($Stage in $Stages | Where-Object { $_ -ne 'backbone' }) {
        Invoke-Stage -Stage $Stage
    }
    Write-Host '[DONE] PEMS-BAY 70/10/20 pipeline completed.'
}
finally {
    Pop-Location
}
