$ErrorActionPreference = "Continue"

$issues = 0

function Write-Ok([string]$message) {
    Write-Host "OK: $message" -ForegroundColor Green
}

function Write-Fail([string]$message, [string[]]$fixes) {
    Write-Host "FAIL: $message" -ForegroundColor Red
    if ($fixes) {
        Write-Host "  Fix:" -ForegroundColor Yellow
        foreach ($fix in $fixes) {
            Write-Host "    $fix" -ForegroundColor Yellow
        }
    }
    $script:issues++
}

function Write-Warn([string]$message, [string[]]$fixes) {
    Write-Host "WARN: $message" -ForegroundColor Yellow
    if ($fixes) {
        Write-Host "  Hint:" -ForegroundColor Yellow
        foreach ($fix in $fixes) {
            Write-Host "    $fix" -ForegroundColor Yellow
        }
    }
}

function Get-ConfigValue([string]$path, [string]$key) {
    if (-not (Test-Path $path)) {
        return $null
    }

    $lines = Get-Content -Path $path
    foreach ($line in $lines) {
        if ($line -match "^\s*$key\s*:\s*(.+)$") {
            $value = $Matches[1]
            $value = ($value -replace "#.*$", "").Trim()
            $value = $value.Trim('"').Trim("'")
            return $value
        }
    }

    return $null
}

function Get-ConfigListValues([string]$path, [string]$key) {
    if (-not (Test-Path $path)) {
        return @()
    }

    $lines = Get-Content -Path $path
    $values = @()
    $inBlock = $false
    foreach ($line in $lines) {
        if ($line -match "^\s*$key\s*:\s*$") {
            $inBlock = $true
            continue
        }

        if ($inBlock) {
            if ($line -match "^\s*-\s*(.+)$") {
                $value = $Matches[1]
                $value = ($value -replace "#.*$", "").Trim()
                $value = $value.Trim('"').Trim("'")
                if ($value) {
                    $values += $value
                }
                continue
            }

            if ($line -match "^\s*\S") {
                break
            }
        }
    }

    return $values
}

Write-Host "openWakeWord training doctor" -ForegroundColor Cyan
Write-Host "==============================" -ForegroundColor Cyan

# a) current directory is repo root
if (Test-Path ".\\configs") {
    Write-Ok "Repo root detected (.\\configs found)"
} else {
    Write-Fail "Repo root not found (.\\configs missing)" @(
        "cd C:\\gerychhh_\\pc_agent"
    )
}

# b) configs/training_config.yaml
if (Test-Path ".\\configs\\training_config.yaml") {
    Write-Ok "configs\\training_config.yaml exists"
} else {
    Write-Fail "configs\\training_config.yaml is missing" @(
        "copy .\\configs\\training_config.example.yaml .\\configs\\training_config.yaml"
    )
}

# c) external/piper-sample-generator/generate_samples.py
$defaultGeneratorPath = ".\\external\\piper-sample-generator\\generate_samples.py"
if (Test-Path $defaultGeneratorPath) {
    Write-Ok "Found $defaultGeneratorPath"
} else {
    Write-Fail "Missing $defaultGeneratorPath" @(
        "git clone https://github.com/rhasspy/piper-sample-generator .\\external\\piper-sample-generator"
    )
}

# d) python import openwakeword
$owwLocation = & python -c "import openwakeword; print(openwakeword.__file__)" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Ok "openwakeword import OK: $owwLocation"
} else {
    Write-Fail "openwakeword import failed" @(
        "python -m pip install -r requirements.txt",
        "python -m pip install openwakeword"
    )
    $owwLocation = $null
}

# e) train.py patch guard
if ($owwLocation) {
    $trainPath = & python -c "import openwakeword.train as t; print(t.__file__)" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "Unable to locate openwakeword.train.py" @(
            "python -c \"import openwakeword.train as t; print(t.__file__)\""
        )
    } elseif (-not (Test-Path $trainPath)) {
        Write-Fail "train.py not found: $trainPath" @(
            "python -m pip install openwakeword"
        )
    } else {
        $trainText = Get-Content -Path $trainPath -Raw
        if ($trainText -match "fp_path = config.get\(\"false_positive_validation_data_path\"\)") {
            Write-Ok "train.py has false_positive_validation_data_path guard"
        } else {
            Write-Fail "train.py missing false_positive_validation_data_path guard" @(
                "python .\\scripts\\patch_openwakeword_train.py"
            )
        }
    }
} else {
    Write-Fail "Patch check skipped (openwakeword not installed)" @(
        "python -m pip install -r requirements.txt"
    )
}

# f) piper_sample_generator_path from training_config.yaml
$configPath = ".\\configs\\training_config.yaml"
$piperPath = Get-ConfigValue -path $configPath -key "piper_sample_generator_path"
if (-not $piperPath) {
    Write-Fail "Missing piper_sample_generator_path in configs\\training_config.yaml" @(
        "Edit configs\\training_config.yaml and set piper_sample_generator_path: \"external\\\\piper-sample-generator\""
    )
} else {
    $generatorCheck = Join-Path -Path $piperPath -ChildPath "generate_samples.py"
    if (Test-Path $generatorCheck) {
        Write-Ok "piper_sample_generator_path points to generate_samples.py"
    } else {
        Write-Fail "generate_samples.py not found at: $generatorCheck" @(
            "Fix piper_sample_generator_path in configs\\training_config.yaml",
            "git clone https://github.com/rhasspy/piper-sample-generator $piperPath"
        )
    }
}

# g) rir_paths and background_paths from training_config.yaml
$rirPaths = Get-ConfigListValues -path $configPath -key "rir_paths"
if ($rirPaths.Count -eq 0) {
    Write-Fail "No entries found for rir_paths" @(
        "Set rir_paths to an existing folder (can reuse data\\negative if you do not have RIR files)"
    )
} else {
    foreach ($rirPath in $rirPaths) {
        if (Test-Path $rirPath) {
            Write-Ok "rir_paths exists: $rirPath"
        } else {
            Write-Fail "rir_paths path missing: $rirPath" @(
                "Create folder: New-Item -ItemType Directory $rirPath",
                "Or update rir_paths in configs\\training_config.yaml"
            )
        }
    }
}

$backgroundPaths = Get-ConfigListValues -path $configPath -key "background_paths"
if ($backgroundPaths.Count -eq 0) {
    Write-Fail "No entries found for background_paths" @(
        "Set background_paths to an existing folder (data\\negative is typical)"
    )
} else {
    foreach ($backgroundPath in $backgroundPaths) {
        if (Test-Path $backgroundPath) {
            Write-Ok "background_paths exists: $backgroundPath"
        } else {
            Write-Fail "background_paths path missing: $backgroundPath" @(
                "Create folder: New-Item -ItemType Directory $backgroundPath",
                "Or update background_paths in configs\\training_config.yaml"
            )
        }
    }
}

# h) check generated clips exist
$outputDir = Get-ConfigValue -path $configPath -key "output_dir"
$modelName = Get-ConfigValue -path $configPath -key "model_name"
if ($outputDir -and $modelName) {
    $modelDir = Join-Path -Path $outputDir -ChildPath $modelName
    $positiveTestDir = Join-Path -Path $modelDir -ChildPath "positive_test"
    if (Test-Path $positiveTestDir) {
        $positiveCount = (Get-ChildItem -Path $positiveTestDir -Filter "*.wav" -ErrorAction SilentlyContinue).Count
        if ($positiveCount -eq 0) {
            Write-Warn "No generated positive_test clips found" @(
                "Run training with --generate_clips before --augment_clips/--train_model"
            )
        } else {
            Write-Ok "Generated positive_test clips: $positiveCount"
        }
    } else {
        Write-Warn "positive_test folder missing" @(
            "Run: python -m openwakeword.train --training_config configs\\training_config.yaml --generate_clips"
        )
    }
}

Write-Host "==============================" -ForegroundColor Cyan
if ($issues -eq 0) {
    Write-Host "Done: no problems found." -ForegroundColor Green
    exit 0
} else {
    Write-Host "Done: problems found = $issues" -ForegroundColor Red
    exit 1
}
