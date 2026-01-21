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

# a) текущая директория = корень репо
if (Test-Path ".\\configs") {
    Write-Ok "Вы в корне репозитория (найден .\\configs)"
} else {
    Write-Fail "Не найден .\\configs — вероятно, вы не в корне репозитория" @(
        "cd C:\\gerychhh_\\pc_agent"
    )
}

# b) configs/training_config.yaml
if (Test-Path ".\\configs\\training_config.yaml") {
    Write-Ok "Найден configs\\training_config.yaml"
} else {
    Write-Fail "Не найден configs\\training_config.yaml" @(
        "copy .\\configs\\training_config.example.yaml .\\configs\\training_config.yaml"
    )
}

# c) external/piper-sample-generator/generate_samples.py
$defaultGeneratorPath = ".\\external\\piper-sample-generator\\generate_samples.py"
if (Test-Path $defaultGeneratorPath) {
    Write-Ok "Найден $defaultGeneratorPath"
} else {
    Write-Fail "Не найден $defaultGeneratorPath" @(
        "git clone https://github.com/rhasspy/piper-sample-generator .\\external\\piper-sample-generator"
    )
}

# d) python import openwakeword
$owwLocation = & python -c "import openwakeword; print(openwakeword.__file__)" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Ok "openwakeword импортируется: $owwLocation"
} else {
    Write-Fail "openwakeword не импортируется" @(
        "python -m pip install -r requirements.txt",
        "python -m pip install openwakeword"
    )
    $owwLocation = $null
}

# e) train.py пропатчен (guard на false_positive_validation_data_path)
if ($owwLocation) {
    $trainPath = & python -c "import openwakeword.train as t; print(t.__file__)" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "Не удалось определить путь к openwakeword.train.py" @(
            "python -c \"import openwakeword.train as t; print(t.__file__)\""
        )
    } elseif (-not (Test-Path $trainPath)) {
        Write-Fail "Файл train.py не найден: $trainPath" @(
            "python -m pip install openwakeword"
        )
    } else {
        $trainText = Get-Content -Path $trainPath -Raw
        if ($trainText -match "fp_path = config.get\(\"false_positive_validation_data_path\"\)") {
            Write-Ok "train.py содержит guard для false_positive_validation_data_path"
        } else {
            Write-Fail "train.py не содержит guard для false_positive_validation_data_path" @(
                "python .\\scripts\\patch_openwakeword_train.py"
            )
        }
    }
} else {
    Write-Fail "Проверка патча пропущена (openwakeword не установлен)" @(
        "python -m pip install -r requirements.txt"
    )
}

# f) piper_sample_generator_path из training_config.yaml
$configPath = ".\\configs\\training_config.yaml"
$piperPath = Get-ConfigValue -path $configPath -key "piper_sample_generator_path"
if (-not $piperPath) {
    Write-Fail "Не найден ключ piper_sample_generator_path в configs\\training_config.yaml" @(
        "Откройте configs\\training_config.yaml и добавьте piper_sample_generator_path: \"external\\\\piper-sample-generator\""
    )
} else {
    $generatorCheck = Join-Path -Path $piperPath -ChildPath "generate_samples.py"
    if (Test-Path $generatorCheck) {
        Write-Ok "piper_sample_generator_path указывает на generate_samples.py"
    } else {
        Write-Fail "По piper_sample_generator_path не найден generate_samples.py: $generatorCheck" @(
            "Исправьте piper_sample_generator_path в configs\\training_config.yaml",
            "git clone https://github.com/rhasspy/piper-sample-generator $piperPath"
        )
    }
}

# g) rir_paths и background_paths из training_config.yaml
$rirPaths = Get-ConfigListValues -path $configPath -key "rir_paths"
if ($rirPaths.Count -eq 0) {
    Write-Fail "В configs\\training_config.yaml нет значений для rir_paths" @(
        "Добавьте папку с RIR, например: rir_paths: [\"data\\\\rir\"]"
    )
} else {
    foreach ($rirPath in $rirPaths) {
        if (Test-Path $rirPath) {
            Write-Ok "rir_paths существует: $rirPath"
        } else {
            Write-Fail "Путь из rir_paths не найден: $rirPath" @(
                "Создайте папку с RIR: New-Item -ItemType Directory $rirPath",
                "Или исправьте rir_paths в configs\\training_config.yaml"
            )
        }
    }
}

$backgroundPaths = Get-ConfigListValues -path $configPath -key "background_paths"
if ($backgroundPaths.Count -eq 0) {
    Write-Fail "В configs\\training_config.yaml нет значений для background_paths" @(
        "Добавьте папку с фонами, например: background_paths: [\"data\\\\negative\"]"
    )
} else {
    foreach ($backgroundPath in $backgroundPaths) {
        if (Test-Path $backgroundPath) {
            Write-Ok "background_paths существует: $backgroundPath"
        } else {
            Write-Fail "Путь из background_paths не найден: $backgroundPath" @(
                "Создайте папку с фонами: New-Item -ItemType Directory $backgroundPath",
                "Или исправьте background_paths в configs\\training_config.yaml"
            )
        }
    }
}

Write-Host "==============================" -ForegroundColor Cyan
if ($issues -eq 0) {
    Write-Host "Готово: проблем не найдено." -ForegroundColor Green
    exit 0
} else {
    Write-Host "Готово: найдено проблем — $issues" -ForegroundColor Red
    exit 1
}
