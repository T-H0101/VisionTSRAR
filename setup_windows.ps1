# VisionTSRAR Windows 快速配置脚本（PowerShell）
# 适用于 Windows 本地环境（非 WSL2）
# 使用方法：在 PowerShell（管理员）中运行 .\setup_windows.ps1

# 设置错误处理
$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Green
Write-Host "VisionTSRAR Windows 快速配置脚本" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""

# 检查管理员权限
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "错误：请以管理员身份运行此脚本" -ForegroundColor Red
    Write-Host "右键点击 PowerShell，选择'以管理员身份运行'" -ForegroundColor Yellow
    exit 1
}

Write-Host "✓ 管理员权限检查通过" -ForegroundColor Green
Write-Host ""

# 1. 检查 WSL2 是否已安装
Write-Host "[1/6] 检查 WSL2 状态..." -ForegroundColor Yellow
try {
    $wslStatus = wsl --list --verbose 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ WSL2 已安装" -ForegroundColor Green
        Write-Host $wslStatus
        Write-Host ""
        Write-Host "建议：使用 WSL2 运行 VisionTSRAR（性能更好，兼容性更佳）" -ForegroundColor Cyan
        Write-Host "运行：wsl" -ForegroundColor Cyan
        Write-Host ""
        
        $useWSL = Read-Host "是否要在 WSL2 中运行？(y/n)"
        if ($useWSL -eq 'y' -or $useWSL -eq 'Y') {
            Write-Host ""
            Write-Host "正在启动 WSL2..." -ForegroundColor Yellow
            wsl
            exit 0
        }
    } else {
        throw "WSL2 未安装"
    }
} catch {
    Write-Host "WSL2 未安装或发生错误" -ForegroundColor Yellow
    Write-Host "将继续 Windows 本地配置..." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "提示：建议安装 WSL2 以获得更好的体验" -ForegroundColor Cyan
    Write-Host "运行：wsl --install" -ForegroundColor Cyan
    Write-Host ""
}

# 2. 检查 Python
Write-Host "[2/6] 检查 Python 环境..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python 已安装：$pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "错误：未找到 Python" -ForegroundColor Red
    Write-Host "请安装 Python 3.10: https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}
Write-Host ""

# 3. 检查 Conda
Write-Host "[3/6] 检查 Conda 环境..." -ForegroundColor Yellow
try {
    $condaVersion = conda --version 2>&1
    Write-Host "✓ Conda 已安装：$condaVersion" -ForegroundColor Green
} catch {
    Write-Host "Conda 未安装" -ForegroundColor Yellow
    Write-Host "正在下载 Miniconda..." -ForegroundColor Yellow
    
    $minicondaUrl = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
    $installerPath = "$env:TEMP\Miniconda3-latest-Windows-x86_64.exe"
    
    Invoke-WebRequest -Uri $minicondaUrl -OutFile $installerPath
    
    Write-Host "正在安装 Miniconda..." -ForegroundColor Yellow
    Start-Process -FilePath $installerPath -ArgumentList "/InstallationType=JustMe", "/RegisterPython=0", "/AddToPath=1", "/S", "/D=$env:USERPROFILE\Miniconda3" -Wait
    
    Remove-Item $installerPath
    
    # 刷新环境变量
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    
    Write-Host "✓ Miniconda 安装完成" -ForegroundColor Green
}
Write-Host ""

# 4. 创建 Conda 环境
Write-Host "[4/6] 创建 VisionTSRAR Conda 环境..." -ForegroundColor Yellow
$envExists = conda env list | Select-String "visiontsrar"
if ($envExists) {
    Write-Host "✓ Conda 环境已存在" -ForegroundColor Green
} else {
    conda create -n visiontsrar python=3.10 -y
    Write-Host "✓ Conda 环境创建完成" -ForegroundColor Green
}
Write-Host ""

# 5. 激活环境并安装 PyTorch
Write-Host "[5/6] 安装 PyTorch..." -ForegroundColor Yellow
Write-Host "选择 PyTorch 版本：" -ForegroundColor Yellow
Write-Host "1) CPU 版本（推荐，兼容性最好）"
Write-Host "2) CUDA 版本（如果 NVIDIA GPU 支持）"
Write-Host "3) 跳过（手动安装）"
$choice = Read-Host "请选择 [1/2/3]"

switch ($choice) {
    "1" {
        Write-Host "安装 CPU 版本..." -ForegroundColor Yellow
        conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
        Write-Host "✓ PyTorch CPU 版本安装完成" -ForegroundColor Green
    }
    "2" {
        Write-Host "安装 CUDA 版本..." -ForegroundColor Yellow
        conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
        Write-Host "✓ PyTorch CUDA 版本安装完成" -ForegroundColor Green
    }
    "3" {
        Write-Host "跳过 PyTorch 安装" -ForegroundColor Yellow
    }
    default {
        Write-Host "安装 CPU 版本（默认）" -ForegroundColor Yellow
        conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
    }
}
Write-Host ""

# 6. 安装项目依赖
Write-Host "[6/6] 安装项目依赖..." -ForegroundColor Yellow
if (Test-Path "long_term_tsf\requirements.txt") {
    conda run -n visiontsrar pip install -r long_term_tsf\requirements.txt
    Write-Host "✓ 项目依赖安装完成" -ForegroundColor Green
} else {
    Write-Host "错误：找不到 requirements.txt" -ForegroundColor Red
    exit 1
}
Write-Host ""

# 创建测试脚本
Write-Host "创建测试脚本..." -ForegroundColor Yellow
Copy-Item "test_visiontsrar_etth1_windows.ps1" -Destination "test_visiontsrar.ps1" -ErrorAction SilentlyContinue
Write-Host "✓ 测试脚本创建完成" -ForegroundColor Green
Write-Host ""

# 配置完成
Write-Host "==========================================" -ForegroundColor Green
Write-Host "✓ 配置完成！" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "下一步操作：" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. 激活 Conda 环境：" -ForegroundColor White
Write-Host "   conda activate visiontsrar" -ForegroundColor Cyan
Write-Host ""
Write-Host "2. 运行测试脚本：" -ForegroundColor White
Write-Host "   .\test_visiontsrar.ps1" -ForegroundColor Cyan
Write-Host ""
Write-Host "3. 或手动运行：" -ForegroundColor White
Write-Host "   cd long_term_tsf" -ForegroundColor Cyan
Write-Host "   conda activate visiontsrar" -ForegroundColor Cyan
Write-Host "   python -u run.py --task_name long_term_forecast ..." -ForegroundColor Cyan
Write-Host ""
Write-Host "4. 查看完整文档：" -ForegroundColor White
Write-Host "   cat Windows 迁移指南.md" -ForegroundColor Cyan
Write-Host ""
Write-Host "提示：建议使用 WSL2 以获得更好的兼容性和性能" -ForegroundColor Cyan
Write-Host ""
