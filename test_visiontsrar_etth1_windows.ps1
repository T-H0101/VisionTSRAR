# VisionTSRAR Windows PowerShell 测试脚本
# 适用于 Windows 本地环境（非 WSL2）
# 使用方法：.\test_visiontsrar_etth1_windows.ps1

# 设置错误处理
$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Green
Write-Host "VisionTSRAR Windows 测试脚本" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""

# 检查是否在正确的目录
if (-not (Test-Path "long_term_tsf")) {
    Write-Host "错误：请在项目根目录运行此脚本" -ForegroundColor Red
    Write-Host "当前目录：$PWD" -ForegroundColor Yellow
    exit 1
}

Write-Host "✓ 目录检查通过" -ForegroundColor Green
Write-Host ""

# 切换到 long_term_tsf 目录
Write-Host "进入 long_term_tsf 目录..." -ForegroundColor Yellow
Set-Location long_term_tsf

# 检查 Conda 环境
Write-Host "检查 Conda 环境..." -ForegroundColor Yellow
$condaEnv = $env:CONDA_DEFAULT_ENV
if ([string]::IsNullOrEmpty($condaEnv)) {
    Write-Host "警告：未激活 Conda 环境" -ForegroundColor Yellow
    Write-Host "请先运行：conda activate visiontsrar" -ForegroundColor Yellow
    Write-Host ""
} else {
    Write-Host "✓ 当前 Conda 环境：$condaEnv" -ForegroundColor Green
}
Write-Host ""

# 创建结果目录
$resultDir = "..\result\visiontsrar"
if (-not (Test-Path $resultDir)) {
    Write-Host "创建结果目录：$resultDir" -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $resultDir | Out-Null
}

# 运行测试
Write-Host "开始运行 VisionTSRAR 测试..." -ForegroundColor Green
Write-Host ""

python -u run.py `
    --task_name long_term_forecast `
    --is_training 1 `
    --root_path ./dataset/ETT-small/ `
    --data_path ETTh1.csv `
    --model_id ETTh1_96_96 `
    --model VisionTSRAR `
    --data ETTh1 `
    --features M `
    --seq_len 96 `
    --label_len 48 `
    --pred_len 96 `
    --enc_in 7 `
    --dec_in 7 `
    --c_out 7 `
    --periodicity 24 `
    --norm_const 0.4 `
    --align_const 0.4 `
    --des 'Exp' `
    --itr 1 `
    --batch_size 8 `
    --num_workers 0 `
    --train_epochs 1 `
    --learning_rate 0.001 `
    --use_gpu 1 `
    --gpu 0 `
    --rar_arch rar_l_0.3b `
    --num_inference_steps 88 `
    --position_order raster `
    --temperature 1.0 `
    --top_k 0 `
    --top_p 1.0 `
    2>&1 | Tee-Object ..\result\visiontsrar\ETTh1_96_96_windows.log

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "测试完成！" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "日志文件：..\result\visiontsrar\ETTh1_96_96_windows.log" -ForegroundColor Cyan
Write-Host ""
Write-Host "下一步：" -ForegroundColor Yellow
Write-Host "1. 查看日志：cat ..\result\visiontsrar\ETTh1_96_96_windows.log" -ForegroundColor White
Write-Host "2. 修改参数后重新运行" -ForegroundColor White
Write-Host "3. 查看完整文档：cat ..\Windows 迁移指南.md" -ForegroundColor White
Write-Host ""
