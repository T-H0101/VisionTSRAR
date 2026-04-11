# VisionTSRAR 快速开始指南

## 🚀 一键配置

### WSL2 用户（推荐）

```bash
# 1. 在 WSL2 终端执行
cd ~/VisionTSRAR

# 2. 运行配置脚本
bash setup_wsl2.sh

# 3. 测试运行
./test_visiontsrar.sh
```

### Windows 本地用户

```powershell
# 1. 在 PowerShell（管理员）执行
cd C:\Projects\VisionTSRAR

# 2. 运行配置脚本
.\setup_windows.ps1

# 3. 激活环境
conda activate visiontsrar

# 4. 测试运行
.\test_visiontsrar.ps1
```

## 📋 可用脚本

| 脚本 | 平台 | 用途 |
|------|------|------|
| `setup_wsl2.sh` | WSL2 | 自动配置 WSL2 环境 |
| `setup_windows.ps1` | Windows | 自动配置 Windows 环境 |
| `test_visiontsrar.sh` | WSL2 | ETTh1 基准测试 |
| `test_visiontsrar_etth1_windows.ps1` | Windows | ETTh1 基准测试 |

## 📖 完整文档

- **Windows 迁移指南**: [`Windows 迁移指南.md`](./Windows 迁移指南.md)
  - WSL2 vs Windows 选择
  - VSCode WSL 集成
  - Conda 环境配置
  - Git 配置和提交
  - 性能预期

- **Ckpt 目录说明**: [`ckpt/README.md`](./ckpt/README.md)
  - 目录结构说明
  - 权重管理最佳实践

## 🎯 快速参考

### Conda 环境

```bash
# 激活环境
conda activate visiontsrar

# 退出环境
conda deactivate

# 查看已安装的包
conda list
```

### VSCode WSL 连接

```bash
# 在 WSL2 终端执行
cd ~/VisionTSRAR
code .
```

### Git 操作

```bash
# 查看状态
git status

# 添加文件
git add .

# 提交
git commit -m "Your message"

# 推送
git push origin main
```

## ⚙️ 手动运行

```bash
cd long_term_tsf
conda activate visiontsrar

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model VisionTSRAR \
    --data ETTh1 \
    --seq_len 96 \
    --pred_len 96 \
    --batch_size 8 \
    --train_epochs 1 \
    --learning_rate 0.001 \
    --use_gpu 1
```

## 🔧 故障排查

### WSL2 无法连接

```powershell
# 在 PowerShell（管理员）执行
wsl --shutdown
wsl
```

### Conda 环境找不到

```bash
# 重新创建环境
conda create -n visiontsrar python=3.10 -y
conda activate visiontsrar
```

### 权重文件缺失

```bash
cd ckpt
python3 download_ckpt.py
```

## 📊 性能预期

| 配置 | 推理速度 | 训练速度 |
|------|---------|---------|
| WSL2 (GPU) | ~5 秒/样本 | ~2 小时/epoch |
| Windows (GPU) | ~6 秒/样本 | ~2.2 小时/epoch |
| CPU | ~30 秒/样本 | ~20 小时/epoch |

## 💡 提示

1. **推荐使用 WSL2**：更好的兼容性，原生 Linux 性能
2. **使用 VSCode WSL 扩展**：无缝开发体验
3. **不要提交权重文件**：使用 `.gitignore` 自动忽略
4. **使用 Conda 环境**：避免依赖冲突

## 📞 需要帮助？

查看完整文档：[`Windows 迁移指南.md`](./Windows 迁移指南.md)

---

**最后更新**: 2024-XX-XX  
**维护者**: VisionTSRAR Team
