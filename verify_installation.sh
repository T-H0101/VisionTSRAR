#!/bin/bash
# VisionTSRAR 环境验证脚本
# 用于检查安装是否成功

echo "=========================================="
echo "VisionTSRAR 环境验证"
echo "=========================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 检查 Conda 环境
echo -e "${YELLOW}[1/5] 检查 Conda 环境...${NC}"
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo -e "${GREEN}✓ Conda 环境：$CONDA_DEFAULT_ENV${NC}"
else
    echo -e "${RED}✗ 未激活 Conda 环境${NC}"
    echo "请先运行：conda activate visiontsrar"
    exit 1
fi
echo ""

# 检查 Python 版本
echo -e "${YELLOW}[2/5] 检查 Python 版本...${NC}"
python_version=$(python --version 2>&1)
echo -e "${GREEN}✓ $python_version${NC}"
echo ""

# 检查 PyTorch
echo -e "${YELLOW}[3/5] 检查 PyTorch...${NC}"
python -c "
import torch
print(f'✓ PyTorch 版本：{torch.__version__}')
print(f'✓ CUDA 可用：{torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ CUDA 版本：{torch.version.cuda}')
    print(f'✓ GPU 数量：{torch.cuda.device_count()}')
" 2>&1 || {
    echo -e "${RED}✗ PyTorch 检查失败${NC}"
    exit 1
}
echo ""

# 检查关键依赖
echo -e "${YELLOW}[4/5] 检查关键依赖...${NC}"
python -c "
import einops
import numpy
import pandas
import scipy
import sklearn
import matplotlib
print('✓ 所有关键依赖已安装')
" 2>&1 || {
    echo -e "${RED}✗ 依赖检查失败${NC}"
    echo "请运行：cd long_term_tsf && pip install -r requirements.txt"
    exit 1
}
echo ""

# 检查权重文件
echo -e "${YELLOW}[5/5] 检查权重文件...${NC}"
if [ -d "ckpt" ]; then
    ckpt_files=$(find ckpt -name "*.pt" -o -name "*.safetensors" 2>/dev/null | wc -l)
    if [ "$ckpt_files" -gt 0 ]; then
        echo -e "${GREEN}✓ 找到 $ckpt_files 个权重文件${NC}"
        find ckpt -name "*.pt" -o -name "*.safetensors" 2>/dev/null | while read file; do
            size=$(du -h "$file" | cut -f1)
            echo "  - $file ($size)"
        done
    else
        echo -e "${YELLOW}⚠️  未找到权重文件${NC}"
        echo "请运行：cd ckpt && python3 download_ckpt.py"
    fi
else
    echo -e "${RED}✗ ckpt 目录不存在${NC}"
fi
echo ""

# 总结
echo "=========================================="
echo -e "${GREEN}✓ 环境验证完成！${NC}"
echo "=========================================="
echo ""
echo "下一步："
echo "1. 如果没有权重文件，下载："
echo "   cd ckpt && python3 download_ckpt.py"
echo ""
echo "2. 运行测试："
echo "   ./test_visiontsrar.sh"
echo ""
echo "3. 或手动运行："
echo "   cd long_term_tsf"
echo "   python -u run.py --task_name long_term_forecast ..."
echo ""
