#!/bin/bash
# VisionTSRAR WSL2 快速配置脚本
# 适用于 Windows WSL2 (Ubuntu 22.04) 环境
# 使用方法：bash setup_wsl2.sh

set -e  # 遇到错误立即退出

echo "=========================================="
echo "VisionTSRAR WSL2 快速配置脚本"
echo "=========================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查是否在 WSL2 中运行
if ! grep -qi "microsoft\|wsl" /proc/version 2>/dev/null; then
    echo -e "${RED}错误：此脚本只能在 WSL2 环境中运行${NC}"
    echo "请在 Windows 中打开 WSL2 终端后运行此脚本"
    exit 1
fi

echo -e "${GREEN}✓ 检测到 WSL2 环境${NC}"
echo ""

# 1. 系统更新
echo -e "${YELLOW}[1/8] 更新系统包...${NC}"
sudo apt update && sudo apt upgrade -y
echo -e "${GREEN}✓ 系统更新完成${NC}"
echo ""

# 2. 安装基础依赖
echo -e "${YELLOW}[2/8] 安装基础依赖...${NC}"
sudo apt install -y \
    python3 \
    python3-pip \
    git \
    wget \
    curl \
    build-essential

# 尝试安装 GUI 库（WSL2 无头模式可能不需要）
echo "安装 GUI 支持库（可选，失败不影响）..."
sudo apt install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    2>/dev/null || echo "⚠️  GUI 库安装跳过（WSL2 无头模式正常）"

echo -e "${GREEN}✓ 基础依赖安装完成${NC}"
echo ""

# 3. 安装 Miniconda
echo -e "${YELLOW}[3/8] 安装 Miniconda...${NC}"
if [ ! -d "$HOME/miniconda3" ]; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    rm /tmp/miniconda.sh
    
    # 初始化 conda
    source ~/miniconda3/bin/activate
    conda init bash
    echo -e "${GREEN}✓ Miniconda 安装完成${NC}"
else
    echo -e "${GREEN}✓ Miniconda 已安装${NC}"
fi
echo ""

# 4. 创建 Conda 环境
echo -e "${YELLOW}[4/8] 创建 VisionTSRAR Conda 环境...${NC}"
source ~/miniconda3/bin/activate
if conda env list | grep -q "visiontsrar"; then
    echo -e "${GREEN}✓ Conda 环境已存在${NC}"
else
    conda create -n visiontsrar python=3.10 -y
    echo -e "${GREEN}✓ Conda 环境创建完成${NC}"
fi
echo ""

# 5. 激活环境
echo -e "${YELLOW}[5/8] 激活 Conda 环境...${NC}"
conda activate visiontsrar
echo -e "${GREEN}✓ Conda 环境已激活${NC}"
echo ""

# 6. 安装 PyTorch
echo -e "${YELLOW}[6/8] 安装 PyTorch...${NC}"
echo "选择 PyTorch 版本："
echo "1) CPU 版本（推荐，兼容性最好）"
echo "2) ROCm 版本（如果 AMD GPU 支持）"
echo "3) CUDA 版本（如果 NVIDIA GPU 支持）"
echo "4) 跳过（手动安装）"
read -p "请选择 [1/2/3/4]: " pytorch_choice

case $pytorch_choice in
    1)
        echo "安装 CPU 版本..."
        conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
        echo -e "${GREEN}✓ PyTorch CPU 版本安装完成${NC}"
        ;;
    2)
        echo "安装 ROCm 版本..."
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
        echo -e "${GREEN}✓ PyTorch ROCm 版本安装完成${NC}"
        ;;
    3)
        echo "安装 CUDA 版本..."
        conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
        echo -e "${GREEN}✓ PyTorch CUDA 版本安装完成${NC}"
        ;;
    4)
        echo -e "${YELLOW}跳过 PyTorch 安装${NC}"
        ;;
    *)
        echo -e "${RED}无效选择，安装 CPU 版本${NC}"
        conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
        ;;
esac
echo ""

# 7. 安装项目依赖
echo -e "${YELLOW}[7/8] 安装项目依赖...${NC}"
if [ -d "long_term_tsf" ]; then
    cd long_term_tsf
    # 使用 Conda 环境中的 pip 安装其他依赖（不包括 torch）
    pip install -r requirements.txt
    cd ..
    echo -e "${GREEN}✓ long_term_tsf 依赖安装完成${NC}"
else
    echo -e "${RED}错误：找不到 long_term_tsf 目录${NC}"
    echo "请确保在项目根目录运行此脚本"
    exit 1
fi
echo ""

# 8. 下载预训练权重
echo -e "${YELLOW}[8/9] 下载预训练权重...${NC}"
if [ -d "ckpt" ]; then
    cd ckpt
    if python3 download_ckpt.py; then
        cd ..
        echo -e "${GREEN}✓ 预训练权重下载完成${NC}"
    else
        cd ..
        echo -e "${YELLOW}⚠️  权重下载失败，可稍后手动下载：${NC}"
        echo "   cd ckpt && python3 download_ckpt.py"
    fi
else
    echo -e "${RED}错误：找不到 ckpt 目录${NC}"
    exit 1
fi
echo ""

# 9. 创建测试脚本
echo -e "${YELLOW}[8/8] 创建测试脚本...${NC}"
cat > test_visiontsrar.sh << 'EOF'
#!/bin/bash
# VisionTSRAR 测试脚本（WSL2）

cd long_term_tsf
conda activate visiontsrar

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_96_96 \
    --model VisionTSRAR \
    --data ETTh1 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --periodicity 24 \
    --norm_const 0.4 \
    --align_const 0.4 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 8 \
    --num_workers 0 \
    --train_epochs 1 \
    --learning_rate 0.001 \
    --use_gpu 1 \
    --gpu 0 \
    --rar_arch rar_l_0.3b \
    --num_inference_steps 88 \
    --position_order raster \
    --temperature 1.0 \
    --top_k 0 \
    --top_p 1.0 \
    2>&1 | tee ../result/visiontsrar/ETTh1_96_96.log
EOF
chmod +x test_visiontsrar.sh
echo -e "${GREEN}✓ 测试脚本创建完成${NC}"
echo ""

# 配置完成
echo "=========================================="
echo -e "${GREEN}✓ 配置完成！${NC}"
echo "=========================================="
echo ""
echo "下一步操作："
echo "1. 快速测试（推荐）："
echo "   ./test_visiontsrar.sh"
echo ""
echo "2. 手动运行："
echo "   cd long_term_tsf"
echo "   conda activate visiontsrar"
echo "   python3 run.py --task_name long_term_forecast \\"
echo "                  --is_training 0 \\"
echo "                  --model VisionTSRAR \\"
echo "                  --data ETTm1 \\"
echo "                  --seq_len 96 \\"
echo "                  --pred_len 96 \\"
echo "                  --use_gpu 0"
echo ""
echo "3. 使用 VSCode 连接："
echo "   code ."
echo ""
echo "4. 查看完整文档："
echo "   cat docs/Windows 迁移指南.md"
echo ""
echo -e "${GREEN}祝你使用愉快！🚀${NC}"