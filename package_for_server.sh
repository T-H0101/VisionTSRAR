# 快速打包脚本
# 使用方法：bash package_for_server.sh

set -e

echo "========================================="
echo "打包 VisionTSRAR 用于服务器部署"
echo "========================================="

PROJECT_NAME="VisionTSRAR"
ZIP_FILE="${PROJECT_NAME}_deploy.zip"

# 获取当前目录
CURRENT_DIR=$(pwd)

echo "正在打包..."

# 打包项目（排除不必要的文件）
zip -r $ZIP_FILE . \
  -x "*.git/*" \
  -x "*.gitignore" \
  -x "long_term_tsf/dataset/*" \
  -x "long_term_tsf/logs/*" \
  -x "long_term_tsf/checkpoints/*" \
  -x "long_term_tsf/results/*" \
  -x "__pycache__/*" \
  -x "*.pyc" \
  -x "*.ipynb_checkpoints/*" \
  -x ".DS_Store" \
  -x "Thumbs.db"

echo "✓ 打包完成：$ZIP_FILE"
echo ""
echo "文件大小："
ls -lh $ZIP_FILE
echo ""
echo "上传到服务器："
echo "scp $ZIP_FILE your_username@server_ip:/home/your_username/"
echo ""
echo "在服务器上："
echo "1. unzip $ZIP_FILE"
echo "2. cd $PROJECT_NAME"
echo "3. bash deploy/setup_server.sh"
echo ""
echo "========================================="
