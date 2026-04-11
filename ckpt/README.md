# 预训练权重目录

此目录存放 VisionTSRAR 所需的预训练权重文件。

## 权重来源

所有权重托管在 HuggingFace: [`yucornell/RandAR`](https://huggingface.co/yucornell/RandAR)

## 权重文件

| 文件名 | 用途 | 大小 |
|--------|------|------|
| `vq_ds16_c2i.pt` | VQ Tokenizer（向量量化变分自编码器） | ~200MB |
| `rbrar_l_0.3b_c2i.safetensors` | RandAR GPT (0.3B参数，class-to-image) | ~600MB |

## 自动下载

首次运行时，权重会自动从 HuggingFace 下载到此目录。

也可以手动下载：

```bash
# 方法1：使用项目提供的下载脚本
python download_ckpt.py

# 方法2：使用 huggingface-cli
pip install huggingface_hub
huggingface-cli download yucornell/RandAR vq_ds16_c2i.pt --local-dir .
huggingface-cli download yucornell/RandAR rbrar_l_0.3b_c2i.safetensors --local-dir .
```

## 注意事项

- 权重文件较大，请确保网络畅通
- 下载完成后无需手动移动文件，保持在此目录即可
- 如果下载中断，重新运行即可（已下载部分会自动跳过）
