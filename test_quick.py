import torch
import torch.nn.functional as F
from visiontsrar.models_rar import RARWrapper

print("加载模型...")
wrapper = RARWrapper("rar_l_0.3b", "ln", "./ckpt", load_ckpt=False)
print("模型加载完成")

print("测试 VQ Decoder 梯度流...")
B, H, W = 2, 224, 224
image_input = torch.randn(B, 3, H, W, requires_grad=True)
recon, rar_loss = wrapper(image_input, 64)

mse_loss = F.mse_loss(recon, torch.randn_like(recon))
total_loss = rar_loss.mean() + 0.001 * mse_loss
total_loss.backward()

decoder_grads = [(n, p.grad.norm().item()) for n, p in wrapper.vq_tokenizer.named_parameters() if "decoder" in n.lower() and p.grad is not None]
print(f"VQ Decoder 有梯度的参数数量: {len(decoder_grads)}")
if decoder_grads:
    print("VQ Decoder 梯度正常")
    for n, g in decoder_grads[:3]:
        print(f"  {n}: {g:.6f}")
else:
    print("VQ Decoder 无梯度!")

rar_grads = [(n, p.grad.norm().item()) for n, p in wrapper.rar_gpt.named_parameters() if p.grad is not None]
print(f"RAR GPT 有梯度的参数数量: {len(rar_grads)}")