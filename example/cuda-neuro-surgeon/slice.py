
import torch

# 假设 inputs 形状为 [B, C, H, W]
inputs = torch.randn(64, 3, 224, 224)
model = torch.nn.Linear(224, 224).cuda()

# 方法一：按每 16 张切
results = []
for sl in torch.split(inputs, 16, dim=0):
    sl = sl.cuda()
    with torch.no_grad():
        out_sl = model(sl)          # 在 GPU 上推理
    out_sl_cpu = out_sl.cpu()       # 拉回到 CPU
    results.append(out_sl_cpu)      # 在 CPU 上累加
    # 释放这次循环的 GPU 内存
    del sl, out_sl
    torch.cuda.empty_cache()
# 最后在 CPU 上拼回完整结果
final = torch.cat(results, dim=0)