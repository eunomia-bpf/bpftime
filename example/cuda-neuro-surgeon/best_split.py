import itertools
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH     = "../cuda-probe-test/llama2-13b-chat-hf/model"
TOKENIZER_PATH = "../cuda-probe-test/llama2-13b-chat-hf/tokenizer"
GPU_DEVICE     = "cuda:0"

# 1) 预加载模型和 tokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=False)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# 2) 准备输入（留在 CPU）
prompt = "In a quiet village nestled between two mountains, a young girl named Lila..."
inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)

L = len(model.model.layers)
split_pairs = itertools.combinations(range(1, L), 2)
results = {}

for s1, s2 in split_pairs:
    # 3) 构造 device_map，把中间 s1~s2 层切到 GPU，其余在 CPU
    device_map = {
        **{f"model.layers.{i}": "cpu"    for i in range(0, s1)},
        **{f"model.layers.{i}": GPU_DEVICE for i in range(s1, s2)},
        **{f"model.layers.{i}": "cpu"    for i in range(s2, L)},
        "model.embed_tokens": "cpu",
        "model.norm":          GPU_DEVICE,
        "lm_head":             GPU_DEVICE,
    }
    submodel = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map=device_map,
        torch_dtype=torch.float16,
        trust_remote_code=False,
    )

    # 4) 根据第一个 split 点 s1，决定输入放到哪个 device
    #    如果 s1 < L，就意味着有层跑在 GPU，所以 inputs 需要到 GPU
    target_device = GPU_DEVICE if s1 < L else "cpu"
    inputs_dev = {k: v.to(target_device) for k, v in inputs.items()}

    # 5) 测量延迟
    if target_device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.time()

    _ = submodel.generate(
        **inputs_dev,
        max_new_tokens=128,
        do_sample=False,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    if target_device.startswith("cuda"):
        torch.cuda.synchronize()
    latency = time.time() - t0

    results[(s1, s2)] = latency
    print(f"切分点 ({s1}, {s2}) → 延迟: {latency:.3f} s")

best_pair, best_latency = min(results.items(), key=lambda x: x[1])
print(f"\n最优切分位置: {best_pair}, 延迟: {best_latency:.3f} s")
