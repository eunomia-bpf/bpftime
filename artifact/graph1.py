import matplotlib.pyplot as plt
font = {'size': 20}
plt.rc('font', **font)
# 数据，单位：纳秒
data = {
    "1B": 1003437,
    "2B": 1010010,
    "4B": 1001819,
    "8B": 998218,
    "16B": 1009018,
    "32B": 1003629,
    "64B": 1007087,
    "128B": 1009663,
    "256B": 1001315,
    "512B": 1001491,
    "1024B": 1005799,
    "2048B": 1004782,
    "4096B": 1019992,
    "8192B": 1013469,
    "16384B": 2038442,
    "32768B": 3049753,
    "65536B": 6096080,
    "131072B": 11549020,
    "262144B": 21867683,
    "524288B": 47203677,
    "1048576B": 90190470,
    "2M": 180424721,
    "4M": 341843995
}

# 将横轴标签转换为数值（字节）
# 如果标签以"B"结尾，直接转换；如果以"M"结尾，则乘以1048576（1M=1048576字节）
x_vals = []
x_labels = []
y_vals = []
for label, latency in data.items():
    if label.endswith("B"):
        val = int(label[:-1])
    elif label.endswith("M"):
        val = int(label[:-1]) * 1048576
    else:
        val = int(label)
    x_vals.append(val)
    x_labels.append(label)
    y_vals.append(latency)

# 设置图形尺寸，增加宽度以防止文字重叠
plt.figure(figsize=(14,7))

# 绘制数据点和折线
plt.plot(x_vals, y_vals, marker='o', linestyle='-', label='Latency')

# 绘制 iGuard baseline 的实线
iguard_baseline = 1041843994
plt.axhline(y=iguard_baseline, color='red', linestyle='-', label='iGuard baseline')

# 设置标题和坐标轴标签，同时增大字体
plt.xlabel("Bytes", )
plt.ylabel("ns", )

# 横轴采用对数刻度显示（数据跨度较大）
plt.xscale('log')
# 设置横轴刻度标签
plt.xticks(x_vals, x_labels, rotation=45, )

plt.legend()
plt.tight_layout()
plt.savefig('end_to_end_latency.pdf')