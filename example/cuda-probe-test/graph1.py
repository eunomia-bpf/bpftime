import matplotlib.pyplot as plt

# 数据，单位：纳秒
data = {
    "1B": 1003437566,
    "2B": 1010010397,
    "4B": 1001819102,
    "8B": 998218467,
    "16B": 1009018666,
    "32B": 1003629551,
    "64B": 1007087170,
    "128B": 1009663413,
    "256B": 1001315648,
    "512B": 1001491377,
    "1024B": 1005799896,
    "2048B": 1004782939,
    "4096B": 1019992620,
    "8192B": 1013469587,
    "16384B": 2038442889,
    "32768B": 3049753159,
    "65536B": 6096080585,
    "131072B": 11549020295,
    "262144B": 21867683896,
    "524288B": 47203677076,
    "1048576B": 90190470969,
    "2M": 180424721898,
    "4M": 341843994045
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
iguard_baseline = 1041843994045
plt.axhline(y=iguard_baseline, color='red', linestyle='-', label='iGuard baseline')

# 设置标题和坐标轴标签，同时增大字体
plt.title("End to end latency (100 lookups)", fontsize=16)
plt.xlabel("字节", fontsize=14)
plt.ylabel("纳秒", fontsize=14)

# 横轴采用对数刻度显示（数据跨度较大）
plt.xscale('log')
# 设置横轴刻度标签
plt.xticks(x_vals, x_labels, rotation=45, fontsize=10)
plt.yticks(fontsize=10)

plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('end_to_end_latency.pdf')