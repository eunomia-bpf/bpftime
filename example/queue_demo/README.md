# bpftime Queue Map Demo

这个示例展示了如何在bpftime中使用`BPF_MAP_TYPE_QUEUE`队列map进行eBPF程序与用户态程序之间的数据传输。

## 项目组成

### 1. target.c - 被监听的目标程序

**作用：**
- 作为被监听的目标程序，提供两个函数供eBPF程序监听
- 模拟真实应用程序的函数调用场景
- 定期调用函数，产生可监听的事件

**主要功能：**
- `target_function(int input_value, const char *msg)`: 主要目标函数
  - 接收整数输入和字符串消息
  - 返回输入值的两倍
  - 每次循环都会被调用
- `secondary_function(void)`: 次要目标函数
  - 无参数的简单函数
  - 每5次主函数调用后被调用一次
  - 包含100ms的延时

**运行特点：**
- 启动时打印进程PID和函数地址
- 无限循环调用函数，每秒一次
- 每10次迭代打印分隔线
- 使用`__attribute__((noinline))`确保函数不被内联优化

### 2. uprobe_queue.c - 用户态监控程序

**作用：**
- 从eBPF程序创建的队列中读取事件数据
- 解析并显示函数调用信息
- 提供队列状态监控

**主要功能：**
- **队列数据读取**: 使用`map_pop_elem_syscall()`从队列弹出事件
- **事件数据解析**: 解析包含时间戳、PID、TID、函数ID等信息的事件结构
- **格式化输出**: 将时间戳转换为可读格式，显示详细的函数调用信息
- **队列状态监控**: 使用`map_peek_elem_syscall()`查看队列头部元素状态
- **实时监控**: 持续轮询队列，实时处理新事件

**技术特点：**
- 使用syscall方式调用BPF map操作，bpftime会拦截这些syscall
- 支持优雅退出（Ctrl+C信号处理）
- 区分不同函数的调用事件
- 显示详细的调试信息

### 3. uprobe_queue.bpf.c - eBPF程序

**作用：**
- 监听目标程序的函数调用
- 收集函数调用信息并推送到队列
- 实现uprobe事件处理

**主要功能：**
- **队列Map定义**: 创建容量为64的队列map存储事件数据
- **计数器Map**: 记录全局函数调用次数
- **Uprobe处理器**: 
  - `target_function_entry`: 监听target_function调用
  - `secondary_function_entry`: 监听secondary_function调用
- **事件数据收集**:
  - 时间戳（纳秒精度）
  - 进程和线程ID
  - 函数参数（target_function的input_value）
  - 进程名称
  - 全局调用计数器
- **队列操作**: 使用`bpf_map_push_elem()`将事件推送到队列

## 数据流程

```
目标程序(target) → eBPF程序(uprobe_queue.bpf.c) → 队列Map → 用户态程序(uprobe_queue.c)
     ↓                        ↓                      ↓              ↓
  函数调用              收集事件信息            存储事件         读取并显示
```

**详细流程：**
1. **target程序**启动并定期调用`target_function`和`secondary_function`
2. **eBPF程序**通过uprobe监听这些函数调用
3. **事件收集**：eBPF程序收集调用信息（时间戳、PID、参数等）
4. **队列推送**：使用`bpf_map_push_elem()`将事件数据推送到队列map
5. **用户态读取**：uprobe_queue程序使用`map_pop_elem_syscall()`从队列弹出事件
6. **数据显示**：格式化并显示事件信息，包括函数类型、调用参数、时间戳等

## 事件数据结构

```c
struct event_data {
    uint64_t timestamp;    // 纳秒时间戳
    uint32_t pid;         // 进程ID
    uint32_t tid;         // 线程ID  
    uint32_t counter;     // 全局调用计数器
    uint32_t function_id; // 函数标识符 (1=target_function, 2=secondary_function)
    int32_t input_value;  // target_function的输入参数
    char comm[16];        // 进程名称
};
```

## 运行方式

### 方式1: 使用运行脚本（推荐）
```bash
./run_demo.sh
```

### 方式2: 手动运行
```bash
# 终端1: 启动bpftime server并运行目标程序
sudo bpftime load ./uprobe_queue
sudo bpftime start ./target

# 终端2: 运行用户态监控程序
sudo bpftime attach $(pgrep target) ./uprobe_queue
```

### 方式3: 使用bpftime CLI
```bash
# 编译
make

# 运行
sudo bpftime load ./uprobe_queue
sudo bpftime start ./target &
sudo bpftime attach $(pgrep target) ./uprobe_queue
```

## 队列Map特性

bpftime实现的队列map具有以下特性：

- **FIFO语义**: 先进先出的数据结构
- **线程安全**: 使用互斥锁保护并发访问
- **容量限制**: 本示例设置为64个元素
- **溢出处理**: 队列满时可选择覆盖最旧元素或拒绝新元素
- **原子操作**: 支持原子的push/pop/peek操作

**支持的操作：**
- `bpf_map_push_elem()`: 推送元素到队列尾部
- `bpf_map_pop_elem()`: 从队列头部弹出元素
- `bpf_map_peek_elem()`: 查看队列头部元素（不删除）

## 预期输出

**target程序输出：**
```
Target program started, PID=12345
Functions to be monitored:
  - target_function: 0x401234
  - secondary_function: 0x401567
Starting periodic function calls...
target_function called with input=1, msg=call_1
Main loop iteration 1 completed, result=2
...
```

**uprobe_queue程序输出：**
```
队列Map FD: 4
开始监控队列事件...
按 Ctrl+C 停止

队列状态: 非空 (头部事件: 函数ID=1, 计数器=1)
[14:30:15.123] target_function() 被调用 - PID:12345 TID:12345 输入值:1 计数器:1 进程:target
[14:30:16.124] target_function() 被调用 - PID:12345 TID:12345 输入值:2 计数器:2 进程:target
...
[14:30:20.128] secondary_function() 被调用 - PID:12345 TID:12345 计数器:6 进程:target
本轮处理了 6 个事件
```

## 技术特点

1. **实时事件传输**: 使用队列map实现eBPF到用户态的实时数据传输
2. **多函数监听**: 同时监听多个不同的函数调用
3. **详细事件信息**: 收集丰富的调用上下文信息
4. **线程安全**: 队列操作完全线程安全
5. **高性能**: 基于共享内存的高效数据传输
6. **可扩展**: 易于添加更多监听函数和事件类型

## 故障排除

1. **权限问题**: 确保以sudo权限运行
2. **bpftime未安装**: 确保bpftime已正确安装并在PATH中
3. **编译错误**: 检查依赖项是否安装（libbpf, clang, llvm）
4. **队列满**: 如果事件产生过快，可能导致队列溢出
5. **函数地址**: 确保target程序的函数没有被内联优化

## 依赖项

- bpftime
- libbpf
- clang
- llvm
- Linux内核头文件

这个demo完整展示了bpftime队列map的使用方法，是学习eBPF队列操作的理想起点。 