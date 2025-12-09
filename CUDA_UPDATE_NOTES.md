# CUDA 12.4 适配说明

## 更新内容

本项目已从 CUDA 12.8 适配到 CUDA 12.4，以兼容 Slurm 系统的 cuda-cudnn/12.4-9.1.1 模块。

### 主要变更

1. **PyTorch 版本调整**
   - 从 `torch>=2.8.0` 降级到 `torch>=2.6.0`
   - 原因：PyTorch 2.8.0 不提供 cu124 版本，仅支持 cu126/cu128/cu129

2. **CUDA 版本更新**
   - PyTorch CUDA 版本：从 cu128 改为 cu124
   - NVIDIA CUDA 运行时库：12.4.127
   - NVIDIA CUDA NVRTC：12.4.127
   - NVIDIA CUDA CUPTI：12.4.127

3. **修改的文件**
   - `pyproject.toml`：更新依赖版本和 CUDA 索引
   - `uv.lock`：重新生成以反映新的依赖

## 安装步骤

### 1. 删除旧的虚拟环境（如果存在）

```bash
rm -rf .venv
```

### 2. 在计算节点上重新安装依赖

提交作业或在交互式节点上执行：

```bash
# 加载必要的模块
module load cuda-cudnn/12.4-9.1.1

# 创建新的虚拟环境并安装依赖
uv sync --extra gpu

# 激活虚拟环境
source .venv/bin/activate

# 验证 CUDA 可用性
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

### 3. 编译 Rust 分词器

```bash
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
```

## 验证安装

运行以下命令验证环境配置正确：

```bash
python -c "import torch; print(torch.__version__); print(f'CUDA: {torch.cuda.is_available()}')"
```

预期输出：
```
2.6.0+cu124
CUDA: True
```

## 兼容性说明

- **系统 CUDA 版本**：12.4.x（通过 module load cuda-cudnn/12.4-9.1.1）
- **PyTorch CUDA 版本**：12.4（cu124）
- **Python 版本**：3.10+
- **GPU 驱动要求**：支持 CUDA 12.4 或更高版本

## 已更新的脚本

- `base_train.sh`：已包含 `module load cuda-cudnn/12.4-9.1.1`
- `speedrun.sh`：无需修改，使用虚拟环境即可
- `environment_dataset_prepare.sh`：无需修改

## 常见问题

### Q: 为什么降级到 PyTorch 2.6.0？
A: PyTorch 2.8.0 不提供 CUDA 12.4 (cu124) 的预编译版本，只有 cu126/cu128/cu129。由于系统 CUDA 驱动为 12.4，无法使用更高版本的 CUDA 运行时。

### Q: PyTorch 2.6.0 和 2.8.0 有什么区别？
A: 对于本项目的训练任务，PyTorch 2.6.0 完全满足需求。主要的 API 和功能保持兼容。

### Q: 如何确认环境配置正确？
A: 在计算节点上运行：
```bash
nvidia-smi  # 检查 GPU 和驱动版本
python -c "import torch; print(torch.cuda.is_available())"  # 应该输出 True
```

## 参考资料

- [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/)
- [PyTorch CUDA 12.4 Index](https://download.pytorch.org/whl/cu124/)