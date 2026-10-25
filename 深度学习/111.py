# -*- coding: utf-8 -*-
"""环境自检脚本：确认本机 PyTorch 的 CUDA 可用性

用途：在部署前快速判断推理会跑在 GPU 还是 CPU 上。
注意：本项目默认使用 CPU 推理即可（详见 README「已知限制」）。
"""
import torch


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"torch 版本        : {torch.__version__}")
    print(f"CUDA 是否可用     : {torch.cuda.is_available()}")
    print(f"将使用推理设备    : {device}")

    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    print(f"张量已创建于      : {x.device}")


if __name__ == '__main__':
    main()