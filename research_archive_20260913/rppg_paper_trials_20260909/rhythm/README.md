# RhythmMamba 项目适配试验

使用作者固定 PURE 预训练权重，不训练、不用参考心率挑参数。官方仓库提交为 `1533ad21cb4ae0d31e9dbb9afb0b1eefbed54aed`，权重 SHA256 为 `442ec9ac71bfc299f41e2c7f671216cfa4f6749d8dd264e53f8eb9d8b65380ee`。下载来源和逐文件哈希见 `upstream_download.json`。

这里检验它能否改善现有项目的视频结果，不是作者论文性能或速度的完整复现。论文与仓库：[AAAI 2025 原文](https://ojs.aaai.org/index.php/AAAI/article/view/33204)、[官方代码](https://github.com/zizheng-guo/RhythmMamba)。

## 实现

`backend.py` 原样执行固定官方模型源码，用 AST 提取同一仓库的 `Mamba` 类及 `selective_scan_ref`。适配层将扫描切到作者的纯 PyTorch 参考函数，关闭快速融合及 causal-conv1d 扩展，保留模型参数和原公式。唯一权重通过 `weights_only=True` 和 `strict=True` 加载，原始源码、配置和权重在每次加载前核验哈希。

运行环境为项目中的独立 `.venv-rhythm-trial`，PyTorch 2.6.0+cu124、torchvision 0.21.0、timm 0.9.16、einops 0.7.0。科学计算依赖通过 `.pth` 引用项目原 `.venv`。本试验没有在原环境中替换依赖。设备为 RTX 4060 Laptop GPU。运行时和作者原始 PyTorch 2.1.2 环境不同。

`run_rhythm_trial.py` 使用固定 V2.2（0.10 s）的人脸框，扩至 1.5 倍并裁成 128×128 RGB。只在不超过 0.10 s 的有界缺口中插值框位置；像素始终来自该时刻实际视频帧。较长缺口分段，保留缺失输出。

每段以包括尾帧在内的全部裁脸像素计算单个均值和总体标准差，随后分成不重叠的 160 帧片段，丢弃不足 160 帧的尾段。每个模型输出片段按作者训练器使用样本标准差归一化。保留原视频帧率，不重采样到 30 Hz。拼接后的真实网络输出另存 `raw_network.csv`，按项目固定 42–210 bpm 零相位带通处理，段首尾各遮掉 1.6 s，保存 `waveform.csv`。

心率使用现有冻结读出程序，从实际保存波形计算 10 s 窗、1 s 步长的局部峰值和离线动态规划结果。拒绝窗口的主心率列保持空值。`observed` 和 `interpolated` 只在保留波形区域标记，二者不能被当作算法正确率。

## 与作者流程的差异

作者配置使用首帧 Haar 静态裁脸、整视频标准化、名义 30 Hz 以及整记录评价。本项目使用动态跟踪框、缺口分段、原始帧率和统一逐窗读出。前处理、后处理及窗口支持不同，不能把本试验数字作为论文基准数字。

段归一化、双向滤波及心率动态规划都使用未来信息，因此这是离线结果。160 帧切片存在边界和丢尾，覆盖率可能下降。参考扫描的手工递推、CPU/GPU 和 Mamba 逐步运算检查不能替代与官方融合 CUDA 内核的逐值对比；本次没有验证后者，也不报告官方速度复现。

## 校验及输出

- `backend_test.py` / `backend_verification.json`：6 项后端检查，包括固定权重严格加载及 160 帧 GPU 输出。
- `test_preprocessing.py` / `preprocessing_verification.json`：10 项无 Torch 前处理检查。
- `protocol_before_inference.json`：推理前固定参数、源代码哈希、全部视频身份。
- `results/<视频>/waveform.csv`：处理后波形和有效性标记；`raw_network.csv`：网络原始输出。
- `results/<视频>/heart_rate.csv`：局部谱峰及离线心率；`segment_diagnostics.json`：片段统计量和丢尾长度。
- `run_summary.json`：全六段完成标记，只有全部成功才写出。
- 上层 `evaluation/`：独立参考评分、共同窗口比较、覆盖率与时间偏移敏感性。

在原项目的 WSL 环境运行：

```bash
"/home/fengbujue/项目/rppg识别/.venv-rhythm-trial/bin/python" -B "/mnt/c/Users/15011/Documents/ChatGPT/New project/rppg_paper_trials_20260909/rhythm/backend_test.py"
"/home/fengbujue/项目/rppg识别/.venv-rhythm-trial/bin/python" -B "/mnt/c/Users/15011/Documents/ChatGPT/New project/rppg_paper_trials_20260909/rhythm/run_rhythm_trial.py" --run-all
```

推理脚本拒绝覆盖已有 `results`，以保护这次冻结试验；不要删除已有结果来反复挑参数。另一次试验应使用单独的版本目录和完整输入清单。压缩包不包含用户原视频，重新运行依赖原项目视频及 V2.2 帧跟踪缓存。
