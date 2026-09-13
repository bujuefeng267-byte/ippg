截至 2026-09-11，优先落地两个有真实公开权重的候选：本轮已经接入的 EfficientPhys，以及可作下一轮的 FacePhys。选择依据是本地可运行和可审核，不是声称它们分别代表所有基准上的最高准确率。

| 候选 | 官方证据与获取 | 输入和落地情况 |
|---|---|---|
| EfficientPhys，WACV 2023 | [论文](https://openaccess.thecvf.com/content/WACV2023/html/Liu_EfficientPhys_Enabling_Simple_Fast_and_Accurate_Camera-Based_Cardiac_Measurement_WACV_2023_paper.html)、[官方代码](https://github.com/ubicomplab/rPPG-Toolbox)、[唯一 PURE 权重](https://raw.githubusercontent.com/ubicomplab/rPPG-Toolbox/b7500b848f84ad7f86e277b4612563b69f4f88f9/final_model_release/PURE_EfficientPhys.pth)，8.66 MB；Responsible AI Source Code License v1.1 | 官方配置 30 Hz、72×72 RGB、180 帧、全局 Standardized，输出对应 DiffNormalized 标签；现有 Torch GPU 环境即可，无需新训练。当前目录已实现并测试适配。 |
| FacePhys，2025-12 预印本 | [论文](https://arxiv.org/abs/2512.06275)、作者 [FacePhys-Release](https://github.com/KegangWangCCNU/FacePhys-Release)、[ONNX 权重](https://raw.githubusercontent.com/KegangWangCCNU/FacePhys-Release/main/weights/model.onnx) 3,159,827 bytes 与 [state.gz](https://raw.githubusercontent.com/KegangWangCCNU/FacePhys-Release/main/weights/state.gz) 109,582 bytes，公开 HEAD 响应已核查；本轮未下载/执行它 | 官方示例 (T,36,36,3)、float32，逐帧输入 `dt=1/fps`，默认 30 Hz，加载初始化状态并连续传递。也可由作者 open-rppg 的 `FacePhys.rlap` 输入 uint8 RGB 人脸序列。需额外 ONNX Runtime，或使用作者的 LiteRT/JAX 运行时；正式试验前还需冻结提交、模型哈希及精确裁脸/像素尺度。 |

FacePhys 官方发布明确使用 RLAP 预训练，许可证是 MIT 加隐私保护附加条款，包含本地推理要求；这不是没有附加条件的 MIT。公开模型目录与 `FacePhys/FacePhys` 的云 API 示例是不同交付形式。本项目不调用上传视频的 API。[发布说明与许可证](https://github.com/KegangWangCCNU/FacePhys-Release)

FacePhys 论文 RLAP 跨数据集结果是 MMPD MAE 5.30、RMSE 10.0 bpm，PURE MAE 0.24、UBFC 0.43 bpm；PURE 含头部运动，MMPD 包含多种活动与光照。这些是作者整套协议的结果，不能直接当成当前运动视频的预期误差。它同样未消除运动挑战：MMPD 误差明显大于实验室数据。[论文表 4](https://arxiv.org/html/2512.06275v1#S4)

EfficientPhys 原论文包含 UBFC、PURE、MMSE 跨数据集验证；其中 EfficientPhys-C 的表 1 MAE 为 UBFC 1.14、PURE 1.33 bpm，但这张表的训练协议和本次唯一 PURE checkpoint 不能混为一谈。论文还有独立 PURE-only→UBFC 协议。本次必须以六个视频的实际输出评价，不能搬论文数值证明已提升。[WACV 原文](https://openaccess.thecvf.com/content/WACV2023/papers/Liu_EfficientPhys_Enabling_Simple_Fast_and_Accurate_Camera-Based_Cardiac_Measurement_WACV_2023_paper.pdf)

最新运动方向还包括 2026-07 的 **CanonicalPhys**：用眼角和嘴角四点单应变换建立稳定人脸坐标，再引入光照权重、跨 ROI 一致性与 POS 蒸馏。官方代码公开 UBFC/MMPD checkpoint，论文使用 72×72、128 帧，并报告 UBFC→MMPD MAE 从匹配 FactorizePhys 的 13.69±0.21 降至 10.91±0.81 bpm。它需要额外逐帧人脸点和姿态缓存，本轮未列为第三个实施候选，也没有执行；论文自己报告大姿态、遮挡和部分迁移任务的退步，不能把几何归一化当成必然改善。[论文](https://arxiv.org/abs/2607.15995)、[官方代码](https://github.com/infraface/CanonicalPhys)

既有记录必须保留：`rppg_paper_trials_20260909/rhythm/README.md` 已说明原 PURE RhythmMamba 固定提交 `1533ad21cb4ae0d31e9dbb9afb0b1eefbed54aed`，动态 128×128 人脸框、160 帧不重叠且丢尾，按原视频 FPS 推理。旧 data1 的 MAE 为 81.43 bpm；这个失败不能拿来证明所有新模型无效，也不能不改实验设计地重复同一权重并称作最新升级。PhysMamba 需要自定义 CUDA 扩展，PhysFormer 更重；两者未在本轮再次下载/安装。[RhythmMamba 官方仓库](https://github.com/zizheng-guo/RhythmMamba)、[PhysMamba 官方仓库](https://github.com/Chaoqi31/PhysMamba)

现在不必先训练新模型：可以先用固定公开权重作无训练基线。若希望对当前相机、活动和人群稳定优化，再收集有同步参考的新视频做训练/验证/测试分离；必须按受试者和整段录制划分，不能把同一视频的重叠十秒窗分到训练与测试两边。当前六组已被反复查看，适合开发回归，不能独立证明泛化。仅有逐秒心率适合心率层的监督，不能替代可靠同步的原始 PPG/ECG 波形来验证形态恢复。
