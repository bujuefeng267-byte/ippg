# RhythmMamba 官方来源与冻结推理审计

审计日期：2026-09-09。只读核对官方仓库；本审计未下载权重、安装包或运行视频。可以试用作者指定 PURE 跨数据集权重；原始 CUDA 扩展不可用时可做 PyTorch 参考扫描适配，但尚未实测与官方内核的数值一致性。

**唯一预先冻结的选择**：提交 `1533ad21cb4ae0d31e9dbb9afb0b1eefbed54aed`，`PreTrainedModels/PURE_cross_RhythmMamba.pth`。它由作者 [PURE→UBFC 推理 YAML](https://github.com/zizheng-guo/RhythmMamba/blob/1533ad21cb4ae0d31e9dbb9afb0b1eefbed54aed/configs/infer_configs/PURE_UBFC-rPPG_RHYTHMMAMBA.yaml#L103) 明确指定；[固定提交权重](https://raw.githubusercontent.com/zizheng-guo/RhythmMamba/1533ad21cb4ae0d31e9dbb9afb0b1eefbed54aed/PreTrainedModels/PURE_cross_RhythmMamba.pth) 的 Git 元数据为 20,044,634 bytes、blob SHA1 `836e171ef226dc5ca72240f000592d5dbd84b293`。文件 SHA256 待下载方记录；本审计没有验证权重载荷或训练日志。不得按当前视频结果再选择 UBFC_cross 或其他 checkpoint。

## 原作者输入与输出

|环节|必须保留的具体口径|
|---|---|
|视频解码|OpenCV 顺序读到 EOF，BGR→RGB；官方 UBFC reader 不重采样，也不保留原始时间戳|
|裁脸|仓库 Haar XML；只在首帧检测；静态框放大 1.5 倍，按作者裁剪边界；动态检测关闭|
|空间尺寸|128×128，INTER_AREA；裁剪后序列缓冲区是 float64|
|输入标准化|**整段视频一个均值与总体标准差**，涵盖 T/H/W/C，并包含最终会丢弃的尾帧；先标准化，后分片；非逐通道、逐帧或逐片|
|片段|160 帧、不重叠、步长 160；不足 160 的尾帧丢弃；转 float32，输入 [B,160,3,128,128]|
|预测|模型输出 [B,160]；每片沿时间减均值、除 torch.std 默认样本标准差；按数字片号拼接|

依据：[视频 reader](https://github.com/zizheng-guo/RhythmMamba/blob/1533ad21cb4ae0d31e9dbb9afb0b1eefbed54aed/dataset/data_loader/UBFCrPPGLoader.py#L95)、[预处理顺序及裁脸](https://github.com/zizheng-guo/RhythmMamba/blob/1533ad21cb4ae0d31e9dbb9afb0b1eefbed54aed/dataset/data_loader/BaseLoader.py#L219)、[全局标准化](https://github.com/zizheng-guo/RhythmMamba/blob/1533ad21cb4ae0d31e9dbb9afb0b1eefbed54aed/dataset/data_loader/BaseLoader.py#L585)、[测试输出归一化](https://github.com/zizheng-guo/RhythmMamba/blob/1533ad21cb4ae0d31e9dbb9afb0b1eefbed54aed/neural_methods/trainer/RhythmMambaTrainer.py#L126)。

在项目导出中，波形对应原帧号 `chunk_index*160 + sample_index`；尾帧保留缺失状态，不移走、填零或改成另一时间轴。采用项目原视频实际 FPS 时，`time_s=frame_index/FPS`。作者配置固定 FS=30；共享评估使用实际 FPS 是需明示的协议适配。

## 模型接口与不应改写的细节

`RhythmMamba(depth=24, embed_dim=96, mlp_ratio=2, drop_rate=0., drop_path_rate=0.1, ...)`，`forward(x)`；内部 Mamba 为 d_state=48、d_conv=4、expand=2、d_inner=192、dt_rank=6。160 帧经过 stride=2 时域卷积成为 80 个 token，再上采样回 160。一般奇数 D 会得到 `2*floor(D/2)` 输出，本轮固定 D=160。作者 trainer 直接调用默认构造函数。[模型](https://github.com/zizheng-guo/RhythmMamba/blob/1533ad21cb4ae0d31e9dbb9afb0b1eefbed54aed/neural_methods/model/RhythmMamba.py#L258)

- Frequency FFN 的 `einsum('bnc,cc->bnc')` 实际取矩阵对角；不可“修正”为常规稠密矩阵乘法。
- IFFT 后转 float32 会丢弃虚部；保持原行为。
- MTP 是四组重复/移位及分段累加平均，应保留原代码。
- 全局视频标准化、未来两帧差分、片段 FFT 和片段输出标准化都使完整流程离线；SSM 的递推性不使整模型自动成为实时因果模型。[FFN/MTP](https://github.com/zizheng-guo/RhythmMamba/blob/1533ad21cb4ae0d31e9dbb9afb0b1eefbed54aed/neural_methods/model/RhythmMamba.py#L92)

## CUDA 与 PyTorch 参考扫描

作者 [setup.sh](https://github.com/zizheng-guo/RhythmMamba/blob/1533ad21cb4ae0d31e9dbb9afb0b1eefbed54aed/setup.sh#L1)：Python3.8、Torch2.1.2、torchvision0.16.2、torchaudio2.1.2、CUDA12.1；[requirements](https://github.com/zizheng-guo/RhythmMamba/blob/1533ad21cb4ae0d31e9dbb9afb0b1eefbed54aed/requirements.txt#L1)：mamba-ssm2.2.2、causal-conv1d1.4.0、timm0.9.16、einops0.7.0。模型使用 timm 的 trunc_normal_、lecun_normal_、DropPath；to_2tuple 仅导入。历史科学包版本较老；新隔离环境的差异应记录。

仅 `use_fast_path=False` **不够**：扫描仍调用自定义 CUDA；`selective_scan_interface` 还无条件导入扩展。[Mamba.forward](https://github.com/zizheng-guo/RhythmMamba/blob/1533ad21cb4ae0d31e9dbb9afb0b1eefbed54aed/setup/mamba/mamba_ssm/modules/mamba_simple.py#L119)

可适配为同仓 [selective_scan_ref](https://github.com/zizheng-guo/RhythmMamba/blob/1533ad21cb4ae0d31e9dbb9afb0b1eefbed54aed/setup/mamba/mamba_ssm/ops/selective_scan_interface.py#L91)，保留接口：

`selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False)`

同时停用 fused Mamba/causal_conv1d_fn，走原 depthwise Conv1d 的因果截取及 SiLU 分支。适配器须绕开无条件 CUDA 导入，并保持参数名/形状、delta bias+softplus、B/C 广播、D 跳连、z 门控和片段初始零状态。纯 Torch 路径可在 **GPU 或 CPU** 上运行；它属于参考实现适配，不是原论文速度基准。

对当前实数状态，递推为 `s_t=exp(delta_t*A)*s_(t-1)+delta_t*B_t*u_t`，随后 `(sum(C_t*s_t)+D*u_t)*silu(z_t)`。静态代码支持同一数学计算；浮点/整模型 CUDA 一致性仍须测试。建议固定随机/零/压力输入，先核 B=4、192通道、80时刻、48状态的扫描与卷积，再核整模型同权重160帧输出；记录最大/均值绝对误差，不用参考 HR 来选路径。

作者 vendored 包标签为1.2.0.post1，但 Mamba 类31–294行与[上游 v2.2.2](https://github.com/state-spaces/mamba/blob/v2.2.2/mamba_ssm/modules/mamba_simple.py)完全相同；scan interface 整文件也相同（SHA256 `fe606b4c7e81b47bb091cf59dc474aece1112a6ca01eb6f22309090e56030bc4`）。外围区别是 triton.layernorm/module路径和未用的 Block 类。

## 权重加载

作者保存 DataParallel 的 state_dict，测试时直接 torch.load；预期带 module. 前缀，但本审计未读取载荷确认。[Trainer](https://github.com/zizheng-guo/RhythmMamba/blob/1533ad21cb4ae0d31e9dbb9afb0b1eefbed54aed/neural_methods/trainer/RhythmMambaTrainer.py#L133)

集成使用更新的 Torch、显式 `torch.load(path, map_location='cpu', weights_only=True)`，验证 string→tensor 字典；仅在所有 key 都带该前缀时去掉 module.，校验全部 key/shape，再 `load_state_dict(..., strict=True)`。加载失败不要退回 weights_only=False 或 strict=False。[加载文档](https://docs.pytorch.org/docs/2.1/generated/torch.load.html)

作者 Torch2.1.2 是历史复现环境；不能据此宣称安全加载。PyTorch 官方 [GHSA-53q9-r3pm-6pq6](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6) 指 weights_only 漏洞影响至2.5.1，2.6.0修复；采用更新版本并记录适配。

## 与项目评分的区别

作者默认把保留片段拼接后按**整视频**评分；`USE_SMALLER_WINDOW=False` 时配置中的 WINDOW_SIZE=10 不生效。[评估入口](https://github.com/zizheng-guo/RhythmMamba/blob/1533ad21cb4ae0d31e9dbb9afb0b1eefbed54aed/evaluation/metrics.py#L45)

作者由标签 PPG 估算参考 HR，使用去趋势、0.75–2.5Hz 零相位带通及45–150 bpm频域搜索；其 SNR 包含 H1+H2±0.1Hz，还使用20log10功率比。[后处理](https://github.com/zizheng-guo/RhythmMamba/blob/1533ad21cb4ae0d31e9dbb9afb0b1eefbed54aed/evaluation/post_process.py#L105) 这些均不同于项目既定的设备HR参考、10秒/1秒窗、0.7–3.5Hz、H1-only/10log10协议。统一评分应放在预测结束后独立完成，并标为“共同协议比较”，不能当作直接复现论文表格。

## 需要显式记录的边界

- 首帧 Haar 未检出时，官方 fallback 是 [0,0,H,W]，后续却按 [x,y,w,h] 解释；非正方形视频存在宽高互换。须记录 fallback，不可标为人脸成功；修正或改检测器要单列适配。[相关代码](https://github.com/zizheng-guo/RhythmMamba/blob/1533ad21cb4ae0d31e9dbb9afb0b1eefbed54aed/dataset/data_loader/BaseLoader.py#L283)
- 多脸实际选择最大宽度框；消息虽称最大脸，应保留代码行为。
- 常量输出除标准差可能成为 NaN，应拒判并记录。
- 官方后处理 nfft=1e5/sr 为浮点，现代 SciPy 可能存在兼容差异；使用冻结项目评价器不改变神经预测，但须与作者原指标区分。
- 全部结论是源码审计。实际 checkpoint tensor结构、GPU扩展安装、端到端数值一致性与运行速度仍由集成验证；本审计未声称这些已经通过。

完整固定URL、SHA256和接口契约保存在 rhythm_source_audit.json。

