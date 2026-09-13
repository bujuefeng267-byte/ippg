本目录是 **EfficientPhys 固定预训练权重的独立试验分支**。它直接从视频的人脸裁图生成网络输出，并与原项目共用评价时间窗；不能提前视为准确率升级，也不是作者论文成绩的复现。现有 V25 的像素跟踪、局部筛选和回退代码均保留在原分支，神经分支只复用其几何框，没有消费三步光度处理后的 RGB。

唯一模型来自 [rPPG-Toolbox 官方仓库](https://github.com/ubicomplab/rPPG-Toolbox)，固定提交 `b7500b848f84ad7f86e277b4612563b69f4f88f9`。权重是 [PURE_EfficientPhys.pth](https://raw.githubusercontent.com/ubicomplab/rPPG-Toolbox/b7500b848f84ad7f86e277b4612563b69f4f88f9/final_model_release/PURE_EfficientPhys.pth)，8,659,174 bytes，SHA256 `e65a962e07bcac32a668e6acb9f8ed43cdb1b01cfb97262654dc5b55c0cf3a49`。所有下载来源、大小及哈希在 `upstream_manifest.json`，许可证原文在 `upstream/LICENSE`，名称为 Responsible Artificial Intelligence Source Code License v1.1，不能写成 MIT。

| 环节 | 本轮固定处理 |
|---|---|
| 输入身份 | 原 data1–6 完整视频 SHA256 与 V24 frame_trace 校验，保留原总帧数及实际 FPS |
| 人脸裁图 | 原几何框放大 1.5 倍，INTER_AREA 到 72×72 RGB；至多 0.10 s 的内部缺口仅插值框，读取当时实际像素；长缺口分段 |
| 模型时间轴 | 从整片时间零点定义 30 Hz 网格；高帧率先用 Kaiser polyphase 抗混叠，再按实际时间做微小相位修正，不把 180 Hz 帧直接当成 30 Hz |
| 标准化 | 每个连续有效段全部裁图共享一个均值与总体标准差，与官方全局标准化形式一致，分段是适配差异 |
| 网络 | 原样 EfficientPhys 源码，frame_depth=10、img_size=72；权重 weights_only=True、strict=True |
| 分片 | 每片 181 个实际模型输入样本产生 180 个相邻帧差分输出；步长 90，补充最后完整片覆盖尾部，不复制末帧制造零差分 |
| 波形 | 先对重叠片的原始导数作正 Hann 加权，再每段单次累加；差分 i 属于 [t_i,t_i+1]，累加值放到 t_i+1。零初值仅表示未知加性常数 |
| 后处理 | 30 Hz 上固定三阶零相位 0.7–3.5 Hz 带通及段标准化，每段首尾各遮掉 1.6 s；再按实际时间回采样到原视频时轴 |
| 心率 | 重新读取已保存 waveform.csv，按冻结 V25 evidence DP、10 s/1 s 窗、42–210 bpm 生成；不读取参考值 |

官方 [PURE→UBFC 配置](https://github.com/ubicomplab/rPPG-Toolbox/blob/b7500b848f84ad7f86e277b4612563b69f4f88f9/configs/infer_configs/PURE_UBFC-rPPG_EFFICIENTPHYS.yaml) 是名义 30 Hz、72×72、静态 Haar 首帧框放大 1.5、180 帧分片、Standardized 输入和 DiffNormalized 标签。官方后处理也对预测先 `cumsum`；本轮的动态框、实际相邻末帧、重叠、时间重采样和项目频带都属于明确适配。因此网络导数、积分信号和最终滤波信号分别保存，不能将三者混称原始 PPG 测量。

| 文件 | 内容 |
|---|---|
| `network_30hz.csv` | 真实网络导数、积分 BVP、滤波波形、全片 30 Hz 时间、段号和导数支持标记 |
| `waveform.csv` | 原视频时间轴上的 base、covered、observed、interpolated，缺失为 NaN |
| `heart_rate.csv` | 从保存波形得到的旧局部峰、证据候选、最终离线心率及拒绝原因 |
| `crop_provenance.csv` | 原帧的裁图可用、实际观察及框插值标记 |
| `segment_diagnostics.json` | 抗混叠比例、段长度、分片位置、均值/标准差、尾部支持情况 |
| `summary.json` | 视频、代码、模型及输出哈希；只描述推理，不写参考误差 |

`observed/interpolated` 说明直接裁图支持，不涵盖整个神经网络或离线滤波的时间感受野。原时轴 180 Hz 上的波形是 30 Hz 估计的时间回插，没有新增真实时间分辨率。积分结果振幅、基线和形态没有临床校准。

运行环境使用现存 `/home/fengbujue/项目/rppg识别/.venv-rhythm-trial/bin/python`，Torch 2.6.0+cu124、RTX 4060 Laptop 8 GB；没有安装新依赖。主 `.venv` 没有 Torch，不应据此误判设备不支持深度模型。

`test_trial.py` 的 10 项测试覆盖 30/180 Hz 时间与频率一致性、28.6 Hz 混叠负例、短缺口和长缺口、完整尾部、导数积分时序、缺失输出及官方权重 CPU/GPU 数值一致性。`test_receipt.json` 将测试绑定到适配源码哈希。`run_trial.py --run-all` 拒绝覆盖输出，并在首次推理前记录参数、代码、权重及六个视频身份；结束时复核源文件不变。

本轮输出在 `/home/fengbujue/项目/rppg识别/results/data1_6_v26_20260911/neural_efficientphys/`。参考误差由项目的独立评价器在推理之后计算，不能通过挑权重、调时间偏移或按参考修正波形改善成绩。这六组属于开发回归，不能称独立泛化验证。
