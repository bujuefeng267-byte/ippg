# rPPG 项目 GitHub 来源总清单

核对日期：2026-09-11。按当前可访问任务历史、源代码声明、试验日志、文献笔记整理；“已用于程序”“已独立试验”“阅读或备选”分别标注。研究仓库主页已重新打开核查。这里记录研究来源，不是声称所有仓库已合并进一个模型，也不是完整软件依赖 SBOM。

## 核对范围

- 当前任务 `01a08093-7b0c-7cc1-a7c6-6ca1868b2dff`，以及当前文件夹内的代码、报告、下载与运行凭据。
- 同目录另一任务「我的电脑为什么打不开@"\\wsl.localhost\Ubuntu\home\fengbujue\项目\rppg识别…」，`01a07ebc-9504-73e0-9312-213a6eb61a89`：共 11 轮，已读到历史末尾。
- 相关任务「GitHub项目链接」，`6aa0ba43-aa60-83e9-82c6-cab57b1dfbcb`：完整 1 轮。
- 相关任务「设计光流对照实验」，`6aa22525-3948-83e9-9790-aac77ad262fb`：完整 5 轮。
- WSL 原项目的来源声明由独立只读核对补充；没有恢复或运行已删除数据。

## 18 个研究与方法仓库

| 编号 | GitHub 仓库 | 在本项目中的用途 | 实际使用状态 |
|---|---|---|---|
| 1 | [phuselab/pyVHR](https://github.com/phuselab/pyVHR) | 面部颜色到 POS/CHROM 波形的传统 CPU 管线依据。 | **实际实现来源**。现有 `analyze_rppg.py` 文件头及 POS 实现保留来源说明；原项目文档明确为轻量独立实现，不是直接调用整个 pyVHR 包。 |
| 2 | [hschn58/rPPG](https://github.com/hschn58/rPPG) | 前后向像素跟踪、局部颜色异常筛选、多区域分析。 | **思路重实现并试验**。V2.3/V2.4 三步骤保留；跟踪失败回退及具体稳健筛选是本项目实现，未原样运行其整条上游管线。 |
| 3 | [KegangWangCCNU/open-rppg](https://github.com/KegangWangCCNU/open-rppg) | 封装预训练 rPPG 模型的独立推理入口。 | **早期已接入并运行**。2026-09-08 使用 open-rppg 0.1.1 的 `RhythmMamba.pure` 转换权重，完成 UBFC 与 0907 视频试验；不是当前 V2.4 默认后端。 |
| 4 | [zizheng-guo/RhythmMamba](https://github.com/zizheng-guo/RhythmMamba) | 深度学习心率/波形独立对照。 | **作者模型及固定权重已独立运行**。2026-09-09 六段来源完成；有适配过的前处理、扫描执行和统一后处理，不能称完整复现作者论文协议。与编号 3 的转换权重试验是两条路径。 |
| 5 | [gkaur102/cPACE](https://github.com/gkaur102/cPACE) | 颜色投影、跨区域相位选择、包络校正。 | **作者核心已独立运行**。在六段来源的相同 ROI RGB 上测试 homodyne 开/关分支；没有成为 V2.4 默认算法。 |
| 6 | [ubicomplab/rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) | 经典方法公式核对、标准评测管线和神经基线参考。 | **源码核对及框架参考**。曾用其 POS 实现核对 hschn58 的算法命名；没有把工具箱内每个模型都运行一遍。 |
| 7 | [stzzz99289/calirppg](https://github.com/stzzz99289/calirppg) | 区分信号质量、输出覆盖率与经过校准的正确概率。 | **方法与评测思路参考**。没有训练或安装其置信度校准模型。 |
| 8 | [yahskapar/MA-rPPG-Video-Toolbox](https://github.com/yahskapar/MA-rPPG-Video-Toolbox) | Motion Matters：用运动迁移扩充训练视频，启发受控运动对照设计。 | **调研及试验设计参考**。没有运行其神经运动生成或重新训练流水线。 |
| 9 | [PhysiologicAILab/iBVP-Dataset](https://github.com/PhysiologicAILab/iBVP-Dataset) | 参考 PPG 的质量标注、RGB/热成像与生理参考数据设计。 | **数据与质量评估参考**。没有取得该完整数据集，也没有将接触 PPG 质量模型当作当前 rPPG 正确率。 |
| 10 | [Chaoqi31/PhysMamba](https://github.com/Chaoqi31/PhysMamba) | SlowFast 时间差分与 Mamba 的神经方法。 | **阅读/备选**。未见本项目运行其模型或训练的凭据。 |
| 11 | [xjtucsy/CodePhys](https://github.com/xjtucsy/CodePhys) | 通过潜在码本查询提高视频生理估计的稳健性。 | **另一任务中的调研备选**。未见下载、运行或集成记录。 |
| 12 | [csiro/orientation-uv-rppg](https://github.com/csiro/orientation-uv-rppg) | 根据头部朝向进行面部纹理/UV 映射，研究姿态变化下的皮肤对齐。 | **另一任务中的调研备选**。当前程序没有实现这套完整 UV 映射。 |
| 13 | [Tianyang-Dai/rPPG-VQA](https://github.com/Tianyang-Dai/rPPG-VQA) | 多方法频率共识与视频质量评估。 | **近期思路参考**。未执行其完整训练/质量模型，V2.4 的分支规则是项目自己的实现。 |
| 14 | [baachraf/patchpca-codec-rppg](https://github.com/baachraf/patchpca-codec-rppg) | 局部区域 PCA、CHROM 锚定及视频压缩伪影研究。 | **近期调研候选**。没有把其 PatchPCA/P-Hybrid 分支加入当前推理。 |
| 15 | [KegangWangCCNU/ME-rPPG](https://github.com/KegangWangCCNU/ME-rPPG) | 低内存、持续状态的 rPPG 网络与推理接口。 | **近期调研候选**。没有独立运行该仓库的 ME 模型；open-rppg 包内存在相关权重文件不等于做过模型试验。 |
| 16 | [Alex036225/PhaseNet](https://github.com/Alex036225/PhaseNet) | PHASE-Net 的空间与时间建模方法。 | **近期调研候选**。没有下载运行该模型；现成训练数据与我们的 UBFC 测试可能重合，尚未作为独立基线。 |
| 17 | [chnbld/open-rppg](https://github.com/chnbld/open-rppg) | 近期检索中的同名仓库线索。 | **仅作为线索查阅**。不将它与编号 3 的作者关联仓库混同，不算另一次已运行模型试验。 |
| 18 | [contactless-healthcare/Camera-based-Monitoring-for-Full-Fitness-Cycle](https://github.com/contactless-healthcare/Camera-based-Monitoring-for-Full-Fitness-Cycle) | RGB 与运动轨迹联合处理、运动参考抑制及运动阶段比较。 | **相关任务中的作者代码/方法参考**。没有发现该上游仓库完整运行或原样集成记录；当前 NLMS 等改编不等于完整复现。 |

## 工程与设备资料

这些是定位、计算或设备资料来源，作用不同于 rPPG 信号算法。

| 仓库 | 用途与状态 |
|---|---|
| [google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe) | 实际使用的面部关键点与皮肤区域定位工具。 |
| [opencv/opencv](https://github.com/opencv/opencv) | 实际使用的视频解码、图像处理、Lucas–Kanade 光流、几何估计及跟踪验证工具。 |
| [pytorch/pytorch](https://github.com/pytorch/pytorch) | RhythmMamba 作者权重试验的张量/神经网络运行框架；不意味着传统 POS/CHROM 分支改成神经模型。 |
| [state-spaces/mamba](https://github.com/state-spaces/mamba) | RhythmMamba 所依赖的状态空间模块来源；实际使用作者仓库内的 Mamba 类及非融合 `selective_scan_ref`。代码由 RhythmMamba 固定提交绑定，`v2.2.2` 仅作为审计比对版本。 |
| [polarofficial/polar-ble-sdk](https://github.com/polarofficial/polar-ble-sdk) | H10、Verity Sense 等设备输出与模式说明的资料依据；不能据此声称当前 Python 分析脚本已调用 Polar 蓝牙 SDK。 |

`RhythmMamba.py` 原始源码注释还间接引用 [huggingface/transformers](https://github.com/huggingface/transformers) 与 [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)。这两项是保留的上游实现注释，不是本项目另行选取、安装并评测的 rPPG 模型。

## 已固定的外部版本

| 来源 | 本地记录 |
|---|---|
| hschn58/rPPG | 提交 `8dd44360d159eccde7f85a5a6542e3b44c621029`；保留阅读源码快照；自身上游管线未运行。 |
| cPACE | 提交 `4cd7a438bf9dd30af8ec3afc365dad53861b6c97`；作者 Python 核心原样保留，适配与评分单独实现。 |
| RhythmMamba 作者版 | 提交 `1533ad21cb4ae0d31e9dbb9afb0b1eefbed54aed`；`PURE_cross_RhythmMamba.pth` SHA256=`442ec9ac71bfc299f41e2c7f671216cfa4f6749d8dd264e53f8eb9d8b65380ee`。 |
| open-rppg 早期试验 | 包版本 `0.1.1`，`RhythmMamba.pure`；转换权重 `RhythmMamba.pure.weights.h5` SHA256=`dc2f9c8e421a70cf31db54a6b3334852f165d4ffa2cf1d15f493afc499ca7bcb`。没有将包版本伪称 Git 提交。 |

## 本地证据入口

- [现有基线代码的 pyVHR 来源声明](</C:/Users/15011/Documents/ChatGPT/New project/rppg_motion_v24/analyze_rppg.py:1>)。
- [早期 open-rppg 接入和运行验证](</C:/Users/15011/Documents/ChatGPT/New project/rppg_motion_upgrade/交付验证.md>)；[0907 模型运行记录](</C:/Users/15011/Documents/ChatGPT/New project/rppg_motion_upgrade/validation_final/neural_user0907/neural_summary.json>)。
- [cPACE 实际试验说明](</C:/Users/15011/Documents/ChatGPT/New project/rppg_paper_trials_20260909/cpace/README.md>)；[RhythmMamba 作者权重试验说明](</C:/Users/15011/Documents/ChatGPT/New project/rppg_paper_trials_20260909/rhythm/README.md>)。
- [hschn58/rPPG 阅读笔记](</C:/Users/15011/Documents/ChatGPT/New project/rppg_hschn58_review_20260910/公开项目阅读笔记.md>)；[V2.3 思路重实现](</C:/Users/15011/Documents/ChatGPT/New project/rppg_motion_v23/README.md>)。
- [V2.2 文献记录](</C:/Users/15011/Documents/ChatGPT/New project/rppg_motion_v22/literature.md>)；[V2.4 近期文献与仓库记录](</C:/Users/15011/Documents/ChatGPT/New project/rppg_motion_v24/research_v24.md>)。

状态解释采用最新运行凭据：早期文献笔记中的“尚未运行 RhythmMamba”随后已经被实际试验更新；V2.4 调研中的“本轮未执行外部模型”也不能被扩大为历史上从未使用 open-rppg。

当前 V2.4 核心仍是项目自行组织的面部采样、像素跟踪与筛选、失败回退、POS/CHROM、多区域融合、颜色累积校正及离线心率读出。外部论文和仓库的成绩不等于本项目成绩，借鉴某一步也不等于接入其完整模型。

## 自有仓库与未确证项

- [bujuefeng267-byte/rppg](https://github.com/bujuefeng267-byte/rppg) 是原项目 `.git/config` 和 `PROJECT_OPERATIONS.md` 确认的自有仓库，单列供定位，不计入 18 个外部研究来源。
- 相关任务还提到 RhythmFormer、Contrast-Phys+、rPPG-MAE、Face2PPG 等论文方法；没有取得这些讨论所对应的明确 GitHub 链接或本项目运行凭据，未将网络上同名仓库补成历史已借鉴来源。
- 旧任务中虽出现已删除 `_gudb` 的文件名，本次可访问记录未找到其历史 GitHub 来源地址；不根据名称猜测仓库，也没有恢复数据。

## 程序的视频输入要求与拍摄建议（2026-09-11 补充）

本节对应当前 `run_motion_v24_experimental.sh` 启动的 V2.4 实验版，是根据实际源码核对的输入说明。它不等同于上述所有外部神经模型的输入要求；例如某个独立模型内部使用 128×128 裁脸，不代表拍摄视频必须是 128×128。

### 拍摄时可以直接采用的配置

**建议起点：固定 30 fps、1920×1080 彩色视频、每段 30–60 秒、单人面部清晰、稳定均匀光照，保留原视频。** 这些是依据现有处理方式和约 30 fps 样本给出的工程建议，未证明为最优配置，也不能保证心率正确。拍运动视频时仍可正常运动，尽量让面部保持在画面内，并减少严重拖影、遮挡及照明变化。

| 项目 | 程序的实际处理与限制 | 拍摄或提交建议 |
|---|---|---|
| 文件格式 | 通过 OpenCV VideoCapture 读取。AVI、MP4、MOV 都有本项目处理记录；能否读取取决于文件内的编码和本机解码器，并非只看扩展名。 | 提交原始彩色视频，先验证可解码；不要仅修改后缀来转换格式。 |
| 帧率 | 读取文件报告的 fps。前端拒绝非有限 fps 或 fps<5；完整融合另要求 `max_bpm < 30×fps`，所以默认上限 210 bpm 下必须 **fps>7**。带通上限还裁到 `0.48×fps`，完整保留 3.5 Hz 上限需约 fps≥7.292。 | **优先固定 30 fps**。低帧率的数学允许值不是质量达标线；60 fps 可作为独立试验，但没有证明它在当前项目上一定更准。不能通过重复帧或只改文件 fps 制造高采样率。 |
| 时间轴 | 当前时间戳为 `帧序号 / fps`，没有读取逐帧真实 PTS，也不会主动发现所有变帧率、重复帧或丢帧问题。 | 优先真实恒定帧率（CFR），避免变速、慢动作回放文件及拼接。VFR 若要用于精确参考对齐，需要另行实现真实时间戳处理；直接转成 CFR 不会恢复丢失的观测。 |
| 分辨率 | **没有强制 720p、1080p 或 4K，也没有固定宽高比要求。** 检测/光流内部对宽度超过 960 像素的画面缩小处理，颜色仍从原分辨率帧采样；960 不是输入上限。 | 可从 **1920×1080** 开始。现有 3840×2160 也能处理，但解码和像素处理负担更大；总分辨率不能代替面部实际大小和清晰度。 |
| 面部大小 | 原图中由关键点 2%–98% 分位数定义的人脸区域，宽和高均须至少 **40 像素**，否则拒绝采样。 | 40 像素只是代码最低门槛；建议面部占据明显区域，额头和双颊可见，勿把远处小脸当作合格输入。当前没有验证过统一的最佳人脸像素尺寸。 |
| 皮肤颜色与曝光 | 按三通道 RGB 和 0–255 数值尺度采样。每个 ROI 至少要有 **25 个有效像素**，各通道都严格处于 **20 与 245 之间**；过暗、过曝像素会被排除。 | 使用可见光彩色视频，保持自然颜色和均匀照明，尽量稳定曝光与白平衡，关闭美颜/滤镜。灰度、纯红外或热成像不属于当前 POS/CHROM 的既定输入假设；未提供相机 RAW/Bayer/16-bit 原始数据的专用适配入口。 |
| 视频时长 | 默认 **10 秒心率窗、约 1 秒步长**；CLI 最低允许 6 秒窗。短片可以被读取，但不足一个完整有效窗时不会产生有效 HR。每段连续有效 RGB 两端还各屏蔽约 1.6 秒滤波边缘。 | 建议每段 **30–60 秒或更长**，为边缘和缺失留余量。不能理解为“拍满 10 秒就一定输出”；即使超过 15 秒也仍须通过质量检查。 |
| 人数与朝向 | Face Mesh 配置为 `max_num_faces=1`，没有多人分别输出及固定人员身份锁定。 | 一次分析一个明确目标，优先正面或轻度转头；避免多张脸竞争、大角度转头及面部长期离开画面。 |
| 短缺口 | 默认只对两端均有真实采样的内部缺口，且长度不超过 `floor(0.10×fps)` 帧时插值；不补首尾、不跨长缺口。 | 尽量连续拍摄并保存完整帧序列。文件读得出来不代表每段都有可用波形。 |
| 质量与输出 | 融合至少需要两个独立 ROI 通过筛查；各参与 ROI 在窗口内真实采样比例至少 90%、采样质量均值至少 0.20，且波形和频率共识等条件合格。最终心率还检查实际保存波形全窗有限、真实采样比例≥90%、插值比例≤10%。 | 这些是**逐窗口**条件，不能用整段视频的人脸可用率代替；检测到脸也不等于该窗可以输出正确 PPG/HR。 |
| 生理参考 | 推理本身不需要 Polar、ECG 或接触 PPG。参考仅用于事后计算误差。 | 比较心率时一并提供带时间戳、单位明确的参考数据和同步依据；验证波形形态还需要合格的同步接触 PPG。 |

### 你的最新 data1 视频

| 属性 | 现有记录 |
|---|---|
| 文件 | `Video_20260909_YJC_2.avi` |
| 标称帧率 | 30.000030 fps，约 30 fps |
| 分辨率 | 3840×2160 |
| 已解码帧数 | 1555 |
| 按帧数/fps计算的时长 | 51.8333 秒 |
| V2.4 输出记录 | 波形覆盖 88.75%；心率输出 37/42 窗，即 88.10% |

这个文件的帧率、分辨率和时长属于当前程序已经处理过的规格。元数据并不能单独证明采集时间轴无丢帧或抖动，规格合适也不等于误差合格；当前该片与 Polar 的同步仍是估计值。

### 实现依据和边界

- `motion_upgrade_v24_20260910/baseline_frontend.py:76–116`：RGB、面部大小与有效像素要求；`:145–169`：解码、fps、单人检测与内部缩小；`:213`：帧序号时间轴。
- `motion_upgrade_v24_20260910/analyze_motion_v2.py:97–113`：默认参数与窗口校验。
- `motion_upgrade_v24_20260910/analyze_rppg.py:173–177`：带通上限与 fps 的关系。
- `motion_upgrade_v24_20260910/motion_fusion.py:237–241`、`waveform_hr.py:32–35`：完整流程的 fps/心率范围校验。
- `motion_upgrade_v24_20260910/legacy_motion.py:199–231`：内部短缺口与滤波边缘；`:262–290`：最终心率窗口检查。
- [OpenCV 视频属性说明](https://docs.opencv.org/4.13.0/d4/d15/group__videoio__flags__base.html)：VideoCapture 的 FPS、尺寸与后端属性说明。最终限制以本项目固定版本源码和实际解码结果为准。

本次只补充说明文档，没有修改推理参数、程序代码或原视频，也没有因新增文档而重新计算性能。


## 主要借鉴项目：作者实测误差与视频规格（2026-09-11 补充）

本节回答“原作者是否做过应用、与参考心率相比误差多少、实验视频的帧率和分辨率是什么”。选取与本项目方法关系较直接的 pyVHR 和 hschn58/rPPG。以下是公开资料核查结果，不是本项目重新运行这两个仓库的复现实验结果。

### 1. pyVHR：有公开的参考心率对照，但必须指明算法版本与数据集

仓库：[phuselab/pyVHR](https://github.com/phuselab/pyVHR)。它包含多种方法，不能用一个误差数字代表整个框架。

作者论文 *Enhancing rPPG pulse-signal recovery by facial sampling and PSD Clustering*，刊于 Biomedical Signal Processing and Control 101（2025），107158；2024-11-19 在线发表。正文第 11 页 Table 2 给出如下平均绝对误差 MAE，单位 bpm（次/分钟）：

| 作者采用的方法 | PURE 数据集 MAE | UBFC 数据集 MAE |
|---|---:|---:|
| PatchClustering + CHROM | **0.92 bpm** | **1.11 bpm** |
| PatchClustering + POS | **1.85 bpm** | **1.41 bpm** |

这四项是作者加入面部分块与功率谱聚类后的结果，不能写成普通 CHROM/POS 的固定精度，也不是我们当前 V2.4 程序的结果。正文实验设置为 100 个面部标志点、8 秒时间窗；本次未核实到明确的外层窗口步长，不能套用其他示例的默认值。Table 2 报告的是 MAE，不应改写为 RMSE、最大误差或准确率百分比。来源：[作者论文，正文第 10–11 页](https://air.unimi.it/retrieve/handle/2434/1121060/2596995/1-s2.0-S1746809424012163-main.pdf)。

视频规格由数据集发布方说明核实：

| 数据集 | 视频帧率 | 视频分辨率 | 拍摄条件 | 参考测量 |
|---|---:|---:|---|---|
| PURE | **30 fps** | **640×480 像素**，发布方称裁剪后尺寸 | 10 人，每人 6 种约 1 分钟的状态：静止、说话、慢/快平移、小/中幅转头；发布方明确均在休息状态下采集 | pulox CMS50E 指夹式血氧仪，同步参考波形采样率 60 Hz |
| UBFC-rPPG | **30 fps** | **640×480 像素**，未压缩 8-bit RGB | 室内，距相机约 1 米；UBFC1 为静坐任务，UBFC2 为坐着做限时数学游戏 | CMS50E 透射式血氧仪，提供参考 PPG 波形和心率 |

来源：[PURE 原始发布页](https://www.tu-ilmenau.de/en/university/departments/department-of-computer-science-and-automation/profile/institutes-and-groups/institute-of-computer-and-systems-engineering/group-for-neuroinformatics-and-cognitive-robotics/data-sets-code/pulse-rate-detection-dataset-pure)、[UBFC 原始发布页](https://sites.google.com/view/ybenezeth/ubfcrppg)。

表中尺寸是数据集视频尺寸，不是神经网络输入张量或单个面部小块的尺寸。论文 Table 2 列名仅写 UBFC；这里保留该名称，不把它自行扩展为 UBFC1/2 的某种合并统计口径。PURE 官方描述为 60 个序列的采集设计，不应据此推断论文实际使用的有效文件数。

运动场景的边界也需要保留：同一篇论文第 10 页 Table 1 中，LGI-PPGI 的 Gym 场景，PatchClustering 的 MAE 为 **14.42 bpm**。这一数值是跨基础算法汇总的结果，不是单独 CHROM 的误差，不能直接与上表算改进率。它说明作者公开实验中的误差也会随运动场景显著增大。[论文 Table 1](https://air.unimi.it/retrieve/handle/2434/1121060/2596995/1-s2.0-S1746809424012163-main.pdf)。

### 2. hschn58/rPPG：有实验展示，尚未找到可核验的同步参考误差

仓库：[hschn58/rPPG](https://github.com/hschn58/rPPG)。本次核查对应提交 `8dd44360d159eccde7f85a5a6542e3b44c621029`。

| 所问项目 | 能从公开材料确认的内容 | 可以得出的结论 |
|---|---|---|
| 是否实际使用过 | README 展示约 300 帧、5 个面部区域的 CHROM/POS 估计和波形、SNR | 有程序运行与实验展示 |
| 与参考心率相比误差多少 | 未找到同步参考心率序列、逐窗配对误差表或作者汇总 MAE/RMSE | **目前无法给出可信的作者误差数字；不应填写 0 或猜测数值** |
| 视频帧率 | 多个分析脚本固定按 30 fps 处理，录制脚本也有自动测量帧率的逻辑 | 30 fps 是可见的分析设置，未证明所有示例视频的实际采集帧率均为 30 fps |
| 视频分辨率 | 五区域录制脚本设置保存尺寸为 1760×1328 | 这是该脚本的输出设置，不能证明每段实验视频的原生采集分辨率 |
| 视频长度 | README 约 300 帧；录制脚本存在 10 秒示例设置 | 若确为 30 fps，300 帧约等于 10 秒，但缺少逐视频元数据核对 |

证据：[固定版本 README](https://github.com/hschn58/rPPG/blob/8dd44360d159eccde7f85a5a6542e3b44c621029/README.md#multi-region-heart-rate-estimation-chrom--pos)、[五区域录制脚本](https://github.com/hschn58/rPPG/blob/8dd44360d159eccde7f85a5a6542e3b44c621029/Full_Stack/Video_get_to_npy/YuNet_and_Template_match/VID_TO_DATA_5_Regions.py#L232-L273)、[CHROM 分析脚本](https://github.com/hschn58/rPPG/blob/8dd44360d159eccde7f85a5a6542e3b44c621029/Full_Stack/CHROM_method/CHROM_method_npy_input.py)。

手动填写录制开始和结束的心率，不能替代整段视频同步参考序列。README 还明确提到运动处理后重复出现约 2.5 Hz（约 150 bpm）的伪峰，来源尚未解决；不能仅凭曲线平滑、频谱有峰或 SNR 较高就认定测量准确。[README 的实验与问题说明](https://github.com/hschn58/rPPG/blob/8dd44360d159eccde7f85a5a6542e3b44c621029/README.md)。

### 3. 对本项目的解释

30 fps、640×480 的视频可以在上述受控数据上支持较低心率误差；这不构成将本项目运动视频缩小到该尺寸就会提高精度的证据。跨项目比较还需要统一视频、同步参考、评估窗口和输出覆盖率定义。这里列出的作者表格没有给出与本项目完全同口径的人脸可用率、PPG 波形覆盖率和心率覆盖率，不能将未报告项当作 100%。

本次只补充文档，未修改算法，也未把文献中的性能数字写入本项目实验结果。
