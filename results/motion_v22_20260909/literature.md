# rPPG V2.2 文献与实现边界笔记

检索截至 2026-09-09。仅用原论文、出版方页面及作者仓库核实；未运行这些外部模型，也未下载权重。这里的 CPU 可行性是依据算法与依赖作出的工程判断，不是本机速度实测。以下 6 篇按与当前项目的用途归类，不按跨数据集 MAE 排名。

目前优先保留可审计的经典方法，独立检验 ROI 采样和缺口规则；神经模型宜先采用公开权重做隔离基线。现有少量视频适合诊断与回归测试，不足以证明跨人、跨运动或跨相机泛化。固定 10% 截尾均值、0.10/0.15 s 短缺口消融是本项目工程选择，不能称为下列论文的完整复现或论文已证明的最优参数。

## 6 篇可用资料

| 文献与发表状态 | 机制、训练及数据前提 | 代码、权重与当前适用性 |
|---|---|---|
| **cPACE**：*Missed isochromatic cardiac pulsation in remote photoplethysmography: detection, impact, and removal*。Biomedical Optics Express 17(7)，2026；PubMed 记录在线发表 2026-06-24，卷期日期 2026-07-01。 | 自适应颜色投影、特征向量选择、跨 ROI 相位一致性及包络校正；无需神经训练。论文涉及 UBFC-Phys 及 CMU-rPPG 数据，不能据此承诺跑步条件效果。作者将它定位为研究诊断工具。 | NumPy/SciPy 主体可从已有 ROI RGB 建立独立 CPU 分支；权重不适用。默认实现依赖从视频估计的 HR 种子，包络校正会改变波形，详见下节。[原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC13372377/)、[出版 DOI](https://doi.org/10.1364/BOE.599752)、[作者代码](https://github.com/gkaur102/cPACE)。 |
| **Cali-rPPG**：*A Unified Uncertainty-Aware Framework for Remote Photoplethysmography*，ICIC 2025，Springer 章节首次在线 2025-07-25。 | 将 rPPG 估计、HR 分布预测和置信度校准分成模块。推理时不需参考信号，但作者流程先用 **PSD–HR 标注对**拟合置信度模型，再做校准；不等于无监督获得可信概率。 | 有作者实现，支持 POS/CHROM 等经典方法；仓库有 weights 目录，但本次未核实可直接复用的最终校准参数。当前可借鉴误差—覆盖率与分层诊断，正式概率需独立校准集。[论文](https://link.springer.com/chapter/10.1007/978-981-95-0033-8_18)、[作者流程](https://github.com/stzzz99289/calirppg)。 |
| **iBVP Dataset**：*RGB-Thermal rPPG Dataset with High Resolution Signal Quality Labels*，Electronics 13(7):1334，2024-04-02 正式发表。 | 同步 RGB/热成像和耳部接触 PPG，含呼吸、认知任务及引导头动；对**参考 PPG**提供人工与模型质量标注，强调参考本身也可能失真。 | 作者提供 SQA-PhysMD 模型、推理代码和 checkpoint 入口，数据需学术 EULA 申请。可马上借鉴参考质量审计；不能将接触 PPG 质量模型未经验证地当成人脸 rPPG 的正确率。数据并非本次已取得。[原文](https://www.mdpi.com/2079-9292/13/7/1334)、[作者数据及代码入口](https://github.com/PhysiologicAILab/iBVP-Dataset)。 |
| **Motion Matters**：*Neural Motion Transfer for Better Camera Physiological Measurement*，WACV 2024；预印本始于 2023。 | 用神经运动迁移扩充训练运动类型和幅度；需要有生理标签的源视频、驱动视频和合成模型，再训练 rPPG 网络。不是对一条现有波形做滤波。 | 作者释放增强管线和预训练模型入口。原环境为较旧 PyTorch/CUDA，整套生成不宜当作本轮 CPU 快速修复；当前可借鉴“分别改变运动幅度/类型”的受控压力测试设计，简单几何扰动不是该论文复现。[会议原文](https://openaccess.thecvf.com/content/WACV2024/papers/Paruchuri_Motion_Matters_Neural_Motion_Transfer_for_Better_Camera_Physiological_Measurement_WACV_2024_paper.pdf)、[作者项目](https://motion-matters.github.io/)、[作者代码](https://github.com/yahskapar/MA-rPPG-Video-Toolbox)。 |
| **PhysMamba / SlowFast Temporal Difference Mamba**，Chaoqi Luo 等；CCBR 2024，会议论文集于 **2025-02-07** 出版；2024-09 预印本。 | SlowFast 时间差分与 Mamba 建模，监督训练；作者提供 PURE、UBFC-rPPG、MMPD 数据说明及交叉数据集训练配置。 | 作者代码可见，但本次未核实可直接下载使用的训练后权重；安装步骤明确含 cu118、causal-conv1d 和 Mamba。属于中期路线，不是当前 CPU 即插即用升级。不要与 Yan 等的另一篇同名 PhysMamba 混用。[正式章节](https://link.springer.com/chapter/10.1007/978-981-96-1071-6_23)、[预印本](https://arxiv.org/abs/2409.12031)、[作者代码](https://github.com/Chaoqi31/PhysMamba)。 |
| **RhythmMamba**：*Fast, Lightweight, and Accurate Remote Physiological Measurement*，AAAI 2025；2024-04 预印本题名略有不同。 | 多时间尺度 Mamba 与频率信息建模；需要人脸视频与生理标注训练，公开配置覆盖数据集内和跨数据集评估。 | 已看到 MMPD、PURE、UBFC 及 VIPL 对应 `.pth` 文件列表，未下载/加载验证。作者 requirements 含 mamba-ssm、causal-conv1d，需先解决 CUDA 环境或验证 CPU 替代实现。中期可优先用冻结公开权重做独立基线。[AAAI 原文](https://ojs.aaai.org/index.php/AAAI/article/view/33204)、[预印本](https://arxiv.org/abs/2404.06483)、[代码](https://github.com/zizheng-guo/RhythmMamba)、[权重列表](https://github.com/zizheng-guo/RhythmMamba/tree/main/PreTrainedModels)。 |

## cPACE：可以接什么，不能直接照搬什么

设原始 ROI 颜色均值为 μ，单位方向 q=μ/‖μ‖，投影矩阵 P=I−qqᵀ。Pq=0，所以在同一颜色坐标系中，它可去除沿 q 的分量。作者代码对 RGB 逐通道作时间均值归一化和去趋势，再投影；在 HR 种子 ±0.30 Hz 内计算投影信号的 3×3 协方差，得到两个特征向量。用种子 ±0.25 Hz 内的跨 ROI PLV，在所有 ROI 间统一选择 v1 或 v2。输出的候选是向量作用于投影信号，而不是单纯生成正弦。[投影源码](https://github.com/gkaur102/cPACE/blob/main/projections.py)

已逐行读过作者 `pipeline.py`、`run_ubfc_phys.py`、`run_ubfc_rppg.py`：默认 seed 来自额头 GREEN 的谱峰；两个批处理将 truth 用于评分，**未传入参考 HR 作为 seed**。API 允许外部 override，但“允许”不能被写成“作者默认泄漏参考”。它的整体 HR 汇总方式与本项目逐窗 HR/离线 DP 不同，不能原样数值对照。[管线](https://github.com/gkaur102/cPACE/blob/main/pipeline.py)、[UBFC-Phys 批处理](https://github.com/gkaur102/cPACE/blob/main/run_ubfc_phys.py)、[UBFC-rPPG 批处理](https://github.com/gkaur102/cPACE/blob/main/run_ubfc_rppg.py)

其 `homodyne_correct` 先窄带滤波，再对 Hilbert 解析信号求包络 A 与相位 φ，输出 A/max(A_slow, floor)·cosφ。默认慢包络截止 0.30 Hz、floor 为慢包络中位数的 5%。因此输出继承窄带信号相位，带外谐波会受压制；它**不是对原始宽带波形仅作幅度修正**。不能把更集中谱峰直接解释为更好的原始脉搏形态恢复。[包络源码](https://github.com/gkaur102/cPACE/blob/main/homodyne.py)

两个接入前的工程检查由以上源码推导，不是论文已经证明的缺陷：第一，共同干扰若主导 GREEN，seed 和跨 ROI 一致性仍可能一致地选择错误频段。第二，逐通道均值归一化改变颜色坐标；对 RGB(t)=μ·(1+a(t))，归一化后的扰动方向变成 (1,1,1)，通常不再与原始 q 平行。源码仍用原始 μ 构造 P，故不能直接声称它会消除这一归一化坐标中的共同亮度扰动。应先用无参考合成案例检查坐标一致性；这项推断不足以否定论文全部结论。

## 现在可落实的两个机制

1. **颜色分量归一化的独立分支实验**：沿用同一 ROI 原始 RGB、有效帧和宽带评价口径，把 q 投影与包络处理拆成开关，先测试纯颜色脉搏、共同亮度变化、局部污染及共同错误频率。全部 seed 仅由视频产生并保存。若为了保留形态而改为宽带幅值校正，就必须命名为“受 cPACE 启发的改编”，不能引用原论文成绩作为本项目结果。它适合下一项独立 CPU 实验，不应与本轮截尾均值混成一次无法归因的修改。
2. **将可观测质量与最终正确率分离**：分别保存有效像素/帧比例、插值长度、候选冲突、频率稳定性；在有可信参考的测试集上同时统计误差与覆盖率。加入固定幅度的 ROI/亮度/局部遮挡扰动，观察输出敏感性。这是借鉴 Cali-rPPG 的校准分层与 Motion Matters 的受控运动比较做法，当前可无需训练落地；这些原始诊断量仍不是校准概率。长缺口不能因插值而被当作新增实测覆盖。参考质量检查可借鉴 iBVP，但要保留数据类型差异。

当前固定 10% 截尾均值是稳健空间采样假设，短缺口规则是连续性假设。应逐项冻结、在相同窗集合和全时段覆盖下报告收益/退步；不要使用参考 HR 挑选 RGB 分量、拟合时间偏移或决定推理输出。已有视频参与调试后，最终泛化结论仍需独立新数据。

## 中期神经路线与未核实项

先建立 RhythmMamba 冻结权重的隔离推理基线，核对权重训练数据、裁脸、输入尺寸、归一化、时间长度和后处理，并实际测试环境与速度。若 CPU 路线受 Mamba 内核限制，应明确记录，不能把 FLOPs 低等同于 CPU 可直接运行。随后再考虑 PhysMamba 或 Motion Matters 增广训练；需要足够的跨人训练数据、独立校准集及保留测试集，少量当前视频不适合从零训练后宣称泛化。

[rPPG-Toolbox 官方仓库](https://github.com/ubicomplab/rPPG-Toolbox)适合作为经典/神经方法的对照基础；其原论文是 [NeurIPS 2023 Datasets and Benchmarks](https://proceedings.neurips.cc/paper_files/paper/2023/file/d7d0d548a6317407e02230f15ce75817-Paper-Datasets_and_Benchmarks.pdf)，不是 2024–2026 新论文。复现时固定检测器和预处理，不能只换模型名字就称公平对照。

“GPhys”本次未找到能确认对应 rPPG 方法的论文、作者及官方代码，因此不列为已核实方法；需要精确题名或作者再确认。公开权重存在不等于本次已复现，论文的 MAE、运动条件与推理速度也不能替代本项目实测。
