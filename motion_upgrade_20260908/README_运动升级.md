# rPPG 运动处理实验版（2026-09-08）

本次更新在独立目录中提供可运行的算法对照。旧 `analyze_rppg.py` 不修改；目录内同名文件是原脚本的冻结副本。尚未证明跑步时心率准确。

## 运行入口

在 WSL 项目目录运行，输出目录必须是新目录，防止覆盖旧结果：

```bash
cd /home/fengbujue/项目/rppg识别
.venv/bin/python motion_upgrade_20260908/analyze_rppg_motion.py \
  videos/user_0907/Video_20260907_171742478.avi \
  --frontend robust --output results/motion_upgrade_20260908/my_run
```

默认输出 POS、CHROM 各自的频谱峰与离线连续跟踪。加 `--nlms` 才增加 NLMS 对照。NLMS 在当前静态验证中退化，暂不推荐为默认方法。

`--frontend baseline` 使用原始分辨率与原 Face Mesh 设置；`robust` 在最多 960 像素宽图像上检测，失败时独立重检，再尝试有前后向一致性与 RANSAC 验证的光流短时跟踪。颜色仍从原视频当前帧提取，不复制前一帧图像。两种前端共享原有三块皮肤区域采样方法；没有宣称完成三维皮肤配准。

有 UBFC 参考时加入：

```bash
--reference-ubfc videos/ubfc_subject1/ground_truth.txt
```

## 新版输出与约束

- `frame_trace.csv`：每帧 RGB、运动参考、人脸框及 `mesh/redetected/flow_tracked/missing` 来源。光流采样计入可用 RGB，但不计作新检测。
- `pos_waveform.csv`、`chrom_waveform.csv`：基本波形、实验 NLMS 波形、是否真实采样和是否插值。
- `*_heart_rate.csv`：局部谱峰、离线动态规划轨迹、接收/拒绝原因、采样覆盖率。提供参考时增加参考心率。
- `summary.json`：全部配置、源代码 SHA-256、输入路径/大小/修改时间、各模块结果。
- `comparison.png`、`qa/`：结果对照及少量关键点抽查图。QA 图片含人脸，仅供本地检查。

只插值内部不超过 0.1 秒的 RGB 缺口；不外推首尾，不跨越长缺口滤波。每个有效片段去掉两端各 1.6 秒，减少滤波及重叠相加边缘影响。心率窗默认 10 秒，步长约 1 秒，需要至少 90% 真实 RGB，且插值不超过 10%。常量信号、未填补缺口和过于分散的频谱拒绝输出。阈值是可复现的初始工程设置，不是经临床校准的置信度。

`peak_concentration` 只是选定峰附近 ±6 bpm 的频谱能量比例。它不是相对真实心率计算的 SNR，也不是心率准确概率。`accepted` 仅表示通过初步数据筛查。

连续跟踪使用未来窗口，是离线方法。允许最大 12 bpm/s 的转移，惩罚频率跳变；跨拒绝窗口重新开始。平滑的错误频率仍然可能被持续追踪。该实现不是论文 AMTC 的完整复现。

NLMS 采用 8 抽头、步长 0.1 的双轴运动参考抵消器，参考经过带通及归一化，与论文实现并非完全相同。运动参考与真实脉搏相关时可能删除脉搏，静态微小跟踪误差也可能造成负效果，因此仅显式开启作消融比较。

## 已完成的验证

1. 8 项测试：短缺口/长缺口区别、缺失窗口不生成跟踪值、常量信号拒绝、独立运动噪声抑制、无运动参考直通、孤立假谱峰、光流平移/空白图拒绝、参考重复时间戳及禁止外推。
2. UBFC subject1 全片（1547 帧）与 9 月 7 日原始 AVI 全片（1557 帧），分别运行 baseline 与 robust 前端，比较 POS/CHROM、NLMS 关闭/开启、峰值/连续跟踪。
3. 已抽查动态帧 51、300、1200 的关键点图，点位位于脸部。抽查不能证明每帧皮肤区域均准确。

### UBFC subject1（robust 前端，40 个共同有效窗口）

| 方法 | 局部峰值 MAE / RMSE (bpm) | 连续跟踪 MAE / RMSE (bpm) |
|---|---:|---:|
| POS | 3.209 / 4.305 | 2.561 / 3.061 |
| CHROM | 2.767 / 3.564 | 2.545 / 3.141 |
| POS + NLMS（实验） | 19.912 / 28.815 | 3.085 / 3.732 |
| CHROM + NLMS（实验） | 27.450 / 35.103 | 2.985 / 3.679 |

总计 44 个窗，40 个窗通过筛查，覆盖率 90.9%。基准前端和 robust 前端的无 NLMS 结果在此样本一致。以上参考是文件第二行的设备 HR，在第三行时间戳上按相同窗平均；一个重复时间戳合并。视频时间使用名义 fps；未根据预测结果拟合时移。它不同于旧报告按帧对齐的处理，不能直接与旧报告数字比较。只有一个被试，且窗口高度重叠，不能作总体显著性或运动准确性结论。

### 9 月 7 日动态视频（无参考）

| 指标 | baseline 前端 | robust 前端 |
|---|---:|---:|
| 可用 RGB 帧数 | 1048 / 1557 | 1415 / 1557 |
| 可用 RGB 比例 | 67.31% | 90.88% |
| 原 Mesh 成功帧 | 1048 | 1044 |
| 独立重检恢复帧 | 0 | 14 |
| 光流跟踪采样帧 | 0 | 357 |
| 未取得 RGB | 509 | 142 |
| 新筛查规则下通过的心率窗 | 0 / 42 | 14 / 42 |

14 个窗口通过不代表心率准确。当前 POS 与 CHROM/不同滤波分支仍出现约 60 与 150 bpm 的分歧，说明运动/谐波问题尚未解决。未计算该视频的 MAE/RMSE，不以曲线平滑程度选择“真值”。

## 独立深度模型入口

`run_open_rppg.py` 使用 open-rppg 封装的 `RhythmMamba.pure` 权重，复用逐帧人脸框；不是论文作者原始 PyTorch 权重的复现。以模型元数据决定输入大小，导出原始 BVP，并记录库版本、权重哈希、有效片段与采样信息。内部最多插值 0.1 秒的人脸框，颜色仍从当前帧采样；不跨长缺口预测。

```bash
.venv-neural/bin/python motion_upgrade_20260908/run_open_rppg.py \
  videos/user_0907/Video_20260907_171742478.avi \
  --trace results/motion_upgrade_20260908/user0907_robust/frame_trace.csv \
  --output results/motion_upgrade_20260908/neural_user0907
```

深度结果的实际运行状态见同目录 `交付验证.md`。该环境独立于原 `.venv`。默认 CPU，后续 GPU 加速应另行核查驱动与环境。原视频不上传外部服务。

## 依据、复现和下一步

- [Zhu 等，运动条件下模块化 rPPG 比较](https://par.nsf.gov/servlets/purl/10630048)：借鉴运动参考抵消与频率连续性，不冒充完整复现。
- [RhythmMamba 官方工程](https://github.com/zizheng-guo/RhythmMamba)：后续严格论文对照入口。
- [open-rppg](https://github.com/KegangWangCCNU/open-rppg)：本次独立推理封装。
- [pyVHR](https://github.com/phuselab/pyVHR)：原项目 POS/CHROM 实现依据，其许可说明保留。

下一步需要同步参考的真实运动数据。使用相同被试、相同窗口及有效率比较所有分支，并在独立被试上验证参数。不要在同一测试视频上调参后再把该视频结果当作泛化证据。若目标为 PPG 波形恢复，需要同步接触式 PPG；ECG/胸带 HR 能验证心率，但不能作 PPG 波形形态真值。
