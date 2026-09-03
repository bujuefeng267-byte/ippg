# rPPG 项目完整操作记录

> 更新时间：2026-09-03
> WSL 项目目录：`/home/fengbujue/项目/rppg识别`
>
> GitHub 仓库：<https://github.com/bujuefeng267-byte/rppg>

## 1. 项目目标

本项目验证两件事：

1. 普通 RGB 摄像头拍摄的人脸视频能否通过传统开源 rPPG 方法提取脉搏波并估计心率；
2. 静止和运动条件下算法的误差、运动伪影与失效趋势如何变化。

当前实现 POS 和 CHROM 两种方法，使用 MediaPipe 定位人脸皮肤区域，并通过带通滤波、Welch 功率谱和滑动窗口估计心率。完整实验结论见 `rPPG静态验证与动态实验对比汇总.md`。

## 2. GitHub 首次共享范围

首个版本包含：

- 全部 Python 和 Shell 源代码；
- `requirements.txt` 和运行说明；
- 72 bpm 无隐私合成视频；
- 已生成的 CSV、JSON 和 PNG 分析结果；
- 本操作记录和实验汇总报告。

首个版本不包含：

- `.venv/`：可由依赖文件重新创建，不应上传；
- `__pycache__/`、`*.pyc`：运行缓存；
- UBFC 真人原视频、参考数据原文件；
- Kaggle 动作候选原视频和裁剪视频；
- 任何许可不明确、包含真人隐私或超过 GitHub 普通文件大小限制的数据。

GitHub 普通 Git 仓库对单文件有 100 MB 限制。UBFC 视频约 1.33 GiB、Kaggle 视频约 514 MiB，因此不能直接推送。后续若要共享数据，应优先提供数据集官方下载说明；确有必要时再单独评估 Git LFS 和数据许可。

## 3. WSL 环境部署

```bash
cd /home/fengbujue/项目/rppg识别
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

确认主要脚本参数：

```bash
.venv/bin/python analyze_rppg.py --help
.venv/bin/python scan_motion.py --help
.venv/bin/python assess_motion_candidate.py --help
```

## 4. 输入视频规范

- 推荐 1080p 或 720p、恒定 30 fps、MP4；
- 相机固定、照明稳定、关闭美颜和自动滤镜；
- 尽量保持正脸并保证额头和双颊可见；
- 每种跑步机速度采集 90～120 s；
- 摄像头和接触式 PPG、Polar 或 ECG 必须同步记录；
- 加速、恒速、减速和恢复阶段分别标记。

原始数据建议放在：

```text
videos/
├── self_test_72bpm.mp4
├── ubfc_subject1/
│   ├── vid.avi
│   └── ground_truth.txt
└── kaggle_motion_subject001/
    ├── video.mov
    └── clips/motion_first60.mp4
```

除合成视频外，`videos/` 内容默认由 `.gitignore` 排除。

## 5. 单个视频提取 rPPG

快捷运行：

```bash
cd /home/fengbujue/项目/rppg识别
./run.sh videos/你的文件.mp4 pos
./run.sh videos/你的文件.mp4 chrom
```

直接调用并自定义窗口：

```bash
.venv/bin/python analyze_rppg.py videos/你的文件.mp4 \
  --method pos \
  --window 10 \
  --step 1 \
  --min-bpm 42 \
  --max-bpm 180 \
  --output results/你的文件/pos
```

每个算法结果目录输出：

- `rppg_waveform.csv`：逐帧时间、RGB 特征和归一化 rPPG；
- `heart_rate.csv`：滑动窗口心率与频谱 SNR；
- `summary.json`：视频、人脸检测和心率统计；
- `rppg_report.png`：波形与心率报告图。

## 6. 72 bpm 合成视频自检

```bash
.venv/bin/python make_self_test_video.py
./run.sh videos/self_test_72bpm.mp4 pos
./run.sh videos/self_test_72bpm.mp4 chrom
```

当前结果：POS 和 CHROM 的中位心率均为 72.070 bpm，相对目标误差约 0.070 bpm。该步骤只验证程序链路，不代表真实人体场景准确度。

## 7. UBFC 真人准静态验证

```bash
./run.sh videos/ubfc_subject1/vid.avi pos
./run.sh videos/ubfc_subject1/vid.avi chrom

.venv/bin/python analyze_ubfc_alignment.py \
  videos/ubfc_subject1/ground_truth.txt \
  --pos-dir results/vid/pos \
  --chrom-dir results/vid/chrom \
  --output results/vid/comparison
```

当前核心结果：

| 指标 | POS | CHROM |
|---|---:|---:|
| 滑窗 HR MAE | 3.92 bpm | 3.18 bpm |
| 滑窗 HR RMSE | 6.34 bpm | 4.80 bpm |
| MAPE | 3.65% | 2.99% |
| Bias | −1.78 bpm | −1.02 bpm |
| 误差在 ±5 bpm 内 | 77.3% | 81.8% |
| 波形相关系数 | \|r\|=0.474 | r=0.533 |
| 基频相干性 | 0.731 | 0.791 |

整段心率估计较准确，CHROM 整体优于 POS，但局部窗口仍会出现约 22～23 bpm 的严重误差。

## 8. 运动视频筛选与伪影分析

整段扫描动作强度：

```bash
.venv/bin/python scan_motion.py \
  videos/kaggle_motion_subject001/video.mov \
  --sample-hz 4 \
  --window 30 \
  --output results/kaggle_motion_subject001/motion_scan
```

对动作片段运行 POS 和 CHROM：

```bash
.venv/bin/python analyze_rppg.py \
  videos/kaggle_motion_subject001/clips/motion_first60.mp4 \
  --method pos \
  --output results/kaggle_motion_subject001/pos_first60

.venv/bin/python analyze_rppg.py \
  videos/kaggle_motion_subject001/clips/motion_first60.mp4 \
  --method chrom \
  --output results/kaggle_motion_subject001/chrom_first60
```

合并动作量和 rPPG 输出：

```bash
.venv/bin/python assess_motion_candidate.py \
  --motion results/kaggle_motion_subject001/motion_scan/motion_timeseries.csv \
  --pos results/kaggle_motion_subject001/pos_first60/heart_rate.csv \
  --chrom results/kaggle_motion_subject001/chrom_first60/heart_rate.csv \
  --output results/kaggle_motion_subject001/motion_effect
```

当前候选视频的高动作窗口相对低动作窗口：POS 心率标准差增至约 2.84 倍，CHROM 增至约 1.68 倍。由于同步 Empatica 真值尚未取得，该数据只能证明运动使结果不稳定，不能计算运动状态下的 MAE 或判断哪个算法的心率更接近真实值。

## 9. 静态与动态统一评价指标

有同步标准信号时，按相同滑动窗口计算：

- MAE、RMSE、MAPE 和 Bias；
- 误差在 ±5 bpm、±10 bpm 内的窗口比例；
- Bland–Altman 95% 一致性界限；
- 波形 Pearson 相关系数和基频相干性；
- SNR、人脸检测率和有效输出率；
- 光流或人脸中心速度与绝对心率误差的关系。

连续窗口误差超过 10 bpm、有效输出率低于 80%、相关系数低于 0.3 或相干性低于 0.4，可作为动态失效候选判据。最终应画出“速度—MAE”“速度—失败窗口比例”和“运动量—绝对误差”曲线。

## 10. 结果目录

```text
results/
├── self_test_72bpm/{pos,chrom}/
├── vid/{pos,chrom,comparison}/
├── ubfc_subject1_motion_scan/
└── kaggle_motion_subject001/
    ├── motion_scan/
    ├── pos_first60/
    ├── chrom_first60/
    └── motion_effect/
```

重要结果文件：

- `results/vid/comparison/detailed_comparison.png`；
- `results/vid/comparison/comparison_metrics.json`；
- `results/vid/comparison/hr_window_comparison.csv`；
- `results/kaggle_motion_subject001/motion_scan/motion_scan.png`；
- `results/kaggle_motion_subject001/motion_effect/motion_vs_rppg.png`；
- `results/kaggle_motion_subject001/motion_effect/motion_effect_summary.json`。

## 11. Git 初始化和首次推送

项目整理完成后，在 WSL 执行：

```bash
cd /home/fengbujue/项目/rppg识别
git init -b main
git add .
git status
git commit -m "Initial rPPG validation pipeline"
git remote add origin https://github.com/bujuefeng267-byte/rppg.git
git push -u origin main
```

首次提交前必须检查 `git status`，确认 `.venv/`、真人原始视频和第三方大文件没有被暂存。GitHub 远程仓库的创建、公开/私有设置以及协作者邀请，需要在登录的 GitHub 账号中完成。

后续同步：

```bash
git status
git add .
git commit -m "描述本次修改"
git push
```

## 12. 数据来源和使用提醒

- 算法路线参考：[pyVHR](https://github.com/phuselab/pyVHR)；
- 静态标准数据：UBFC-rPPG；
- 动作候选：[Kaggle rPPG Dataset](https://www.kaggle.com/datasets/ashfakyeafi/rppg-dataset)；
- Kaggle 页面许可当前为 Unknown，不应把其原视频上传到公开仓库；
- 真人视频和生理数据可能涉及隐私、知情同意与数据集许可；
- 摄像头 rPPG 是研究性估计，不能作为医疗诊断依据。
