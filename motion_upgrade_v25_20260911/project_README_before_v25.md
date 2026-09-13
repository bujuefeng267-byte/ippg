# rPPG 识别基线

本目录用于验证：普通摄像头人脸视频可以通过现有开源 rPPG 方法得到估计波形和心率，并观察运动时的失效情况。

当前默认采用 POS，另提供 CHROM。算法路线参考开源项目 [pyVHR](https://github.com/phuselab/pyVHR)；人脸跟踪使用 MediaPipe。这里输出的是摄像头估计的 rPPG/BVP，不是接触式传感器测得的真实 PPG。

## 目录

- `videos/`：放入原始 MP4 视频
- `results/`：每段视频的 CSV、图像和摘要
- `.venv/`：隔离的 Python 环境

## 运行

```bash
cd /home/fengbujue/项目/rppg识别
./run.sh videos/static.mp4 pos
./run.sh videos/static.mp4 chrom
```

每次运行产生：

- `rppg_waveform.csv`：逐帧 RGB 和归一化 rPPG 波形
- `heart_rate.csv`：滑动窗口心率与频谱信噪比
- `summary.json`：帧率、时长、人脸检出率、心率统计
- `rppg_report.png`：波形和心率图

## 建议视频规格

- 1080p 或 720p，恒定 30 fps，MP4
- 正脸、稳定照明、固定相机、关闭美颜
- 每段 90–120 秒；所有速度保持相同机位和光照

## 自检

项目附带一段无隐私的合成测试视频，目标频率是 72 bpm：

```bash
.venv/bin/python make_self_test_video.py
./run.sh videos/self_test_72bpm.mp4 pos
```

真实实验仍需 Polar、指夹血氧仪或 ECG 同步记录，才能计算绝对心率误差；只有摄像头视频时可以验证可提取性、稳定性和运动失效趋势。

## 2026-09-08：运动监测多区域实验版 V2

新增额头、左右脸颊独立采样、区域共识、歧义拒绝和失锁重捕获；入口为 `./run_motion_v2.sh VIDEO --output NEW_DIRECTORY`。一次运行同时输出 POS/CHROM 对照和 fusion 实验分支。

全量验证的 UBFC 共同窗口参考基频 SNR 相对旧 POS 配对增益中位数为 +1.118 dB，但 fusion MAE 从旧 POS DP 的 2.561 增至 3.660 bpm，R5 从 88.64% 降至 54.55%。新 fusion 尚不适合替换默认算法，原 `run.sh` 保留。

详见 [代码运行说明](motion_upgrade_v2_20260908/README_运动PPG升级.md) 和 [指标变化报告](results/motion_upgrade_v2_20260908/rPPG运动监测_V2升级与指标变化_2026-09-08.md)。完整链路仍是离线处理，用户运动视频缺少同步参考时不计算绝对精度。


## 2026-09-09：运动 rPPG V2.1 独立实验入口

使用 `./run_motion_v21.sh VIDEO --output NEW_DIRECTORY` 调用新版本。最终心率从保存的融合波形估计；零相位滤波、重叠融合和 DP 均为离线处理，DP 使用未来窗口。区域共识频率另存为提议。

本次保留视频的验证属于代码回归，独立新数据验证仍待完成。详见[运行说明](motion_upgrade_v21_20260909/README.md)、[代码修复与指标变化报告](results/motion_upgrade_v21_20260909/rPPG_V2.1代码修复与指标变化_2026-09-09.md)和[完整评价结果](results/motion_upgrade_v21_20260909/evaluation/summary.json)。原 `run.sh`、`run_motion_v2.sh` 及旧代码和结果均保留。


## 2026-09-09：V2.2 采样与短缺帧实验

`./run_motion_v22.sh VIDEO --output NEW_DIRECTORY` 使用固定 10% 截尾均值采样，默认短缺帧上限 0.10 秒；`--max-gap 0.15` 是单列验证的连续性选项。最终心率仍从实际保存的波形估计，全链路为离线处理。

详见 [V2.2 运行说明](motion_upgrade_v22_20260909/README.md) 与 [全部视频比较及研究路线](results/motion_v22_20260909/优化结果与研究路线.md)。这是已看过资料的回归实验；是否涨幅以逐片表格为准。原有各版本入口和历史结果保留。
