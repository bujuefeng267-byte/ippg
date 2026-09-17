# 运动视频 rPPG 识别与验证

本目录用于验证：普通摄像头人脸视频可以通过现有开源 rPPG 方法得到估计波形和心率，并观察运动时的失效情况。

早期 `run.sh` 基线采用 POS，另提供 CHROM；当前保留方案使用下方的 `run_retained_hr.sh` 入口。算法路线参考开源项目 [pyVHR](https://github.com/phuselab/pyVHR)；人脸跟踪使用 MediaPipe。这里输出的是摄像头估计的 rPPG/BVP，不是接触式传感器测得的真实 PPG。

## 当前保留方案：V28 连续波形＋V32 窗口心率（2026-09-17）

新视频使用 `run_retained_hr.sh`。它保留 V28 的连续 rPPG 波形，同时从各自保存的实测十秒窗口输出 V32 心率。心率窗口不能拼接成一条新的连续 PPG。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
./run_retained_hr.sh --video /完整路径/视频.mp4 --out results/新视频_retained
```

优先查看 `result_summary.json`、`window_hr/heart_rate.csv` 和 `window_hr/windows_manifest.csv`。输出目录必须尚不存在。原 `run_recommended.sh` 保持 V28 连续模式，窗口心率方案使用上面的明确入口。

在同一批14段开发录像上，保留方案相对 V28 的 MAE 为 **34.24 → 33.05 bpm**，±5 bpm 内比例 **18.99% → 19.33%**，心率输出覆盖率均为 **86.40%**。收益主要来自 data5；后八段的大误差尚未解决。这不是14段视频上都最好的统一算法：V26 在 data5 更好，POS6 在 data8 更好，因此两条对照仍保留，不按参考心率自动挑选版本。

[运行与版本选择](当前推荐版本.md) · [发布说明与统一对照](docs/retained_release_20260917.md) · [保留入口输出定义](motion_positive_refinement_20260917/README.md)

## V28 连续模式与历史对照（2026-09-13）

保留 V28 的 10 秒心率窗口和原质量门槛。V29、V30、V31 是独立实验，均未替换推荐版本；V31 的保护实验保住了原结果，但六段开发视频的最终波形、心率和精度没有提升。

在 Ubuntu / WSL 中，先在仓库根目录建立 `.venv` 并按 `requirements.txt` 安装依赖，再运行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
./run_recommended.sh --video /完整路径/视频.mp4 --out results/新视频_v28
```

输出目录应尚不存在。输入应为能完整解码的恒定帧率人脸视频。详见 [V28 输入规格、运行方法与输出字段](motion_upgrade_v28_20260912/README_video_v28.md)。`run.sh` 是早期示例入口，不代表当前推荐算法。

| 六视频 V28 开发集指标 | 结果 |
|---|---:|
| 心率 MAE | 17.42 bpm |
| 心率 RMSE | 27.86 bpm |
| ±5 bpm 达标率（有输出且有参考的窗口） | 43.46%（113/260） |
| ±5 bpm 有效覆盖率（全部参考窗口） | 36.57%（113/309） |
| 心率输出覆盖率 | 84.14%（260/309） |
| 波形时间覆盖率 | 88.37% |

六段录像已用于反复开发，参考沿用估计时间对齐；这些是开发集回归结果，不能代表独立新视频精度。完整链路为离线处理。输出波形或高覆盖率本身不能证明测量准确。

### 最新代码、报告与图

- [当前推荐版本](当前推荐版本.md) · [V28 算法说明](V28运动心率改进说明.md)
- [V28 六视频报告](results/data1_6_v28_20260912/final_report/report.md) · [六视频 PPG 折线图](results/data1_6_v28_20260912/final_report/six_video_v28_ppg.png) · [六视频心率比较图](results/data1_6_v28_20260912/final_report/six_video_hr_v26harmonic_v28.png)
- [V28 精度排查与 V30 实验](V28精度排查与改进实验_20260912.md) · [突降原因核查](突降原因核查_20260912.md)
- [V31 保护机制与六视频结果](V31保护机制与六视频结果_20260912.md) · [V31 实验源码](motion_upgrade_v31_20260912/) · [V31 完成的独立核验](results/data1_6_v31_20260912/qa_evaluation_v31_completed.json)
- [V26](V26运动心率改进说明.md) · [V27](V27运动心率改进说明.md) · [V29 短窗口实验](V29短窗口与连续性说明.md)
- [借鉴的 GitHub 项目、论文与来源总清单](rPPG_GitHub来源总清单_2026-09-11.md)

- [历史调研、指标解释和论文实测归档](research_archive_20260913/README.md)

### 仓库同步范围

同步项目源码、测试、启动入口、分析报告、导出的数值结果、折线图及实验核验记录。原始人脸视频、原始传感器参考文件、含人脸的检查截图、虚拟环境和下载的模型权重仅保留本地。合成测试视频可随仓库分发。

历史冻结清单和部分报告保留当时的绝对路径及校验和，不因同步重写。GitHub 在线阅读请从本页相对链接进入；重跑旧批量实验仍需本地原始数据及原实验路径。分析新的输入视频可使用上面的 `run_retained_hr.sh` 保留方案入口。

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


## 2026-09-11：V2.5 保留覆盖的心率判断升级

当时的 V2.5 实验入口为 `./run_motion_v25.sh VIDEO --output NEW_DIRECTORY`：保留 V2.4 像素跟踪、局部异常筛选、失败回退和实际波形，启用带运动证据的多候选心率与轨迹重获。

六段开发录像的合并 MAE 39.16 → 25.20 bpm，R5 24.27% → 30.10%；每段波形、HR 覆盖率均保持不变。data1 未改善，data4 的 ±5 bpm 窗数 46 → 45，data3/5/6 仍存在明显误差。参考沿用估计时间同步；本次不是独立泛化验证，完整链路仍为离线处理。

三步全部开启的 `run_motion_v25_all_experimental.sh` 在本批覆盖较差，仅作实验。原 `run.sh` 与所有历史版本均保留。

[V2.5 运行说明](motion_upgrade_v25_20260911/README.md) · [逐步修改与六视频指标变化](results/data1_6_v25_20260911/V25代码优化与六视频指标变化.md) · [六组 PPG/心率折线图](results/data1_6_v25_20260911/presentation/all_six_ppg_hr.png)
