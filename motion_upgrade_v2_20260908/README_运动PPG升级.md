# 运动中 rPPG 监测：V2 多区域实验版

本版新增独立皮肤区域采样、多区域频率共识、拒绝歧义与失锁重捕获、实际波形融合及统一评价。目标是改善运动中的脉搏信号质量。全量验证发现：UBFC 参考基频 SNR 有小幅改善，但心率误差与覆盖退步，因此 V2 fusion 是实验分支，不能作为已验证优于旧版的默认算法。

## 运行

在 WSL 项目中使用新增入口，输出目录必须尚不存在：

```bash
cd /home/fengbujue/项目/rppg识别
./run_motion_v2.sh videos/self_test_72bpm.mp4 --output results/my_v2_test
./run_motion_v2.sh videos/ubfc_subject1/vid.avi --output results/my_v2_ubfc --reference-ubfc videos/ubfc_subject1/ground_truth.txt
```

`--reference-ubfc` 仅支持该数据格式，参考值在全部预测完成后用于评分；不参与采样、候选选择或融合。其他视频没有同步真值时不要传入不配对的参考文件。

原 `run.sh` 入口继续可用。本入口同时保存原有 POS/CHROM 对照分支及新的 fusion 实验分支，不自动将 fusion 选为最优结果。

## 文件与输出

| 文件 | 作用 |
|---|---|
| `motion_frontend.py` | 保存额头、左脸颊、右脸颊独立 RGB、采样有效标记及质量代理值；沿用旧人脸跟踪与短时光流桥接 |
| `motion_fusion.py` | 每个区域独立 POS/CHROM，跨区域共识与前向候选状态，融合实际宽带输入波形 |
| `analyze_motion_v2.py` | 运行入口，生成所有分支，检查缓存身份、帧数与 SHA-256 |
| `analyze_rppg.py`、`legacy_motion.py` | 冻结的旧版信号处理与评分，用于对照 |
| `test_frontend_v2.py`、`test_fusion_v2.py`、`test_cli_v2.py` | 21 项采样、融合与入口回归测试 |
| `frame_trace.csv` / `.json` | 独立区域采样及视频身份、缓存校验 |
| `roi_waveforms.csv` | 三个区域的 POS/CHROM 波形 |
| `pos_heart_rate.csv`、`chrom_heart_rate.csv` | 同一采样输入下的旧逻辑对照 |
| `fusion_heart_rate.csv` | 融合候选、接受/拒绝原因、状态、参与区域数、是否自身生成波形 |
| `fusion_waveform.csv` | 真实输入信号融合的结果，非按估计心率生成的正弦波 |
| `fusion_diagnostics.csv` | 每窗每通道候选峰、支持度及运动频率重叠情况 |
| `summary.json` | 参数、源文件哈希、运行时间、参考身份和窗口覆盖 |

fusion 表中的 `ridge_bpm` 为兼容字段名，含义是前向共识候选；POS/CHROM 的同名字段是利用未来窗口的离线 DP。`time_s` 是窗口中心，完整窗口到 `window_end_s` 才齐备。整条链路仍使用离线零相位滤波和重叠波形融合，不能据此前向状态宣称实时监测。

`accepted` 表示通过工程规则，不表示真值正确。`quality_proxy` 不是准确概率。`waveform_generated` 表示该窗确实提供了至少两个区域的融合波形；邻窗重叠可能使没有自行生成的窗也有有限波形，两者应分开统计。

## 固定比较方法

使用相同完整视频、相同约 10 秒窗口/约 1 秒步长、相同 42–210 bpm 搜索范围，保留拒绝和缺失窗口。实际帧数为 `round(window_s * fps)`；UBFC 实际窗长 10.0123 秒，参考心率仍按旧协议窗口中心 ±5 秒求平均。全部参数在真人参考评分前冻结，未根据本次结果搜索阈值。

同时查看参考 H1 SNR、MAE/RMSE、±5 bpm 命中比例 P5、参考窗口正确输出比例 R5 和输出覆盖。既比较各自接受窗口，也比较共同窗口。SNR 使用参考基频 ±0.1 Hz、0.7–3.5 Hz 总带宽和 Hann/8192 点频谱；不把算法自己选中的谱峰当参考，不等于绝对运动伪影能量或形态还原质量。

## 复核

部署目录内运行测试：

```bash
cd /home/fengbujue/项目/rppg识别
.venv/bin/python -m unittest discover -s motion_upgrade_v2_20260908 -p 'test_*v2.py' -v
```

本次完整评估保存在项目 `results/motion_upgrade_v2_20260908/`。独立评价器及旧基线配套目录保留在 Windows 工作区 `C:/Users/15011/Documents/ChatGPT/New project/`，从该工作区经 WSL 重算：

```bash
/home/fengbujue/项目/rppg识别/.venv/bin/python '/mnt/c/Users/15011/Documents/ChatGPT/New project/rppg_motion_v2/evaluate_v2.py' --output '/mnt/c/Users/15011/Documents/ChatGPT/New project/rppg_motion_v2/evaluation_recheck'
```

`synthetic_motion_benchmark.py` 是 RGB 混合信号仿真，不能替代真人运动视频验证。其输出新建于脚本同级 `synthetic_stress/`，不覆盖已有结果。已知脉搏频率下的零误差也不能证明分离出了真实生理脉搏形态。

最小窗口为 6 秒；若按小数帧率取整不足 6 秒，会向上取 1 帧并在 summary 中同时记录 requested_window_s 和 effective_window_s。短于窗口的视频输出空心率表及 NA 误差，不编造结果。

