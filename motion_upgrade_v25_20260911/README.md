# V2.5：保留波形覆盖的心率判断升级

目的：在运动视频中减少错误频率锁定，保留真实波形输出。推荐配置为原 V2.4 波形融合与分支选择，加上新多候选心率判断；像素跟踪、局部异常筛选、跟踪失败回退全部保留。

## 运行新视频

```bash
cd /home/fengbujue/项目/rppg识别
./run_motion_v25.sh videos/your_video.avi --output results/your_video_v25
```

输出目录必须不存在。默认窗口 10 秒、步长 1 秒、心率范围 42–210 bpm、短缺帧上限 0.10 秒。原分辨率、原帧率解码，不因展示而降低分析帧率。`run.sh` 和所有历史入口保留；本入口是六段开发录像验证后的推荐研究配置，不是已经完成独立临床或人群验证的模型。

实际启用参数：

```text
--pixel-mode tracking_screened --algorithm-mode guarded_fusion
--max-gap 0.10 --motion-evidence legacy --hr-mode evidence --routing-mode legacy
```

这里的 `--motion-evidence legacy` 只控制波形融合阶段。`--hr-mode evidence` 内仍启用新的可靠运动证据：运动观测比例、以脸高/秒计的绝对速度、连续段运动频谱，未知运动不会被标为静止。

## 三步修改与采用决定

1. **运动证据**：修正仅按归一化运动峰判断干扰的问题，区分微小抖动、大幅运动及缺失。波形融合阶段的 `reliable` 模式暂为实验选项：data6 改选候选后只剩一个相互一致的贡献 ROI，损失波形覆盖。
2. **心率多候选与重获**：在真实保存波形上比较多个具有实际频谱能量的峰，结合周期、自相关、二/三次谐波和软运动惩罚。DP 保留连续性，同时允许有代价的轨迹重获。不会直接把心率乘二/除二，或按参考心率选峰。此步采用。
3. **分支选择**：新增对称证据选择与未决状态，允许在大频率分歧时选择任一分支。测试中覆盖损失偏大，暂保留为 `--routing-mode evidence` 实验选项。

全部三步实验入口：

```bash
./run_motion_v25_all_experimental.sh videos/your_video.avi --output results/your_video_v25_all
```

不要把全开模式当成最佳版本。六段已看过的录像用于开发选择，并非独立测试集。

## 主要输出

| 文件 | 内容 |
|---|---|
| `fusion_waveform.csv` | 原视频逐帧时间、实际融合 rPPG 波形 `base`、覆盖及观测/插值来源；无效值为空/NaN，非零值填充 |
| `fusion_heart_rate.csv` | 每个计划窗口的最终 `ridge_bpm`、`accepted`、`status`，以及全部心率候选证据 |
| `fusion_proposals.csv` | ROI 共识频率提议，不能作为最终心率结果 |
| `fusion_diagnostics.csv` | ROI 和方法的贡献、权重、候选质量 |
| `branch_routing.csv` | 基线/跟踪分支选择原因 |
| `frame_trace.csv` / `.json` | 逐帧采样、像素跟踪来源、运动信息和精确缓存校验 |
| `summary.json` | 实际参数、源码哈希、覆盖统计 |
| `comparison.png` | 采样与心率概览；六段本次 PPG/心率折线图另见结果目录的 presentation |

`raw_spectral_peak_bpm` / `spectral_peak_bpm` 是最大频谱峰；`evidence_local_bpm` 是新增证据评分的局部首选；`ridge_bpm` 才是最终离线轨迹。`legacy_ridge_bpm` 仅为诊断。工程分数、候选权重、覆盖率均不是准确概率。

最终频率必须由当前保存波形中的局部峰支持：相对功率至少 5%、峰突出度至少 2%，DP 可选择峰周围 ±3 bpm 内且自身相对功率至少 5% 的格点。保留原采样、平坦/弥散频谱和波形贡献门槛；无输出的窗口不保持上一心率、不插值补心率。

## 验证与局限

详见 [六段逐步修改与指标变化报告](../results/data1_6_v25_20260911/V25代码优化与六视频指标变化.md)。六段使用同一原始前端缓存（视频身份、帧数、逐帧内容哈希、前端源码均验证），重新计算波形、候选和心率；本次复用前端并非重新解码视频。

推荐版与 V2.4 逐样本波形在数值精度内一致（复算容差 1e-8），覆盖及接受窗掩码完全相同；六段合并 MAE 从 39.16 降为 25.20 bpm，R5 从 24.27% 升为 30.10%，覆盖率不变。data4 的 ±5 bpm 窗数从 46 降为 45，平均误差虽降低，也不能称为每项指标都提升。data1 未改善，data3/5/6 仍有明显误差。

参考为用户提供的 Polar 设备心率；相机与参考没有共享硬件同步，沿用原文件时间估计和七个固定偏移敏感性检查，不寻找最低误差的偏移。波形本身未变化，本轮没有证明波形形态、SNR 或逐搏准确度提升。零相位滤波、重叠融合、DP 都属于离线处理，不能把结果当成实时性能。

合成与回归测试：

```bash
.venv/bin/python -B -m unittest discover -s motion_upgrade_v25_20260911 -p 'test_*.py'
```

`development_protocol.json` 是最初三步的开发协议。`run_preserved_waveform.py` 是看到三步结果后增加的受控消融：原评分参数、同步规则、验收门槛不变。完整五阶段都保存在结果目录，没有按录像编号定制算法参数。
