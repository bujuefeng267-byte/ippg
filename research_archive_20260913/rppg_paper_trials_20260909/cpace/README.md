# cPACE 作者核心的独立视频实验

已经用 V2.2 `trimmed_gap10` 的相同六段原片 ROI RGB 缓存运行 cPACE 两个分支。这里只验证**作者核心在本项目输入与输出协议下的行为**，不称为论文完整协议复现。原片的检测/采样已由 V2.2 完成，本次运行耗时只含缓存读取、核心信号处理与输出。

作者源：<https://github.com/gkaur102/cPACE>，固定提交 `4cd7a438bf9dd30af8ec3afc365dad53861b6c97`，MIT许可证保留在 `author_source/LICENSE`。所有作者 `.py` 未修改；未执行安装脚本，未改既有虚拟环境。`project_estimator/`是 V2.2 `legacy_motion.py` 与 `analyze_rppg.py` 的独立副本，实际哈希记录在冻结协议中。

## 冻结接入规则

- 输入只含 `rppg_motion_v22/validation/trimmed_gap10/{case}/frame_trace.csv` 的RGB、ROI有效标记和原始帧时间，以及对应元数据。没有向推理提供任何心率参考。
- 对每ROI独立执行V2.2的内部短缺口插值，上限`floor(0.10×fps)`帧，不外推，不填长缺口。ROI集合变化时切段；每段需额头及至少一侧面颊。所有作者滤波/seed/投影只在此段内执行。
- 作者默认 `.7–3.0 Hz`、`eigen_bw=.30 Hz`、GREEN种子、跨ROI特征向量选择保持不变。对同段分别运行默认 `homodyne=True` 与 `False`。外部种子参数从未传入；seed随段保存。至少4秒段，输出两端各屏蔽1.6秒，与V2.2边缘政策一致。
- 作者返回多ROI真实波形，没有定义统一的融合波形。因此主`base`固定为`Forehead`，不平均ROI，不按参考选择通道。额头缺失时不会偷偷换ROI，全部ROI另存。
- 默认homodyne先窄带化再作Hilbert包络/相位处理，所以其形态已受限制。它不是根据HR生成的正弦；相位来自真实作者波形。但不能把由窄带化带来的高谱集中度自动称为形态恢复。
- CSV保存后重新读取`base/observed/interpolated`，用冻结V2.2 legacy估计最终10秒窗、1秒步、42–210 bpm局部峰与离线DP。拒绝窗两个最终HR列均为NaN。主HR搜索范围与作者核心42–180 bpm不同，已显式保留，不据结果调整。
- 原生作者`hr_per_roi/hr_consensus_bpm`仅为整段诊断：采用作者内部10秒/2秒窗并跨窗汇总，发生在项目边缘屏蔽之前，不能与最终逐窗HR混为一项指标。

## 输出

| 路径 | 内容 |
|---|---|
| `results/{case}/` | 主分支：作者默认homodyne开启 |
| `results_no_homodyne/{case}/` | 固定消融：作者同一核心关闭homodyne；全部报告，不择优替换主支 |
| 各目录`waveform.csv` | 原始帧时间`time_s`、实际额头波形`base`及`observed/interpolated/covered`；缺失留NaN |
| 各目录`heart_rate.csv` | 从该保存波形重算的局部峰/离线DP、接受状态与覆盖标记 |
| 各目录`roi_waveforms.csv` | 同一次作者核心的全部ROI波形，供诊断，不自动改选主通道 |
| 各目录`segment_diagnostics.json` | ROI/段边界、视频seed、匹配特征向量、每段作者原生HR与PLV |
| 各目录`summary.json` | 帧数/FPS、输入原片及trace哈希、代码哈希、覆盖、限制及输出哈希 |
| `protocol_before_inference.json` | 所有真实推理之前冻结的参数、源码与6条trace哈希、运行命令 |
| `run_summary.json` | `all_six_successful=true`为两分支全部6段完成标记 |
| `output_verification.json` | 所有12套输出的保存波形—HR重算、时间轴、NaN、覆盖及哈希一致性 |

六段case为`user0904, user0907, ubfc, kaggle_full, synthetic72, data1`。没有把同一视频的播放衍生片段另算独立样本，也没有重新选择参考时间偏移。

## 合成自检与已经看到的限制

7项自检通过：视频注入72 bpm可由两支输出72 bpm；NaN长缺口不桥接、跨缺口窗口不输出HR；缺口之后的改变不影响此前段；短缺口保留插值出处；全缺失及缺额头不自动换通道；样本标记与有限波形匹配。

另一个自检故意只施加1.6 Hz共同亮度变化，不注入任何心搏。作者默认核心仍输出约96 bpm，并被统一HR估计接受22窗。这是**被记录的模型限制**，不是“测试证明没有假脉搏”；没有为消除此现象改动原投影数学。原始各通道均值归一化改变共同方向，而作者仍以原始均值方向投影，相关无参考几何检查也已保存。

真实data1的视频GREEN种子为49.1292 bpm；结果原样保留，没有依主参考将其换到100–113或相机约94 bpm。这表明固定公开方法也可能由初始颜色谱选择错误频段，最终优劣需由独立评价脚本在既有冻结参考协议下报告。

## 运行记录

在既有WSL环境中运行（不安装依赖），完整原命令同时记录在协议JSON：

```bash
/home/fengbujue/项目/rppg识别/.venv/bin/python -B "/mnt/c/Users/15011/Documents/ChatGPT/New project/rppg_paper_trials_20260909/cpace/test_cpace_trial.py"
/home/fengbujue/项目/rppg识别/.venv/bin/python -B "/mnt/c/Users/15011/Documents/ChatGPT/New project/rppg_paper_trials_20260909/cpace/run_cpace_trial.py" --run-all
/home/fengbujue/项目/rppg识别/.venv/bin/python -B "/mnt/c/Users/15011/Documents/ChatGPT/New project/rppg_paper_trials_20260909/cpace/verify_outputs.py"
```

脚本拒绝覆盖已存在的结果；需复做应使用新的独立实验目录。作者MIT声明需随再分发源码保留。论文与算法背景见[原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC13372377/)及[DOI](https://doi.org/10.1364/BOE.599752)。
