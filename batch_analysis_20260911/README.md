# data1–6 批量分析与交付脚本

本目录只新增批处理、参考评价、诊断和展示工具。现有 `motion_upgrade_v24_20260910` 推断源码未修改。

主结果：`/home/fengbujue/项目/rppg识别/results/data1_6_20260911/六组视频PPG与心率对照报告.md`。

六组原始 ZIP：`/mnt/c/Users/15011/Desktop/视频/data1.zip` 至 `data6.zip`。

解压输入：`/home/fengbujue/项目/rppg识别/videos/data1_6_20260911/`。

## 运行方法

在 WSL 终端进入项目：

```bash
cd /home/fengbujue/项目/rppg识别
```

`run_batch.py` 使用安装的 V2.4 实验程序和冻结参数运行六段视频，最多并行两段。当前结果已完成，再调用会复用完成记录；发现中断产生的半成品会停止，需检查后使用新的输出目录，避免静默覆盖原结果。该脚本固定服务本次 data1–6，不会自动接收名称相同但内容不同的新一批数据。

```bash
.venv/bin/python -B batch_analysis_20260911/run_batch.py
```

单段直接调用原项目入口（为新运行选择未存在的结果目录）：

```bash
./run_motion_v24_experimental.sh /absolute/path/video.avi --output /absolute/path/new_result
```

参考评价在推断完成后单独运行，不会把真实心率传入模型：

```bash
.venv/bin/python -B batch_analysis_20260911/evaluate_batch.py
```

重新生成六段 PNG/SVG 和完整动态 MP4。`--overwrite` 只替换本批 `presentation` 里的生成展示文件：

```bash
.venv/bin/python -B batch_analysis_20260911/render_results.py --overwrite
.venv/bin/python -B batch_analysis_20260911/assemble_report.py --package
```

独立回算核查（使用新的回执文件，已有回执保留）：

```bash
.venv/bin/python -B batch_analysis_20260911/qa_batch_saved_outputs.py \
  --batch-root results/data1_6_20260911 \
  --qa-core batch_analysis_20260911/qa_recompute_waveform_hr.py \
  --source-root motion_upgrade_v24_20260910 \
  --expected-frames results/data1_6_20260911/reference_audit/expected_frames.json \
  --output results/data1_6_20260911/qa/waveform_replay_qa_repeat.json
```

## 解释口径

- 主预测为 `fusion_heart_rate.csv` 中 `accepted=True` 的 `ridge_bpm`，是离线估计；拒绝行保持空白。
- 主波形为 `fusion_waveform.csv` 的 `base`，是相对幅度。有效数值覆盖率不等于波形生理形态的准确率。
- 参考同步为“原始 ZIP 中视频修改时间－实际解码时长”的估计，默认偏移为 0，另外报告七个预定偏移。不能按最小 MAE 选择时间偏移。
- 主准确率为误差不超过 ±5 bpm 的有效配对窗口占比 P5；同时报告把缺失计入失败的全参考窗口成功率 R5。
- 参考是用户指定的 Polar 设备心率，不是 ECG；PPG 的重建时间字段不自动提供与相机的逐搏同步。
- 图像与视频只展示冻结的保存结果，不补波形、不补心率、不把图像缩放后的帧率用于推断。
- `qa_recompute_waveform_hr.py` 为本次使用的独立审计核心副本，哈希由批量 QA 脚本固定校验；不应直接使用它针对旧六案例的默认命令行。

全部输出、来源哈希、逐窗诊断、采样帧数核对与分享包均保存在项目的本次结果目录。
