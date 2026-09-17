# 收益保留入口与负收益逻辑清理

当前采用 **V28连续PPG + V32独立窗口心率**。新入口统一使用相同规则，不按视频编号或参考心率切换模型。

14组固定对照确认：相对V28，data5的MAE从19.44降到4.93 bpm，其余13组心率不变；整体MAE34.24→33.05 bpm，P5 18.99%→19.33%，心率覆盖率86.40%不变，旧正确窗损伤0。这是已存在V32优势的统一复核与入口整合，不是新发现一个更准确的模型。data6仍约26 bpm误差，后八组大误差尚未解决。V26完整谐波在data5的3.66 bpm仍优于本入口，原代码和产物保留。

本轮新增的POS6周期保护消除了上一轮data6额外退步，却也消除了data12收益，所有HR回到原POS6。因此不在活动入口采用它。无保护计分、因果退出、退出组合、区域替换都不进入此入口。旧冻结实验源码和结果仅作为复现档案保留。

## 使用

在WSL项目根目录，对新视频运行：

```bash
./run_retained_hr.sh --video /absolute/path/video.mp4 --out results/new_retained_run
```

复用已完成的V28结果，无需重新解码视频：

```bash
./run_retained_hr.sh \
  --v28-output results/data1_6_v28_20260912/direct_guard/data5 \
  --trace-cache results/data1_6_20260911/data5/inference/frame_trace.csv \
  --out results/new_data5_retained
```

输出目录必须不存在。原连续模式的`run_recommended.sh`、`run_motion_v28.sh`继续可用；需要本轮验证的窗口心率结果，使用上面新入口。

## 输出与来源

- `result_summary.json`：实际文件路径和来源说明。
- `window_hr/heart_rate.csv`：推荐查看的窗口心率；每行指向实际波形文件。
- `window_hr/windows_manifest.csv`、`window_hr/window_waveforms/`：心率对应的真实10秒波形窗口。相邻窗口可以来自不同信号来源，不能拼成一条连续PPG解释。
- `continuous_v28/waveform.csv`：新视频模式下的原V28连续PPG。复用模式下连续波形保留在原目录，其路径写在结果摘要中。

不填补缺失HR，不把参考心率输入推理，不宣称连续PPG形态或SNR提高。`window_hr/waveform.csv`只是未变的V28上下文，替换心率的信号来源必须查看窗口库。

## 退役入口和复现

`run_candidate_readout_experimental.sh`已停用，不再公开提供未通过采用检查的候选修改。历史全量代码位于`motion_controlled_upgrade_20260917/`，冻结证据在`results/controlled_upgrade_20260917/`；本轮精简实验和独立评价在`motion_positive_refinement_20260917/`及`results/positive_refinement_20260917/`。

21项机制/缺失/评价测试通过。新旧selection在14组结果完全一致；V32重放前六组逐值复现历史结果；70项推理的独立评价、波形样本审核和来源哈希校验通过。数据均是反复开发使用的数据，参考同步仍为估计，不能当成独立泛化验证。
