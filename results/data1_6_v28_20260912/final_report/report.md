# V28：保留 V26 保守融合收益，减少误替换

**通过六视频开发集的既定升级条件。** 相比 V25，总体 MAE 从 25.20 降至 17.42 bpm，下降 7.78 bpm（30.87%）；±5 bpm 达标输出从 93/260 增至 113/260，六段视频的心率和波形覆盖率均保持 V25 水平。

**本轮保留的是 V26 保守融合的收益，未保留完整谐波分支的全部优势。** data5 保留 19.44 bpm 的 MAE，完整谐波分支在该视频曾达到 3.66 bpm；完整谐波分支总体达标率、覆盖率和 RMSE 仍优于本轮。当前尚未达到“基本所有心率都在 ±5 bpm 内”，data1、data3、data6 仍需继续改进。

## 四个固定版本的总体结果

| 版本 | MAE↓ bpm | RMSE↓ bpm | P5↑ % | R5↑ % | HR覆盖↑ % | 波形时间覆盖↑ % | 达标/输出 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| V25 | 25.20 | 37.99 | 35.77 | 30.10 | 84.14 | 88.37 | 93/260 |
| V26 完整谐波 | 18.75 | 26.63 | 45.63 | 38.83 | 85.11 | 89.19 | 120/263 |
| V26 保守融合 | 20.28 | 30.95 | 43.46 | 36.57 | 84.14 | 88.37 | 113/260 |
| V28 直接运动保护 | 17.42 | 27.86 | 43.46 | 36.57 | 84.14 | 88.37 | 113/260 |

P5 = ±5 bpm 达标窗口数 / 有心率输出且有有效参考的窗口数。R5 = 达标窗口数 / 全部有参考的计划窗口数；缺失输出不算成功。心率覆盖率 = 输出窗口数 / 全部计划窗口数。本批四版本均有 309 个参考合格计划窗口。总体 MAE 按有效窗口汇总，未对六个视频 MAE 简单平均。

波形时间覆盖率按每段有限样本数 / 原始 fps 累加，再除以总视频时长；data4 约 180 fps，其余约 30 fps，因此不能把总体帧覆盖率当成总体时间覆盖率。V28 的总体帧覆盖率另为 90.26%。波形覆盖表示有数值可输出，不等同于真实 PPG 形态恢复率。

## 六段视频全部展示

| 视频 | V25 MAE | V26完整谐波 MAE | V26保守融合 MAE | V28 MAE |
| --- | --- | --- | --- | --- |
| data1 | 13.10 | 32.66 | 33.21 | 13.10 |
| data2 | 3.02 | 2.15 | 3.02 | 3.02 |
| data3 | 32.87 | 41.66 | 32.87 | 32.87 |
| data4 | 2.56 | 2.38 | 2.56 | 2.56 |
| data5 | 60.72 | 3.66 | 19.44 | 19.44 |
| data6 | 26.00 | 20.89 | 26.00 | 26.00 |

以上 MAE 单位均为 bpm。下表为最终 V28；所有指标均来自同一组保存输出，没有逐视频挑选最优版本。

| 视频 | P5 % | R5 % | HR覆盖 % | 波形覆盖 % | 达标/输出/计划 |
| --- | --- | --- | --- | --- | --- |
| data1 | 0.00 | 0.00 | 88.10 | 88.75 | 0/37/42 |
| data2 | 87.88 | 78.38 | 89.19 | 90.45 | 29/33/37 |
| data3 | 18.64 | 16.92 | 90.77 | 91.85 | 11/59/65 |
| data4 | 97.83 | 90.00 | 92.00 | 92.58 | 45/46/50 |
| data5 | 57.14 | 52.83 | 92.45 | 93.25 | 28/49/53 |
| data6 | 0.00 | 0.00 | 58.06 | 75.42 | 0/36/62 |

data1 恢复 V25 的 13.10 bpm，避免 V26 保守融合的 33.21 bpm 退步；data5 保留 V26 保守融合的 19.44 bpm，相比 V25 的 60.72 bpm 改善；data2、data3、data4、data6 的有效心率值和覆盖掩码与 V25 保持一致。因此本轮改善来自 data5 收益的保留和 data1 误替换的阻止，未解决 data3、data6 原有误差。

## 代码只改变一个判断

V26 保守融合原本使用旧心率处的综合运动风险（直接频率、双倍频、半频的加权最大值）决定是否替换波形。V28 在原判断全部通过后，还要求旧心率频率本身的直接运动风险达到 **同一个已有阈值 0.50**。未增加新阈值、未按视频调参，也未把参考心率输入推理。

原规则继续保留：综合旧风险 ≥0.50、候选综合风险至少降低 0.30、至少两个不同物理面部区域、完整有限窗口、原 V25 缺失掩码、正 Hann 权重的凸混合，以及按实际贡献来源记录采样标记。新门只否决原有替换，不在原本拒绝的位置新增替换。最终心率从重新保存并读取的融合波形计算，未直接复制候选心率，也未用心率数值合成正弦波。

| 视频 | V26可替换窗口 | V28可替换窗口 | 直接运动证据不足而否决 |
| --- | --- | --- | --- |
| data1 | 15 | 0 | 15 |
| data2 | 0 | 0 | 0 |
| data3 | 0 | 0 | 0 |
| data4 | 0 | 0 | 0 |
| data5 | 26 | 26 | 0 |
| data6 | 0 | 0 | 0 |

替换窗口使用完整 10 秒 Hann 权重叠加，因此“可替换窗口数”不等于改变后的心率窗口数或替换秒数。缺乏直接运动证据只说明本轮不应执行该替换，不能据此证明原心率正确。

## 升级检查及仍未达到的目标

| 事先约定的条件（相对 V25） | 结果 |
| --- | --- |
| 总体 MAE 下降 | 通过 |
| 总体 R5 至少增加 5 个百分点 | 通过 |
| 每段 MAE 增加不超过 3 bpm | 通过 |
| 每段 P5、R5 各下降不超过 3 个百分点 | 通过 |
| 每段 HR、波形覆盖率各下降不超过 3 个百分点 | 通过 |
| 共同有效窗口 MAE 下降 | 通过 |

V28 与 V25 有相同的 260 个有效比较窗口，本次没有新增或丢失心率输出。目前仅 1/6 段视频达到 P5≥95%。升级条件通过只支持继续开发使用；未证明已实现高准确率或对新视频同样有效。

## 心率折线图

主图比较 V25、V26 保守融合和 V28；黑线为 Polar 参考心率，灰带为参考 ±5 bpm，不是置信区间。各面板使用相同心率纵轴。多个版本重合时曲线会覆盖，完整数值保存在比较 CSV。

![六视频心率：V25、V26保守融合、V28](<//wsl.localhost/Ubuntu/home/fengbujue/项目/rppg识别/results/data1_6_v28_20260912/final_report/six_video_hr_v25_v26conservative_v28.png>)

另图完整保留 V26 谐波分支，便于同时查看本轮保住的结果和放弃的收益。

![六视频心率：V26完整谐波、V28](<//wsl.localhost/Ubuntu/home/fengbujue/项目/rppg识别/results/data1_6_v28_20260912/final_report/six_video_hr_v26harmonic_v28.png>)

## PPG 波形与逐视频输出

下图直接绘制最终 waveform.csv 的 base 样本，不增加平滑、补零或跨缺失插值；NaN 显示为断线。各视频保留自己的幅度尺度，不用于跨视频比较脉搏振幅。保存波形为测得信号的凸混合，其中可以包含 V26 从实际光学信号中选出的窄带分量。窄带处理会影响形态，因此不能把外观更规则解释为生理 PPG 形态更准确。目前只有心率参考，没有同步 PPG 波形真值，未报告形态相关系数或 SNR 的改善。

![六视频最终PPG波形](<//wsl.localhost/Ubuntu/home/fengbujue/项目/rppg识别/results/data1_6_v28_20260912/final_report/six_video_v28_ppg.png>)

| 视频 | 可直接分享的PPG＋心率图 | 原始波形CSV | 原始心率CSV |
| --- | --- | --- | --- |
| data1 | [打开图](<//wsl.localhost/Ubuntu/home/fengbujue/项目/rppg识别/results/data1_6_v28_20260912/final_report/data1_v28_ppg_heart_rate.png>) | [waveform.csv](<//wsl.localhost/Ubuntu/home/fengbujue/项目/rppg识别/results/data1_6_v28_20260912/direct_guard/data1/waveform.csv>) | [heart_rate.csv](<//wsl.localhost/Ubuntu/home/fengbujue/项目/rppg识别/results/data1_6_v28_20260912/direct_guard/data1/heart_rate.csv>) |
| data2 | [打开图](<//wsl.localhost/Ubuntu/home/fengbujue/项目/rppg识别/results/data1_6_v28_20260912/final_report/data2_v28_ppg_heart_rate.png>) | [waveform.csv](<//wsl.localhost/Ubuntu/home/fengbujue/项目/rppg识别/results/data1_6_v28_20260912/direct_guard/data2/waveform.csv>) | [heart_rate.csv](<//wsl.localhost/Ubuntu/home/fengbujue/项目/rppg识别/results/data1_6_v28_20260912/direct_guard/data2/heart_rate.csv>) |
| data3 | [打开图](<//wsl.localhost/Ubuntu/home/fengbujue/项目/rppg识别/results/data1_6_v28_20260912/final_report/data3_v28_ppg_heart_rate.png>) | [waveform.csv](<//wsl.localhost/Ubuntu/home/fengbujue/项目/rppg识别/results/data1_6_v28_20260912/direct_guard/data3/waveform.csv>) | [heart_rate.csv](<//wsl.localhost/Ubuntu/home/fengbujue/项目/rppg识别/results/data1_6_v28_20260912/direct_guard/data3/heart_rate.csv>) |
| data4 | [打开图](<//wsl.localhost/Ubuntu/home/fengbujue/项目/rppg识别/results/data1_6_v28_20260912/final_report/data4_v28_ppg_heart_rate.png>) | [waveform.csv](<//wsl.localhost/Ubuntu/home/fengbujue/项目/rppg识别/results/data1_6_v28_20260912/direct_guard/data4/waveform.csv>) | [heart_rate.csv](<//wsl.localhost/Ubuntu/home/fengbujue/项目/rppg识别/results/data1_6_v28_20260912/direct_guard/data4/heart_rate.csv>) |
| data5 | [打开图](<//wsl.localhost/Ubuntu/home/fengbujue/项目/rppg识别/results/data1_6_v28_20260912/final_report/data5_v28_ppg_heart_rate.png>) | [waveform.csv](<//wsl.localhost/Ubuntu/home/fengbujue/项目/rppg识别/results/data1_6_v28_20260912/direct_guard/data5/waveform.csv>) | [heart_rate.csv](<//wsl.localhost/Ubuntu/home/fengbujue/项目/rppg识别/results/data1_6_v28_20260912/direct_guard/data5/heart_rate.csv>) |
| data6 | [打开图](<//wsl.localhost/Ubuntu/home/fengbujue/项目/rppg识别/results/data1_6_v28_20260912/final_report/data6_v28_ppg_heart_rate.png>) | [waveform.csv](<//wsl.localhost/Ubuntu/home/fengbujue/项目/rppg识别/results/data1_6_v28_20260912/direct_guard/data6/waveform.csv>) | [heart_rate.csv](<//wsl.localhost/Ubuntu/home/fengbujue/项目/rppg识别/results/data1_6_v28_20260912/direct_guard/data6/heart_rate.csv>) |

## 参考对齐与适用范围

参考为 Polar 设备约每秒推送的心率数值，不是本程序直接从 ECG 波形重算的心率。沿用原有 ZIP 内时间戳与实际视频时长得到的估计起点，没有硬件同步标记；采用同一套 10 秒窗口、约 1 秒步长与原始采样时钟。data4 以实际解码 10693 帧计算，而非容器声明的 10702 帧。

这些视频此前已被多轮分析，属于开发集；重叠窗口不是独立样本。这轮只运行事前冻结的一个 V28 修改，未在看到新评分后调整阈值。仍应使用独立新拍、能确认同步的视频检验泛化效果。

以下保留全部约定的参考平移敏感性结果，没有选择其中最优平移替代主结果。

| 参考平移 s | 有效窗口 | MAE bpm | P5 % | R5 % |
| --- | --- | --- | --- | --- |
| -5 | 260 | 17.22 | 34.62 | 29.13 |
| -2 | 260 | 17.32 | 38.85 | 32.69 |
| -1 | 260 | 17.38 | 40.38 | 33.98 |
| 0 | 260 | 17.42 | 43.46 | 36.57 |
| 1 | 260 | 17.45 | 44.23 | 37.22 |
| 2 | 260 | 17.47 | 44.62 | 37.54 |
| 5 | 260 | 17.42 | 48.46 | 41.04 |

## 文件与核验

独立评分核验通过，共检查 2271 个数值比较，最大绝对差为 1.42e-14。核验包括固定参考、缺失分母、各视频指标和全部七个平移条件。推理输出已验证只采用真实保存波形、保留原 NaN 掩码，并从该波形独立重算心率。

- [正式评价 JSON](<//wsl.localhost/Ubuntu/home/fengbujue/项目/rppg识别/results/data1_6_v28_20260912/direct_guard/evaluation_summary.json>)
- [独立核验 JSON](<//wsl.localhost/Ubuntu/home/fengbujue/项目/rppg识别/results/data1_6_v28_20260912/qa_evaluation_v28.json>)
- [四版本总体指标 CSV](<//wsl.localhost/Ubuntu/home/fengbujue/项目/rppg识别/results/data1_6_v28_20260912/final_report/four_version_pooled_metrics.csv>)
- [四版本逐视频指标 CSV](<//wsl.localhost/Ubuntu/home/fengbujue/项目/rppg识别/results/data1_6_v28_20260912/final_report/four_version_case_metrics.csv>)
- [六视频逐窗口心率比较 CSV](<//wsl.localhost/Ubuntu/home/fengbujue/项目/rppg识别/results/data1_6_v28_20260912/final_report/six_video_hr_comparison.csv>)
- [替换窗口统计 CSV](<//wsl.localhost/Ubuntu/home/fengbujue/项目/rppg识别/results/data1_6_v28_20260912/final_report/routing_comparison.csv>)
- [全部参考平移敏感性 CSV](<//wsl.localhost/Ubuntu/home/fengbujue/项目/rppg识别/results/data1_6_v28_20260912/final_report/alignment_sensitivity_all_cases.csv>)
- [事前冻结协议](<//wsl.localhost/Ubuntu/home/fengbujue/项目/rppg识别/results/data1_6_v28_20260912/protocol_before_run.json>)

报告与图由 report_v28.py 根据已保存结果生成；报告不参与算法决策。
