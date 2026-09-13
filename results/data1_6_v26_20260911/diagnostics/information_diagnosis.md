# V25 可用信号信息诊断（参考条件 oracle，绝非算法成绩）

只检查保存波形中是否存在与参考频率相容的候选。所有 oracle 结果都借助参考定位，不能用作实际准确率，也没有生成或替换任何预测。

|片段|参考窗|实际 ±5 bpm|最终已有候选支持上限|融合波形实际峰上限|任意原 ROI 实际峰上限|至少2个物理 ROI|至少2 ROI且峰≥20%|
|---|---:|---:|---:|---:|---:|---:|---:|
|data1|42|0|13|5|35|27|14|
|data2|37|29|31|29|33|33|33|
|data3|65|11|34|20|60|51|39|
|data4|50|45|46|44|46|46|46|
|data5|53|8|39|30|49|44|41|
|data6|62|0|16|10|38|29|11|

“已有候选支持”包含峰周围±3 bpm且有实际能量的允许状态；其他“实际峰”列要求局部峰中心距参考≤5 bpm，两者不同。分母固定为原参考合格计划窗；缺失窗口未被悄悄剔除。

## 可检验的后续路线

1. 对已有最终候选却选错的窗口，优先审计候选竞争、谐波/周期和路径重获，冻结全局规则后再测试，不能把 oracle 选中的频率写入推理。
2. 对原 ROI 存在候选而融合波形缺失该候选的窗口，检查跨 ROI 的相消、谱峰竞争和融合前信号质量；实验必须保留原波形作为对照。
3. 对固定波形集中也找不到相容峰的窗口，单纯改最终 HR 读出无法保证达到±5 bpm，需要新的像素/照明/运动补偿或外部同步采集证据。不存在峰只是本定义下未检出，不能直接断言视频没有脉动。
4. 先补共享相机/Polar 同步标记和未用于本轮选择的新录像。候选覆盖上限不是独立人群准确率。

## 限制

- All oracle statistics consult the reference. They are evaluation-only ceilings for fixed candidate sets, not predictions or achieved accuracy.
- An oracle across many correlated channels can pick coincidental noise. Requiring two physical ROIs or >=20% peak power is a sensitivity view, not proof of physiological origin.
- Absence under this fixed spectral definition does not prove the original video lacks physiological information; another extraction method, genuine fundamental/harmonic structure, or imperfect synchronization may alter conclusions.
- RGB spectra can contain illumination/expression/motion energy; a reference-frequency RGB peak is not automatically PPG.
- Motion-frequency coincidence is not proof of artifact: real pulse and periodic movement can coincide.
- Raw color/motion diagnostics require a wholly finite window and do not interpolate missing samples; missing spectra are counted separately.
- The main original file-mtime alignment is retained; no offset is fitted. Unknown Polar notification delay and lack of shared clock markers limit all conclusions.
- Six previously inspected clips, overlapping windows and repeated participants are not held-out validation. No significance or population confidence interval is inferred.
- The generous supported-state oracle allows a <=3 bpm neighborhood around actual peaks, with original energy support; it must not be mislabeled as exact peak-center recovery.
