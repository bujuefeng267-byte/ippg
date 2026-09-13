# V2.3 像素跟踪实验入口与安装核验

本文说明目标安装后的使用方法。是否已安装、是否通过默认晋级，以部署清单与实际核验记录为准；准备阶段不预填通过结论。

## 从原视频运行

进入 WSL 项目目录，指定一个尚不存在的输出目录：

```bash
cd /home/fengbujue/项目/rppg识别
./run_motion_v23_experimental.sh \
  videos/user_0907/Video_20260907_171742478.avi \
  --output results/v23_user0907_fresh_20260910
```

实验入口明确启用 `tracking_screened`，短缺帧上限默认 0.10 秒，HR 窗口 10 秒、步长 1 秒。默认从原视频执行人脸检测、像素跟踪和后续处理，不需要缓存。输出目录已存在时会报错；使用新名字保留每次证据。

显式 `--max-gap 0.15` 是另一项固定候选，须按单列评价解释，不能把覆盖变化直接说成准确度提高。`--help` 可查看完整选项。未经另行验证，不应把改变参数后的结果套用本次默认候选的评价结论。

原入口 `run.sh`、`run_motion_v2.sh`、`run_motion_v21.sh`、`run_motion_v22.sh` 保留。`run_motion.sh` 仅在已固定的性能门槛和实际安装核验均通过后才会指向 V2.3；仅安装实验入口不代表已替换默认算法。

## 输出含义

- `frame_trace.csv`：每个原视频帧一行，包含人脸／几何来源、RGB 有效性和各 ROI 的像素跟踪来源。新颜色轨迹包含相对颜色变化重建，并非逐帧原始 RGB；跟踪失败后的基线回退会显式标记。
- `pos_waveform.csv`、`chrom_waveform.csv`、`fusion_waveform.csv`：保存实际用于估计的波形及观测／插值信息。缺失保留，不自动解释为零信号。
- `*_heart_rate.csv`：两种统一读出为局部频谱峰和离线 DP；同时保存接受状态与拒绝原因。POS/CHROM 中历史诊断峰值字段可能保留未接受窗口的候选值，使用结果必须同时检查 `accepted`；fusion 最终拒判 HR 为空。
- `fusion_proposals.csv`、`fusion_diagnostics.csv`：上游候选共识、可用通道、实际融合贡献。候选窗未自行生成波形时，邻窗仍可能覆盖，因此不能把上游提议状态直接当最终 HR 状态。
- `summary.json`：源码哈希、参数、时间轴、输入来源和各分支覆盖率；`comparison.png` 为检查图。

整个管线为离线处理：滤波和重叠融合可利用后续样本，DP 使用未来窗口。输出覆盖率与波形有限点比例不等于生理准确率；无同步参考的视频不能计算绝对心率误差。

## 安装后的独立回放核验

工作区脚本 `verify_install_v23.py` 不带参数只显示计划。安装成功后，由主代理在 WSL 运行：

```bash
/home/fengbujue/项目/rppg识别/.venv/bin/python -B \
  '/mnt/c/Users/15011/Documents/ChatGPT/New project/rppg_motion_v23/verify_install_v23.py' \
  --run
```

它从 UBFC 和 synthetic72 的完整原视频调用已安装实验入口，**不传 `--cache` 或 `--geometry-cache`**，结果分别写入：

```text
results/motion_v23_20260910/installed_smoke/ubfc/
results/motion_v23_20260910/installed_smoke/synthetic72/
```

核验比较冻结 `tracking_screened_gap10` 的全部 10 种计算 CSV、输入身份、源码与关键摘要。接受状态、来源、计数和缺失掩码严格比较；有限浮点采用固定 `atol=1e-9, rtol=0`，逐列保留最大差和首批不一致行。图像编码和运行耗时不作为计算结果一致性指标。

冻结验证用的是原像素加已保存的 V2.2 几何轨迹，而此次重新执行检测。若新检测不能逐点复现，脚本保留差异、返回失败核验，并将 `matches_validation=false` 写入记录；不会自动放宽容差，也不会悄悄改用几何缓存。确需追加缓存回放时，必须另行命名、单独解释，不能替代这次从原视频运行的结果。

正式记录为工作区 `deployment_verification.json`，另在 `installed_smoke/verification.json` 保存副本；每个案例有日志和详细比较 JSON。核验前后检查原视频、冻结验证输入、已安装文件和旧入口哈希。程序执行成功与数值对照成功分开记录；安装核验通过也不代表新的生理泛化验证通过。
