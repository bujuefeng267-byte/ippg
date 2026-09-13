# V26 单视频实验入口

`analyze_components_v26.py` 可对新视频运行 `component` 或 `harmonic` 实验分支。它没有替换原推荐 V25 入口，也没有修改六视频实验的冻结算法或参数。现有六视频属于开发回归，不能把这些结果视为新受试者的独立泛化验证。

## 使用

在包含本文件及其同目录算法模块的目录中，使用原项目已有 `.venv`：

```bash
/home/fengbujue/项目/rppg识别/.venv/bin/python -B analyze_components_v26.py \
  --video '/完整路径/新视频.avi' \
  --out '/完整路径/新的实验结果目录' \
  --variant component
```

`--variant harmonic` 使用同目录 `component_harmonics_v26.py`，仅在选用时导入。它扩展分量候选阶段的几何运动频谱证据；最终心率仍调用同一 V25 `estimate_evidence`，与冻结六视频的 harmonic runner 一致。

要复用严格匹配的采样缓存，可增加：

```bash
  --trace-cache '/原结果目录/frame_trace.csv'
```

也可以传入包含 `frame_trace.csv` 和 `frame_trace.json` 的目录。输出路径必须尚不存在；程序拒绝覆盖已有结果，包括中断后留下的目录。再次运行请指定新的输出目录。

## 固定处理流程

1. 无缓存时使用原 V25 前端：单人脸几何、三块皮肤 ROI、10% 截尾均值、`tracking_screened` 配对像素跟踪与局部筛选，失败时保留明确标记的原 ROI 回退。
2. 保留 `baseline_*` 原始 ROI 均值；跟踪分支使用默认四阶 0.15 Hz 锚定重建。跟踪重建 RGB 不是新的原始测量值。
3. 保存并重读完整 `frame_trace.csv`，分别对 baseline/tracked 三 ROI 计算 POS 和 CHROM。它们采用相同采样/补点掩码，短缺口上限为 0.1 秒。
4. 使用默认 `ComponentConfig()`：至少两块物理 ROI 支持同一候选；同一区域的多种方法和分支不会重复算作多张票。从实际测得信号中滤出候选附近的分量，保持测得的复数频谱系数和相位关系。
5. 先保存 `waveform.csv`，再重读该文件运行原 `estimate_evidence`。固定 10 秒窗口、1 秒步长、42–210 bpm，不接受参考心率、目标频率或逐视频调参输入。

这条流程使用离线滤波、重叠融合和离线动态规划。输出属于**自适应带限的实测分量**；完整脉搏形态、幅度或血压意义没有得到验证。

## 输入和缺失约定

- 视频必须是可由当前 OpenCV 解码的本地文件。入口使用 `frame / fps` 完整时轴，假设恒定帧率；它没有解析可变帧率视频的逐帧 PTS。
- 固定上限 210 bpm 对应 3.5 Hz，因此此入口要求 `fps > 7`。这是比底层读取器 `fps >= 5` 更严格的频带条件。通常可继续使用项目现有的约 30 fps 视频；180 fps 视频保持原始帧率，不由该传统算法入口下采样。
- 没有新增分辨率或人脸像素数硬门槛。可解码并不等于人脸/皮肤/脉动信息足够；遮挡、模糊、姿态和颜色变化仍可能导致缺失或错误候选。
- 视频不足 10 秒时保留所有原始帧，心率与分量候选表为空；没有完整合格窗口时不伪造估计。
- 波形缺失在 CSV 中为空值（读入后为 NaN），`covered=false`。拒绝窗口的主 `ridge_bpm` 和 `spectral_peak_bpm` 为空，不用 0 或上一心率填补。
- `observed`、`interpolated` 表示实际贡献样本的来源，不代表滤波完整感受范围。覆盖率表示可用性，不表示准确率。

## 缓存身份校验

缓存必须同时匹配当前视频的绝对路径、字节数、修改时间、完整视频 SHA256、前端源码 SHA256、像素参数、锚定参数、采样方法和整片处理设置。还校验 trace 文件 SHA256、行数、帧索引、单调时间轴、实际视频 fps、ROI 颜色有效掩码、质量范围和跟踪来源字段。

新的 `frame_trace.json` 自带 `video_sha256`。旧 V24/V25 trace 只有文件属性身份，必须保留附近原始 `inputs_manifest.json`：入口从最多四层祖先目录寻找该文件，只读取视频路径/字节数/SHA256，不打开任何参考数据文件。缺少可核验的视频哈希时拒绝复用，可以去掉 `--trace-cache` 从视频重新提取。视频移动或修改后，即便名称相近，也不会静默接受旧缓存。

旧 `roi_waveforms.csv` 没有被 trace 元数据逐文件绑定，因此入口不直接当作可信输入；它从经过校验的 trace 重新计算两分支的 ROI 波形。

## 输出文件

| 文件 | 内容 |
|---|---|
| `frame_trace.csv / .json` | 全帧几何、baseline/tracked RGB、采样/跟踪来源及完整身份 |
| `baseline_roi_waveforms.csv` | 原始 ROI 均值生成的三 ROI POS/CHROM 与采样掩码 |
| `roi_waveforms.csv` | 锚定跟踪 RGB 生成的三 ROI POS/CHROM 与相同采样掩码 |
| `waveform.csv` | `time_s, base, covered, observed, interpolated`；`base` 为相对归一化分量 |
| `heart_rate.csv` | 从刚保存波形读出的局部/DP 心率、接受状态与证据诊断 |
| `component_proposals.csv` | 分量候选、实际贡献物理 ROI/通道、是否生成分量 |
| `all_roi_candidates.json` | 各窗口各 ROI/分支/方法的全部候选及证据 |
| `summary.json` | 帧数、计划/接受窗口数、覆盖率、处理路径 |
| `manifest.json` | 固定参数、源码/视频/缓存/输出哈希；全部完成后 `status=complete` |

`component_proposal_bpm` 是滤出分量时使用的候选；最终心率来自保存波形的独立读出，二者不可混为同一个指标。接受状态和质量分也不是准确概率。

## 已完成验证及精度边界

新增 `test_components_cli_v26.py` 共 8 项测试通过，含两种分支的已知 84 bpm 合成颜色脉动、真实短无脸视频的完整时轴/空输出、缓存复用、错误视频/参数/时间轴拒绝、禁止参考输入以及拒绝覆盖。收据为同目录 `components_cli_test_receipt.json`。

只对 data2 做一次缓存回归，没有重跑六组人脸提取或读取参考心率。输出在项目：

`results/data1_6_v26_20260911/entry_smoke_data2/`

其 `entry_qa_receipt.json` 给出完整逐列差异：

- 原始 trace 字节完全相同；重算两分支 ROI 波形与原保存值的最大差约 `1.78e-13`，采样掩码相同。
- 最终局部/DP 心率、接受掩码精确相同：37 个计划窗口、31 个接受窗口。
- 第 3、4 号分量窗口候选从原 74/75 bpm 变为 75/76 bpm。独立依据两套保存候选回算，两个路径在各自输入上的整段 DP 目标分数差均为 **0**，属于同分路径的浮点选取差异。
- 候选相差 1 bpm 会移动带限中心，因此波形不能声称相同：最大绝对差 `0.08531415`，相对 RMSE `2.625%`。物理 ROI 通道选择、生成掩码和最终心率未改变。

这项回归通过的是身份、掩码、ROI 重算容差和最终心率一致性，并没有隐藏波形/候选非唯一性，也没有证明新视频准确率提高。冻结算法没有为消除这个差异而改动。
