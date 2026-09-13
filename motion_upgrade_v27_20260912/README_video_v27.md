# V27 新视频实验入口

`analyze_video_v27.py` 不依赖六段开发视频的位置、编号或参考标签。默认模式为 `late_psd_cluster`，仅代表明确的实验预设；它不是自动选择的最佳模型，也不表示通过升级门槛。

在具有项目依赖的 Python 环境中运行：

```bash
python analyze_video_v27.py /path/to/video.mp4 --output /path/to/new_output
python analyze_video_v27.py /path/to/video.mp4 --output /path/to/another_output --variant all
python analyze_video_v27.py /path/to/video.mp4 --output /path/to/new_output --frontend-cache /path/to/previous_output/frontend
```

输出目录必须不存在。视频至少需要 10 秒，固定使用 10 秒窗口、1 秒步长和 42–210 bpm 范围。入口不接收真实心率或同步偏移参数。

`--variant` 支持以下八种固定模式，也可选择 `all`：

| 模式 | 主心率来源 |
| --- | --- |
| `early_median` | 先在三个区域内合并小 patch 的 RGB，再做支路 BVP 与区域中位数统计 |
| `early_psd_cluster` | 提前合并 RGB，完整 PSD 聚类后做区域心率统计 |
| `late_median` | 保留 12 个独立 patch 的 BVP，再做区域中位数统计 |
| `late_psd_cluster` | 保留独立 BVP，完整 PSD 聚类后做区域心率统计 |
| `fusion_nomotion_local` | 相同 late-PSD 融合波形，无运动证据、逐窗选频 |
| `fusion_motion_local` | 相同融合波形，加入运动证据、逐窗选频 |
| `fusion_nomotion_dp` | 相同融合波形，无运动证据、DP 轨迹选择 |
| `fusion_motion_dp` | 相同融合波形，运动证据与 DP 轨迹选择 |

四种 `fusion_*` 模式需要 `late_psd_cluster` 的真实融合波形，入口会保存这份必要的母输出，并在总清单的 `support_variants` 中标明。不会按参考值选择模式。

## 输出文件

- `frontend/`：独立 patch 原始与跟踪 RGB、全局帧时轴、固定局部锚点、完整关键点和原始元数据。
- `signals/anchored_patch_trace.csv`：各 patch 独立低频锚定后的 RGB。
- `signals/*_patch_waveforms.csv`：实际保存的 POS/CHROM 支路波形、采样支持和来源标记。
- `variants/<mode>/waveform.csv`：由实际支路样本融合得到的波形。缺口为 NaN。
- `variants/<mode>/primary_hr.csv`：该模式的主心率。
- `variants/<mode>/secondary_hr.csv`：重新读取保存的融合波形，再用原 V25 运动证据与 DP 计算的心率。
- `variants/<mode>/ppg_hr.png`：三栏分别显示融合波形、主心率、次心率，缺失不连线。
- `manifest.json`、各模式清单和 `frontend_binding.json`：源代码、输入、缓存与输出 SHA256，参数、来源和窗口数。

四种空间统计模式的主心率是多个 patch 的分层统计量，不能称为图中单条融合波形的心率。普通中位数还可能落在两个实际谱峰之间。次心率专门回答“保存的这条融合波形读出了什么心率”。相对波形幅度不是接触式 PPG 的标定形态；输出覆盖率也不是准确率。

## 缓存验证与测试范围

缓存必须来自当前前端配置和源码，绑定相同视频字节、全部四个前端输出、固定 patch 身份及完整时轴。复用缓存时仍会完整解码一次视频核对实际帧数和 FPS；不将容器声明的帧数当作解码真值。截断缓存、损坏哈希、缺行、错误采样掩码或已有输出目录会被拒绝。

合成集成测试使用声明的前端测试替身，实际执行锚定、支路波形、聚合、保存波形心率读出和 PNG 导出。它验证接口及计算合同，不证明真实视频上的精度；真实缓存烟测另行记录。
