# 实时 rPPG：相机 → 波形 → 心率

默认 `fast` 模式使用已收到的帧，约积累 10 秒后尝试输出，随后每秒更新一个 10 秒窗口。信号质量不足时会保留缺失。原 V28/V32 离线入口 `run_retained_hr.sh` 保留。

## 安装和启动

在 Ubuntu / WSL 的仓库根目录执行（已验证 Python 3.10）：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
bash run_realtime.sh
```

在电脑 Chrome / Edge 打开 <http://127.0.0.1:8765>，点击“开启摄像头”。服务仅监听本机环回地址，默认不保存相机画面。Windows 摄像头由浏览器采集，不需要先转接进 WSL。项目根目录的 `启动实时PPG.bat` 是原作者本机 WSL 路径的便捷入口；其他安装位置请使用上面的命令。

建议实际输入稳定在 20–30 fps、面部清晰且额头及两侧脸颊可见。浏览器默认最大宽度 960 像素；内部 30 Hz 重采样不会增加真实采样信息。默认 JPEG 98 并复用连接；可选择 RGBA 画布颜色无损传输，但其数据量较大，在 Windows 到 WSL 的测试中反而降低送达帧率。

采集持续运行，只保留一个待发送的最新画面。上一请求完成后立即发送最新帧；缺失时间不会被压缩成连续采样。图表显示相对幅值的 rPPG/BVP，不是接触式 PPG 的绝对幅值，也不证明波形形态准确。

## 模式

| 模式 | 启动时间下限 | 用途 |
|---|---:|---|
| `fast`（默认） | 约 10 s | 原实时 R1 的数值等价计算加速 |
| `guarded` | 约 12 s | 固定规则的保守融合实验，仍有部分视频退步 |
| `motion` | 约 14 s 才可能有效输出 | 恢复旧版运动组合，覆盖率可能下降 |

```bash
bash run_realtime.sh --mode guarded --port 8766
bash run_realtime.sh --mode motion --port 8767
bash run_realtime_v1.sh --port 8768  # 原 R1 对照
```

后两种模式的测量窗口末端至少比发布所需输入早 2 秒，断流时可能更久。它们不是离线 V28/V32 的等价实现。模式不会根据视频编号或参考心率自动选择。

## 输出

每次会话保存到 `results/realtime/<时间_编号>/`：

| 文件 | 内容 |
|---|---|
| `frame_trace.csv` | 帧时间戳、皮肤 RGB、跟踪和运动诊断 |
| `heart_rate.csv` | 窗口心率、有效性、状态、计算耗时 |
| `ppg_windows.csv` | 各窗口波形和观测 / 插值 / 有效掩码 |
| `window_*.json` | 与该次心率相匹配的完整实测窗口波形及候选证据 |
| `session.json` | 配置与运行摘要 |

CSV 空白、JSON `null` 表示缺失，不是零。每次窗口发布后不修改，但相邻窗口有重叠，不能将其简单拼成一条不变的连续波形。心率概括 10 秒时间窗，不是拍摄时刻的瞬时心率。停止后图表显示历史结果。

## 文件与摄像头流

```bash
.venv/bin/python realtime_rppg_v2_20260918/run_source.py \
  --source /完整路径/视频.mp4 --realtime --out results/realtime/new_replay

.venv/bin/python realtime_rppg_v2_20260918/run_source.py \
  --source rtsp://摄像头地址 --out results/realtime/new_stream
```

输出目录必须尚不存在。文件回放使用声明帧率的 CFR 时间轴；`camera:0` 只适用于 Python 所在系统能直接访问的摄像头。流媒体时间戳表示读帧到达时间，设备及传输延迟仍需现场验证。

## 自检

从仓库根目录运行，不需要原始研究视频：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m unittest discover -s realtime_rppg_v2_20260918 -p 'test_*.py'
node realtime_rppg_v2_20260918/test_browser_lifecycle.js
```

Node 只用于浏览器脚本模拟测试，不是服务运行依赖。原 R1 的测试也可用 `unittest discover -s realtime_rppg_20260918` 执行。

可通过 `RPPG_TEST_CASEMAP=/路径/inference_case_map.json` 额外启用本地四视频缓存等价检查；默认只使用合成信号。该私有映射及原始数据不随仓库分发。

## 结果与限制

在固定 14 组缓存输入上，`fast` 的 690 个事件与 R1 心率、状态、窗口时间和波形掩码相同，波形最大差为 0。同一进程 42 对窗口计时的中位耗时为 95.46 → 35.97 ms。它是计算加速，**没有提高该批输入上的心率精度**。

新融合方案在部分视频改善、其他视频退步，因此保留为可选实验。14 组资料均已用于开发，参考时钟沿用历史估计；本次未完成实体摄像头与同步传感器的精度验收。

详见[发布与统一对照](../docs/realtime_release_20260918.md)和[汇总证据](validation/summary.json)。本地生成的摄像头结果默认由 `.gitignore` 排除。
