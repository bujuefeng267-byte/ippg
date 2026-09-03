# rPPG 识别基线

本目录用于验证：普通摄像头人脸视频可以通过现有开源 rPPG 方法得到估计波形和心率，并观察运动时的失效情况。

当前默认采用 POS，另提供 CHROM。算法路线参考开源项目 [pyVHR](https://github.com/phuselab/pyVHR)；人脸跟踪使用 MediaPipe。这里输出的是摄像头估计的 rPPG/BVP，不是接触式传感器测得的真实 PPG。

## 目录

- `videos/`：放入原始 MP4 视频
- `results/`：每段视频的 CSV、图像和摘要
- `.venv/`：隔离的 Python 环境

## 运行

```bash
cd /home/fengbujue/项目/rppg识别
./run.sh videos/static.mp4 pos
./run.sh videos/static.mp4 chrom
```

每次运行产生：

- `rppg_waveform.csv`：逐帧 RGB 和归一化 rPPG 波形
- `heart_rate.csv`：滑动窗口心率与频谱信噪比
- `summary.json`：帧率、时长、人脸检出率、心率统计
- `rppg_report.png`：波形和心率图

## 建议视频规格

- 1080p 或 720p，恒定 30 fps，MP4
- 正脸、稳定照明、固定相机、关闭美颜
- 每段 90–120 秒；所有速度保持相同机位和光照

## 自检

项目附带一段无隐私的合成测试视频，目标频率是 72 bpm：

```bash
.venv/bin/python make_self_test_video.py
./run.sh videos/self_test_72bpm.mp4 pos
```

真实实验仍需 Polar、指夹血氧仪或 ECG 同步记录，才能计算绝对心率误差；只有摄像头视频时可以验证可提取性、稳定性和运动失效趋势。
