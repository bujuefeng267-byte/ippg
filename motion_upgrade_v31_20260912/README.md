# V31 实验代码

未通过精度提升条件，推荐入口仍为原 V28。

[阅读六视频结果与结构限制](<//wsl.localhost/Ubuntu/home/fengbujue/项目/rppg识别/V31保护机制与六视频结果_20260912.md>)

对已经完成的 V28 输出进行独立候选实验：

```bash
./run_motion_v31_experimental.sh --v28-output results/原V28输出 --out results/新V31实验 --mode direct_motion
```

V28 输出应包含 frame_trace.csv 和同名 JSON；旧批量输出可额外传 --trace-cache 指向其原缓存。这是实验入口，不能据本次结果声称提高精度。
