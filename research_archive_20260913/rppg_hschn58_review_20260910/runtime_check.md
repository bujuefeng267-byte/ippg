# hschn58/rPPG：入口、输出与验证范围核查

核查提交：`8dd44360d159eccde7f85a5a6542e3b44c621029`。本次阅读当前 `Full_Stack`、README、依赖、CI 与测试，并仅用 Python AST 解析文本；未导入或执行上游程序，未安装依赖，也未对用户视频进行试验。以下结论限定于该提交的当前集成代码。

**结论：它提供了可借鉴的实验流程，但现状需要整理才能复现；README 的“完整、统一、连续输出”描述强于集成代码实际提供的接口。仓库测试不能证明心率精度或覆盖率。**

## 入口与直接阻塞点

| 核查点 | 源码证据 | 影响 |
|---|---|---|
| 语法错误 | `Full_Stack/motion_process/occulsion_processing/YuNet_and_Template_match/motion_process_fb.py:53`、`:70` 的字符串定界符为 U+201C/U+201D 弯引号 | AST 在第 53 行报 `SyntaxError: invalid character U+201C`；原文件无法被 Python 正常解析，运动处理环节直接阻塞 |
| 主入口写死路径 | `Full_Stack/total_driver.py:249`—`:254` 写死 `<USER_HOME>/Documents/Research/My_work_parts/...`；`:137`、`:165`、`:272` 写死数据目录 | 不是在克隆目录直接给视频路径即可运行的统一 CLI；还需修正目录、大小写等 |
| 旧视频入口未完成参数化 | `Full_Stack/total_driver_from_prior_recording.py:224`—`:231` 同样是占位路径，并引用当前仓库没有的 `forward_backward_pixel_matching.py`；实际现存文件是 `_regions.py` 和 `_wholeface.py` | 名称虽然表示分析已有录像，但当前入口需要人工改脚本；没有解析原始视频路径的通用参数 |
| 必需依赖未完整声明 | 两个 driver 的 `:3`、`:5` 导入 `pyppeteer`、`PyPDF2`；`:76` 启动 Streamlit；`motion_process_fb.py:4`、`:5` 导入 pandas、Streamlit。`requirements.txt:10`、`:13`、`:14` 仅注释 pandas、PyPDF2、pyppeteer，未列 Streamlit | 单纯安装 requirements 不足以运行这些入口。`Sensor_formation/dynamic_roi.py:21` 另需 MediaPipe，而 requirements 的 MediaPipe 同样只是注释 |
| 采集部分依赖特定系统与外部资产 | `VID_TO_DATA_whole_face.py:10`—`:14` 直接导入 objc、AVFoundation、Cocoa；`:94` 的 YuNet ONNX 模型为作者本地路径，模型不在本次提交的仓库树中 | 采集入口针对 macOS；当前 Linux/Windows 环境需要另接视频读取和模型路径。README 的平台徽标不是整个采集链跨平台运行的证明 |

AST 核查：16 个 `.py` 文件中 15 个通过，1 个语法错误；6 个 notebook 的 47 个代码单元均可解析，无 IPython magic 跳过。通过语法解析仅证明语法可读，不代表变量、依赖、路径、数值行为或完整流程已经正确。审计脚本及逐项结果为同目录 `static_syntax_check.py`、`static_syntax_check.json`。

## 实际输出与指标

| 环节 | 已实现内容 | 不能等同于 |
|---|---|---|
| 采集与光流 | 帧/区域数据、像素轨迹 `.npz`，可视化；`motion_process_fb.py:223`—`:224` 保存区域 RGB `.npy` | 最终 PPG 波形与心率的统一评估表 |
| CHROM/POS 主脚本 | 对每个 ROI 的整段信号估计一个 BPM 和一个内部频谱分数，绘制波形，输出 PDF 路径供 driver 合并 | 连续逐窗心率 CSV、逐窗拒绝原因、PPG/HR 覆盖率 |
| CSV | `VID_TO_DATA_whole_face.py:423`—`:448` 写出帧序号、整帧 RGB 均值、R/G 与 B/G 比值 | 同步生理参考 CSV；这里名为 AWB gains 的值也不是直接读取的相机硬件白平衡寄存器 |
| PDF 集成 | `total_driver.py:300`—`:306` 合并运动图、CHROM 两分支、POS 图 | 在同一时轴输出多方法的数值评估结果 |

具体信号证据：

- `CHROM_method_npy_input.py:94`—`:95` 固定两端各裁 25 帧、采样率 30；`:111`—`:116` 每个 ROI 调用一次估计；`:154`—`:162` 保存 PDF。
- `CHROM_method_npz_input.py:100`—`:102` 同样固定裁帧与 30 fps；`:128`—`:132` 每区域得到一个 BPM；`:152`—`:160` 保存 PDF。
- `POS_method.py:81`—`:83` 同样固定参数；`:101`—`:107` 每区域调用一次；`:126`—`:134` 保存 PDF。虽然内部 `pos_trace` 有 1.6 秒重叠重建窗，`:65`—`:72` 仍对整个重建结果做一次 Welch 峰值估计，不能因此称为连续心率时序输出。

**三处名为 SNR 的值口径不同，不能直接按图例数字比较：**

1. CHROM `.npy` 分支 `:63`—`:68`：估计峰及二次谐波 ±0.1 Hz 内功率 / 其余带内功率，转 dB。
2. CHROM `.npz` 分支 `:62`—`:63`：估计峰 ±0.2 Hz 内功率 / 其余带内功率，线性比值。
3. POS 分支 `:71`：估计峰 ±0.2 Hz 内功率 / 总带内功率，线性比例。

这些分数均围绕程序自己选出的谱峰计算，没有引入同步真实心率；因此高分不能证明该峰来自脉搏。与用户当前项目比较时，必须统一分数定义、FPS、频带、评估时间窗、参考值和覆盖率分母。

## 测试与验证证据的边界

- `tests/test_smoke.py:1`—`:6` 只测试 NumPy/SciPy/Matplotlib/PyWavelets/OpenCV 导入，以及 `1+1==2`。
- `tests/test_signal_utils.py:3`—`:5` 的 `bandpass` 是原样返回输入的占位函数；`:7`—`:12` 只确认数组长度不变，没有调用 Full_Stack 的真实滤波器。
- `.github/workflows/ci.yml:22`—`:35` 安装 requirements 后运行这些 pytest；无测试被收集时还按成功处理。没有编译/导入全部生产模块，也没有端到端视频测试，因此 CI 通过不能排除上述语法错误。
- 当前集成代码中未发现同步 ECG/接触式 PPG 参考读取与对齐、真实 HR 的 MAE/RMSE 计算、有效波形或 HR 覆盖率统计、受试者划分的验证流程。采集代码的 `VID_TO_DATA_whole_face.py:360`、`:478` 让操作者手动输入开始和结束心率，`:335`—`:339` 写在预览帧上；这与全程同步参考验证不同。
- README `:63` 的“higher precision”及 `:65` 的约 150 BPM 运动处理伪影描述属于作者总结；当前提交未提供足以复算“比用户项目更准”的同口径评估结果。论文能支持什么应由论文核查单独判断。

本次不宜给出该仓库的精度提升百分比。若后续开展试验，先完成可运行性适配，然后保留相同视频、时间轴和参考数据，分别记录实际波形覆盖率、HR 输出覆盖率及有参考数据区间的误差，才能判断是否改进。
