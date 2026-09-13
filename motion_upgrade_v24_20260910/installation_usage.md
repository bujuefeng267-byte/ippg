# V24 独立实验安装与 fresh 复核

本文描述已准备的流程，不代表已经安装或已经通过验证。安装及复核状态以 `deployment.json`、`deployment_verification.json` 为准。性能结论以冻结评测为准，和安装是否正常分开报告。

安装只新增三个位置：

- `/home/fengbujue/项目/rppg识别/motion_upgrade_v24_20260910`
- `/home/fengbujue/项目/rppg识别/results/motion_v24_20260910`
- `/home/fengbujue/项目/rppg识别/run_motion_v24_experimental.sh`

不修改旧入口，没有默认晋级选项。源视频和参考文件不复制、不修改；只复制 11 个核心模块、测试、文档、研究记录和已生成的验证/评测/审查材料。目标路径或本地安装记录已经存在时拒绝再次安装；失败后的 staging 目录保留供检查，不自动删除或覆盖。

## 准备与安装命令

在 WSL 中使用原项目现有 Python，无需安装新依赖：

```bash
PROJECT='/home/fengbujue/项目/rppg识别'
STAGE='/mnt/c/Users/15011/Documents/ChatGPT/New project/rppg_motion_v24'

# 默认只读计划：不执行推理，不写入原项目。
"$PROJECT/.venv/bin/python" -B "$STAGE/deploy_v24.py"
"$PROJECT/.venv/bin/python" -B "$STAGE/verify_install_v24.py"

# 仅内存比较器自检，不读取或运行视频。
"$PROJECT/.venv/bin/python" -B "$STAGE/verify_install_v24.py" --self-check

# 在冻结推理、24/24 运行、测试、波形回算及评测完成后显式执行。
"$PROJECT/.venv/bin/python" -B "$STAGE/deploy_v24.py" --install

# 安装后只读 SHA256 核验，不执行视频。
"$PROJECT/.venv/bin/python" -B "$STAGE/deploy_v24.py" --verify
```

安装前核验冻结源码、测试集合、完整运行身份、评测协议、波形回算及评测表哈希；复制后逐文件检查 SHA256，并核对旧项目根目录的 Python/入口脚本与 README。安装会保留完整性能 gate，无论其结论如何都只建立独立实验入口，不自动声称升级成功。

当前唯一安装候选为 `guarded_fusion_gap10`，入口显式包含：

```text
--pixel-mode tracking_screened --algorithm-mode guarded_fusion --max-gap 0.10
```

若最终选择需要改变，须在安装及 fresh 运行前明确修改、重新冻结脚本和比较合同；不得依据 fresh 误差临时换候选。手动改变 CLI 参数的运行不属于这里的固定比较。

## 安装后的真实视频运行

```bash
cd /home/fengbujue/项目/rppg识别
./run_motion_v24_experimental.sh \
  videos/user_0907/Video_20260907_171742478.avi \
  --output results/v24_user0907_fresh_example
```

输出目录必须是新路径。该命令不带缓存，会重新执行检测、像素跟踪、局部筛选、失败回退、颜色重建、波形融合和 HR 读出。最终 HR 来自实际保存的波形；离线滤波、重叠拼接和 DP 会使用未来样本，不能称为已经验证的实时系统。

正式安装复核使用完整 UBFC subject1 和 synthetic72 两段原视频：

```bash
"$PROJECT/.venv/bin/python" -B "$STAGE/verify_install_v24.py" --run
```

复核调用原项目中已安装的 `.sh` 入口，比较参照来自已安装的冻结 validation 结果；不传 RGB、几何或 tracking cache，不传参考值，不截短视频，也不会在失败后改成缓存重放。输出在 `results/motion_v24_20260910/installed_fresh/`。开始任何视频前，先写入不可覆盖的 `comparison_contract.json`，绑定脚本、核心源码、原视频和比较输入哈希。

## 比较合同

| 项目 | 规则 |
|---|---|
| 实际波形、HR、proposal、分支路由、source/time、采样标记、baseline RGB、新 RGB | 全部 16 个 CSV 的行数及列顺序相同；离散值和缺失位置完全相同；有限浮点数绝对误差 ≤ `1e-9`，相对容差为 0。 |
| 波形文件及各分支输出 | 包括 POS/CHROM/fusion、逐 ROI、baseline/tracked 两分支及路由；CSV 清单多文件或少文件都不通过。 |
| 跟踪 FB 诊断 | 三个 `*_pixel_fb_error_median` 的有限数值差异单独报告；缺失位置仍属于核心合同。三个 `*_pixel_fb_rejected` 离散计数仍须完全一致。 |
| FB 完整报告 | 六个 FB 字段逐帧保存预期值、实际值、差值、是否符合合同；包含一致的值，不只列前几个差异。 |
| Summary | 计算配置、身份、源码、统计和覆盖结果须一致。耗时、提取路径、OpenCV 线程数单列；缓存及输出路径按预声明区分，同时要求 fresh 路径确实不使用缓存。 |
| 原始数据与冻结文件 | 运行前后检查原视频、比较输入、已安装文件和旧入口哈希。 |

报告分别给出四个字段：

- `runtime_integrity_passed`：两次入口执行成功，文件/输入完整性核对通过。
- `core_outputs_match`：核心输出和上述全部必要检查通过。
- `diagnostic_values_match`：跟踪 FB 诊断值也在原数值容差内一致。
- `strict_all_fields_match`：核心输出及诊断值同时满足合同。

若核心一致但 FB 诊断不同，允许出现 `core_outputs_match=true`、`strict_all_fields_match=false`。这时应如实称“核心输出一致，诊断存在差异”，不能称“全部字段严格通过”。程序退出码依据运行完整性和核心输出；不能只看退出码就推断严格全字段一致。性能 gate 原样另存，安装复核不替代生理误差评测。

## 程序主要输出

- `frame_trace.csv`：逐帧 baseline RGB、新 RGB、像素来源和重置/回退诊断。
- `*_waveform.csv`、`roi_waveforms.csv`、`baseline_roi_waveforms.csv`：实际波形和可用/插值信息，缺失不解释为零脉搏。
- `*_heart_rate.csv`：局部读出、离线 DP、接受标记与拒绝原因；必须结合 `accepted` 读取。POS/CHROM 的历史诊断峰列可保留未接受候选，不能计入有效输出。
- `branch_routing.csv`、`*_branch_proposals.csv`、`fusion_proposals.csv`、`fusion_diagnostics.csv`：新旧分支选择与实际融合贡献。
- `summary.json`：参数、输入身份、源码哈希及覆盖统计。

两段 fresh 运行只验证安装与输出重现，不构成新增独立受试者或泛化验证。无同步参考的视频仍不能报告绝对心率准确率。
