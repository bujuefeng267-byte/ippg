# 重跑 cPACE 与 RhythmMamba

入口为当前工作区根目录的 `replay_paper_methods.py`。它使用已冻结的作者代码、项目适配和唯一权重创建新快照，再执行自检与六段视频推理；不会改变 `rppg_paper_trials_20260909` 中的本轮结果，也不会删除已有重跑目录。

在WSL中使用现有科学计算Python运行。例如只准备两种方法的源码与权重快照：

```bash
"/home/fengbujue/项目/rppg识别/.venv/bin/python" -B "/mnt/c/Users/15011/Documents/ChatGPT/New project/replay_paper_methods.py" --method both --name prepare_check_01 --prepare-only
```

这只新建 `C:/Users/15011/Documents/ChatGPT/New project/rppg_paper_replay_prepare_check_01`，核对复制文件SHA256及本轮冻结源码/权重记录，不导入模型、不运行测试或推理。查看新目录的`snapshot_manifest.json`，其中`files`为相对路径到SHA256映射。

正式重跑必须另用未存在的名字：

```bash
"/home/fengbujue/项目/rppg识别/.venv/bin/python" -B "/mnt/c/Users/15011/Documents/ChatGPT/New project/replay_paper_methods.py" --method cpace --name cpace_repeat_01
"/home/fengbujue/项目/rppg识别/.venv/bin/python" -B "/mnt/c/Users/15011/Documents/ChatGPT/New project/replay_paper_methods.py" --method rhythm --name rhythm_repeat_01
"/home/fengbujue/项目/rppg识别/.venv/bin/python" -B "/mnt/c/Users/15011/Documents/ChatGPT/New project/replay_paper_methods.py" --method both --name both_repeat_01
```

`--name`只接受ASCII英文字母、数字和下划线。已有目标目录（包括准备模式建立的目录）会直接拒绝，不会覆盖、删除或自动续跑。失败时保留新目录及日志供检查，重新尝试需新名字。

## 环境与执行次序

| 方法 | 既有WSL解释器 | 次序 |
|---|---|---|
| cPACE | `/home/fengbujue/项目/rppg识别/.venv/bin/python` | 合成自检 → 六段视频两分支 → 保存波形/HR一致性检查 |
| RhythmMamba | 预处理检查用上述科学环境；后端与推理用`.venv-rhythm-trial/bin/python` | 无Torch预处理检查 → 后端/固定权重自检 → 六段视频，显式CUDA |

`both`依次完成cPACE，再执行Rhythm。入口使用`subprocess.run(argv列表, shell=False)`，不经命令shell调用，不安装或更换任何依赖。Rhythm使用本轮已建立的独立`.venv-rhythm-trial`，不自动换权重、改为CPU或切换到另一种扫描实现。原始代码和测试未被这个入口修改。

## 快照与数据依赖

白名单快照保留cPACE作者全部顶层`.py`、README与MIT许可证，固定项目读出文件，以及所选方法的适配/测试源。Rhythm另包含固定官方模型、配置、唯一PURE权重、许可证和源审计JSON。只选Rhythm时也带上`cpace/project_estimator`，因为其冻结runner引用这个相对位置。

Rhythm内置Mamba的两份源正文另保留Apache-2.0许可证及Tri Dao、Albert Gu署名，位于`rhythm/third_party_licenses/Mamba_LICENSE`；同目录`Mamba_LICENSE_provenance.json`记录从固定Rhythm提交`1533ad21cb4ae0d31e9dbb9afb0b1eefbed54aed`的`setup/mamba/LICENSE`直接下载的来源、SHA256与所覆盖源码。重跑快照会复制这两份文件，原Rhythm仓库的MIT许可证也继续保留。此前已建立的准备快照不会被自动补写。

快照不含`.git`、`__pycache__`、旧`results/evaluation`、旧自检通过JSON、旧`run_summary.json`或旧推理协议。每次运行重新自检，并由新副本生成自己的协议和完成标记。源码快照根目录始终是当前工作区的直接子目录，因此冻结runner的`ROOT`解析仍指向原工作区。

原始六段视频和`rppg_motion_v22/validation/trimmed_gap10`缓存必须仍在原位置；本入口不复制视频或重新提取V2.2跟踪缓存。cPACE从缓存RGB运行，Rhythm按缓存人脸框读取真实原片像素。所有方法仍不接受参考HR。本轮输入、窗口、gap和离线边界保持不变，重跑不等于独立新数据验证。

新目录内：

- `snapshot_manifest.json`：所复制文件、SHA256、命令及原来源。
- `replay_logs/`：每个自检/推理步骤的独立日志。
- `replay_execution.json`：正式运行的步骤返回码、耗时和是否全部成功；准备模式不写这个文件。
- `cpace/results/`及`cpace/results_no_homodyne/`：两种固定cPACE结果。
- `rhythm/results/`：Rhythm实际网络输出、后处理波形和心率。

这个入口只重跑推理，没有复制或运行参考评分器。比较误差时需要另按既有冻结评价协议开展独立评分，不能用新重跑结果选择更有利的参考偏移。

## 可选WSL启动脚本

如需放到原项目下的新入口，可使用下列内容，例如另存为`run_paper_replay.sh`；本交付仅提供内容，不安装或改动旧入口：

```bash
#!/usr/bin/env bash
set -euo pipefail
exec "/home/fengbujue/项目/rppg识别/.venv/bin/python" -B "/mnt/c/Users/15011/Documents/ChatGPT/New project/replay_paper_methods.py" "$@"
```

例如运行`bash run_paper_replay.sh --method both --name next_run_01 --prepare-only`。命令中的整条带空格脚本路径需保留引号。
