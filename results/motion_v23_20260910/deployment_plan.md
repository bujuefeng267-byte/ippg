# V2.3 隔离部署准备

2026-09-10，只读盘点完成。尚未安装 V2.3、修改 WSL 文件或切换入口。部署执行须等待主代理形成独立评价和门槛判定。

## 现有项目与约束

- 项目：`/home/fengbujue/项目/rppg识别`；工作区：`C:/Users/15011/Documents/ChatGPT/New project`。
- 沿 WSL `/`、`/home`、`/home/fengbujue`、`/home/fengbujue/项目`、项目本身及其相关子目录，和 Windows 工作区及其祖先路径，未发现 `AGENTS.md`。扫描跳过 Git、虚拟环境和已排除的数据路径。
- 已阅读 `PROJECT_OPERATIONS.md`；它是旧操作记录。其历史门槛和性能数字不是本轮 V2.3 默认晋级判据。未修改 Git 配置、索引、工作树或既有结果；只读状态使用一次性的 `safe.directory` 和 `--no-optional-locks`。工作树原有修改和未跟踪文件需保留。
- `run.sh VIDEO [pos|chrom]` 仍运行根目录旧版 `analyze_rppg.py`，默认 POS。
- `run_motion_v22.sh VIDEO --output NEW_DIRECTORY` 是独立入口，运行 `motion_upgrade_v22_20260909/analyze_motion_v2.py`，默认短缺帧上限 0.10 秒；0.15 秒须显式传参。不能把这两种启动器的参数契约混为一谈。
- V2.2 部署记录中的 7 个推理模块、7 个测试文件、README 及启动器均逐文件 SHA-256 对账通过。旧安装核验记录包含 data1/gap10、0907/gap15，六种 CSV 与验证产物在 1e-9 内一致。

| 现有入口 | SHA-256 |
|---|---|
| run.sh | d9eee11d650bb1a8938d48f16d5bedbb551739f0bc74e0559252dfd37bbf36ad |
| run_motion_v2.sh | 5640dec0bf393573d7691172845794407fee3263af2067c0a84a7848a4e0ca00 |
| run_motion_v21.sh | d88437a67b30101ecf4839e60a8849e0f00bca7e0165e5be32f1e4a9e34d1442 |
| run_motion_v22.sh | 436f9acb7f02dcf966592af7602eac81f37b11cbc7258aa4ad1c5de29dd021b7 |
| run_paper_replay.sh | 7376f2edcfb5508fa14075e378f32cd8e4c3a4860f488789a854b470b7d5f86c |

## 数据完整性

以下六个来源均存在，完整文件哈希与 `rppg_motion_v22/validation/protocol_before_validation.json` 一致。没有复原、查找或读取已删除的数据。

| 来源 | 原视频位置 | 本轮参考边界 |
|---|---|---|
| user0904 | `/mnt/c/Users/15011/Desktop/Video_20260904_145402918.avi` | 无参考 |
| user0907 | 项目 `videos/user_0907/Video_20260907_171742478.avi` | 无参考 |
| UBFC | 项目 `videos/ubfc_subject1/vid.avi` | 数据集配对设备 HR；不是直接 ECG |
| Kaggle | 项目 `videos/kaggle_motion_subject001/video.mov` | 无参考 |
| synthetic72 | 项目 `videos/self_test_72bpm.mp4` | 已知 72 bpm 色彩调制；非真人生理验证 |
| data1 | 工作区 `rppg_data1_20260909/inputs/video/Video_20260909_YJC_2.avi` | Polar 设备 HR，沿用冻结估计同步 |

UBFC 的 WSL 原参考及工作区评分副本均存在且相同：`836852f57f9cc9e04ecab134ebe506f831ad3e7d95ee73771ab8ac32ce845b67`。data1 的原始七个参考文件全部存在；`polar_hr.csv` 为 `ecdd31359f0285ebb0d91834d85fc9faeaf0c1bea9e75a331b3a64bcc2f5daa6`，冻结对齐协议为 `cdde1ebc76692ebe8dd88b7670c6dd1fb56d1a83048c939f5cdb791d5df0ddc5`。不把这些文件存在视为已验证真实硬件同步。

WSL 项目现存视频共 5 个，其中 Kaggle `clips/motion_first60.mp4` 是裁剪副本。Windows 工作区另有 `Video_20260907_171742478_playback.mp4`，是原 0907 的转码副本。这两份均不能当作新增独立样本。除显式指定的 0904 桌面源外，本次未扩展搜索用户其他个人目录。在已检查的项目与工作区内，没有发现新的独立原视频。仍是 5 段真人加 1 段合成；597 个重叠评价窗口不等于 597 个独立样本。

旧缓存只能复现原有前端。新增像素跟踪／运动筛选必须按新源码和参数重新生成缓存，并让新验证记录绑定原视频与新缓存哈希；不能给旧 RGB 缓存改版本号后冒充新前端结果。

## 部署设计

1. 安装到新的 `motion_upgrade_v23_20260910/`，证据保存在新的 `results/motion_v23_20260910/`。目标已存在时验证原安装记录，或报错；不得覆盖旧证据。
2. 先提供明确命名的 `run_motion_v23_experimental.sh`。技术测试和评价对账通过，但性能门槛未通过时，可仅安装该实验入口，并记录失败门槛和原因。
3. 仅在主代理给出的独立评测门槛结果通过时，才允许把稳定别名 `run_motion.sh` 指向 V2.3。保留现有 `run.sh` 和全部版本入口，以免破坏旧 POS/CHROM 参数契约。具体默认推广目标仍需主代理确认。
4. `deploy_v23.py` 默认只做计划／预检。实际安装必须使用 `--install --decision PATH`；决策文件须明确授权安装并绑定冻结源码、测试、全量运行、独立评价与 QA 文件哈希。无结论、缺证据、源码变化或 QA 不通过均停止。性能门槛失败时禁止默认推广。
5. 复制白名单源码、测试和评价证据；不复制原视频、设备原始参考、虚拟环境或外部仓库。逐文件校验，保留新部署清单；安装前后对旧入口及源码哈希复核。路径必须位于明确项目目录，拒绝符号链接和越界路径。
6. 将安装与默认推广拆开：安装新目录和实验入口后，主代理用已冻结的新缓存运行代表性已验证案例，核对保存波形、接受状态和两种 HR 读出。只有安装核验通过且性能门槛通过，才推广默认入口。当前数据不足以据此宣称跨人群泛化或实时监测通过。

部署脚本已按本轮最终 schema 对接：要求 6 来源 × 4 固定模式的 24 个运行组合完整，测试文件哈希一致，冻结门槛哈希一致，评价基线复现和独立实际波形回放通过。几何来源、标记和缺失掩码严格一致；几何数值及 gap 配对缓存使用已冻结的 1e-9 序列化容差，不额外要求 CSV 字节相同。决策中的性能门槛结果必须与 `evaluation/upgrade_gate.json` 中 `default.eligible_for_named_upgrade` 相同。

决策文件 `deployment_decision.json` 字段：`schema_version=1`、`installation_authorized=true`、`mode=experimental|promote_default`、`performance_gate_passed`、`reason`、`entry_point=analyze_motion_v2.py`、`source_hashes`、`evidence`（相对路径到 SHA-256）、`expected_prior_default_sha256`。`launcher_args` 必须为 `["--pixel-mode", "tracking_screened", "--max-gap", "0.10"]`，因为原 CLI 默认像素跟踪是关闭的，不能仅改入口名而未启用新前端。

`evidence` 至少绑定以下文件：

- `validation/protocol_before_validation.json`
- `validation/tests.json`、`validation/runs.json`、`validation/waveform_replay_qa.json`
- `evaluation_upgrade_criteria.json`
- `evaluation/summary.json`、`evaluation/upgrade_gate.json`

额外的 `independent_qa.json` 如有可一同绑定，提供时须标明 `passed=true`；它不是在已冻结工程门槛之外新增的强制文件。独立波形回放依据已约定的 `validation/waveform_replay_qa.json` 和评价摘要中的核验结果。

推广前的 `deployment_verification.json` 须有 `installed_entry_verified=true`、`deployment_manifest_sha256`（绑定工作区 `deployment.json`）、同一 `source_hashes`，以及至少两个不同案例的 `cases` 列表；每例应有 `case`、`exit_code=0`、`matches_validation=true`。该核验文件由实际安装后的回放产生，不能在当前准备阶段预填通过。

准备阶段可执行 `python deploy_v23.py`，它只输出计划；目前已在 Windows 环境完成这一无写入检查。实际安装和推广分别为 WSL Python 下的 `deploy_v23.py --install --decision deployment_decision.json` 与后续 `deploy_v23.py --promote-default --decision deployment_decision.json`。已有安装再次执行 `--install` 只对账原清单，既有目录无本部署清单则报错。失败时保留明确命名的 staging 目录供检查，不进行递归删除或覆盖重试。

准备阶段还完成了纯本地门槛检查：7 种越界／歧义路径、错误文件哈希、未授权安装、性能失败却申请默认推广，共 10 个错误输入均被拒绝；正确哈希和显式开启候选的启动器内容通过。尚未声称真实安装、安装后回放或默认晋级已通过。
