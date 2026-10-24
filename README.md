# OpenVLA-7B LoRA × SimplerEnv × WidowX

机器学习课程 B 类大作业：在 BridgeData V2 子集上对 [`openvla/openvla-7b`](https://huggingface.co/openvla/openvla-7b) 做 LoRA 轻量微调，并在 SimplerEnv WidowX 的两个任务上闭环评测 Base 与 Final。

- 项目类别：B 类（工程实践型）
- 实验报告：[`docs/实验报告.md`](docs/实验报告.md) ｜ [`docs/实验报告.pdf`](docs/实验报告.pdf)
- 早期 Octo 探索说明：[`docs/B_TASK_REPORT.md`](docs/B_TASK_REPORT.md)

## 1. 正式结果

正式初始状态在 [`config/eval_states_b_formal.json`](config/eval_states_b_formal.json)。Base 与 Final 各在 T1、T2 上运行 10 个 episode，共 40 局。

| 模型 | 训练模块 | T1 (`widowx_spoon_on_towel`) | T2 (`widowx_put_eggplant_in_basket`) | B-SimplerScore |
| --- | --- | ---: | ---: | ---: |
| Base `openvla/openvla-7b` | 无本次微调 | 1/10 | 1/10 | 10.0 |
| Final，验证损失最低的 step 300 | LoRA `q_proj`/`v_proj`，rank 8，α 16 | 0/10 | 5/10 | 25.0 |

T1 变化 −10.0 个百分点，T2 变化 +40.0 个百分点，平均分变化 **+15.0 个百分点**。逐 episode 记录见 [`results/b_formal/episodes.csv`](results/b_formal/episodes.csv)，汇总见 [`results/b_formal/results.csv`](results/b_formal/results.csv)。

代表视频在 [`results/b_formal/representative/`](results/b_formal/representative/)：

- `base_t1_seed1119_success.mp4` — Base 在 T1 的一次成功
- `base_t2_seed2219_fail.mp4` — Base 在 T2 的失败
- `final_t1_seed1111_fail.mp4` — Final 在 T1 的失败
- `final_t2_seed2219_success.mp4` — Final 在 T2 的一次成功

## 2. 模型与 checkpoint

| 项目 | 位置 |
| --- | --- |
| Base 模型 | <https://huggingface.co/openvla/openvla-7b> |
| Final LoRA adapter | 17MB，见 [`checkpoint_link.txt`](checkpoint_link.txt) 的下载链接 |
| 加载方式 | `PeftModel.from_pretrained(base, adapter_path)`，或先 merge 后 `AutoModelForVision2Seq` |
| 推理配置 | `unnorm_key=bridge_orig`，`do_sample=False`，见 [`scripts/serve_openvla.py`](scripts/serve_openvla.py) |
| 训练配置 | [`checkpoints/openvla_bridge_lora_v3/train_config.json`](checkpoints/openvla_bridge_lora_v3/train_config.json) |
| 训练日志 | [`logs/openvla_bridge_lora_v3.jsonl`](logs/openvla_bridge_lora_v3.jsonl) |

仓库内**不存放** Base 权重（7B，~14GB）和 LoRA adapter 权重（17MB，走下载链接）。

## 3. 数据

训练样本来自 BridgeData V2，种子 42 抽取 60 个训练分片和 12 个验证分片（约 7.7GB）。按以下流程过滤：

1. 先筛出"表面放置"和"容器放置"两类；
2. 保留指令含 `spoon`/`towel`/`eggplant`/`basket` 的轨迹；
3. 额外 80 个未使用训练分片只补充训练集；
4. 丢弃静止步（xyz 位移 < 5mm，旋转 < 0.02 rad）。

最终：548 条训练轨迹（12 802 步），128 条验证轨迹（535 步）。动作前 6 维使用 Base 自带 `bridge_orig` 的 q01/q99 归一化，夹爪维保持 0/1。

数据样例和统计在 [`artifacts/data_report/`](artifacts/data_report/)。

## 4. 训练超参数

| 项 | 值 |
| --- | --- |
| 可训练模块 | LoRA，targets = `q_proj`, `v_proj` |
| rank / α | 8 / 16 |
| 总参数 / 可训练 | 7.55B / 4.19M（0.056%） |
| 最大步数 | 1200（early stop patience 4） |
| 学习率 | 1e-4，warmup 50 步 |
| 梯度累积 | 8（有效 batch = 8） |
| 精度 | 权重 bfloat16，LoRA 权重 float32 |
| Final 选择规则 | 验证集动作 token loss 最低（early stop 触发后停） |
| 种子 | 42 |

## 5. 环境与依赖

- Ubuntu 22.04，CUDA 12.4，单卡 ≥ 24GB（RTX 3090/4090/A5000 均可）
- Python 3.10
- 核心依赖见 [`requirements.txt`](requirements.txt)

安装：

```bash
conda create -n vlalora python=3.10 -y
conda activate vlalora
pip install -r requirements.txt

# SimplerEnv（editable）
git clone https://github.com/allenai/SimplerEnv.git
cd SimplerEnv && git checkout c701ea271b16b80616d6fed9713c4a50a0adea3b
pip install -e ManiSkill2_real2sim
pip install -e .
```

系统库：

```bash
sudo apt-get install -y libvulkan1 libegl1 libgl1 libglvnd-dev \
  libxext6 libxrender1 libsm6 ffmpeg
```

## 6. 复现步骤

### 6.1 下载 Base 模型

```bash
huggingface-cli download openvla/openvla-7b --local-dir models/openvla-7b
```

### 6.2 下载 Final adapter

从 [`checkpoint_link.txt`](checkpoint_link.txt) 给出的链接下载，解压到 `checkpoints/openvla_bridge_lora_v3/best_adapter/`。

### 6.3 跑正式评测（Base vs Final，各 10×2 局）

```bash
bash scripts/run_b_formal.sh
python scripts/summarize_openvla_formal.py
```

脚本会分别启动 Base 和 Final 的动作服务（[`scripts/serve_openvla.py`](scripts/serve_openvla.py)），再用 SimplerEnv 闭环跑 T1/T2 各 10 局，结果写入 `results/b_formal/`。

### 6.4 重新训练（可选）

```bash
bash scripts/run_b_openvla_v3.sh
```

该命令训练结束后会在开发种子上再跑一轮 Final，那一轮只用于检查流程，不替代 `results/b_formal/`。

## 7. 目录结构

```text
.
├── README.md
├── requirements.txt
├── checkpoint_link.txt
├── scripts/                 # 训练 / 评测 / 数据处理脚本
├── config/
│   ├── eval_states_b_formal.json   # 正式评测初始状态（已固定）
│   ├── eval_states_b.json
│   └── finetune_octo_bridge.py     # 早期 Octo 训练配置
├── checkpoints/openvla_bridge_lora_v3/
│   └── train_config.json
├── logs/
│   └── openvla_bridge_lora_v3.jsonl  # 逐 step train/val loss
├── results/b_formal/
│   ├── episodes.csv          # 逐 episode 明细
│   ├── results.csv           # 汇总表
│   ├── results.json
│   └── representative/       # 4 个代表视频
├── artifacts/data_report/   # BridgeData 子集统计与样本
└── docs/
    ├── 实验报告.md / .pdf
    ├── B_TASK_REPORT.md      # 早期 Octo 方案说明
    └── figures/
```

## 8. 已知问题与分析

- **T1 上 Final 反而掉了 1 局**：T1 训练轨迹只有 12 条，LoRA 容易过拟合到 T2（5 条直接轨迹 + 大量相关轨迹）。
- **Base 在两个任务上都只有 10%**：OpenVLA-7B 默认归一化与 SimplerEnv WidowX 动作尺度需要手动对齐，详见实验报告。
- **训练未到 1200 步上限**：约 step 300 验证 loss 就不再下降，early stop 触发。

## 9. 分工

- 数据处理、训练配置、LoRA 训练
- SimplerEnv 闭环评测、参数扫描、结果整理
- PPT 与实验报告撰写
- 演讲汇报
