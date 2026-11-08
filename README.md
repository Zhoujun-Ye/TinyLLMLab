# TinyLLMLab

面向初学者的单轮 LLM RL 骨架：从已有语言模型出发，生成回答、评分、
计算优势和策略损失，再更新模型。不做预训练、SFT 或 agent 工具交互。

## 文件分工

```text
TinyLLMLab/
├── train.py             运行入口，读取配置并调用 trainer
├── trainer.py           串起采样、奖励、算法、更新与验证
├── models/policy.py     加载已有模型/tokenizer，计算 log probability
├── datasets/dataset.py  prompt 数据读写与校验
├── rollout.py           单轮回答采样，保留 token、mask、分组等信息
├── reward.py            回答评分
├── algorithms/
│   ├── grpo.py          分组优势和策略目标
│   └── ppo.py           后续学习 PPO 的优势和目标
├── configs/
│   ├── grpo.toml
│   └── ppo.toml
├── utils.py             配置读取，后续按需添加 seed/checkpoint 工具
├── check.py             配置和数据检查
├── data/                examples、raw、processed
├── outputs/             权重、日志、轨迹和实验结果
└── tests/
```

只有这一份说明。算法先各用一个文件，变复杂后再拆目录。
不需要环境 step/reset、多轮 agent loop、搜索工具或分布式框架。
目前也没有 verl 接入；后续有需要再加，不把它作为学习前置要求。

## 数据怎么流动

```text
prompt → rollout 生成回答 → reward 评分
       → algorithm 计算优势/损失 → trainer 更新策略 → 下一轮采样
```

JSONL 每行包含 id、prompt、可选 ground_truth 和 metadata。
ground_truth 只给奖励计算，不传入模型提示。初期 prompt 用字符串，
实际 chat template 和 tokenization 由后续的模型/采样实现负责。

可读 JSONL 不是训练 batch。你需要明确 response token、attention/loss mask、
旧策略 log probability、prompt/group ID 和结束原因；prompt/padding 不算生成动作。
参考策略 log probability、KL 及 critic/value 数据按所实现方法需要接入。
PPO 通常还需要价值估计，不能仅换一个 loss 文件就假设整条训练链路兼容。

## 先实现什么

1. 选一个小模型和可明确判分的任务，先固定数据划分与计算预算。
2. 实现 policy 加载和单轮 rollout，先检查一条回答及 token mask，不训练。
3. 实现 reward，用手写的正确/错误回答测试，不调用模型。
4. 实现 GRPO 优势和损失，用小张量验证分组、零方差与 mask。
5. 实现 trainer，检查一次参数更新、生成变化和保存恢复，再扩大实验。
6. 之后再考虑 PPO；agentic RL 留到单轮训练清楚之后。

GRPO 是建议起点，不是已实现或自动选择的训练算法。当前未定义完整张量批次
协议；避免一开始引入复杂的注册器、抽象类或统一 Trainer。

## 本地检查

Python 3.11+，在根目录运行，无需安装项目：

```bash
source .venv/bin/activate
python check.py inspect-config
python check.py validate-data data/examples/prompts.jsonl
python -m unittest discover -s tests -v
python train.py --config configs/grpo.toml
```

没有环境时可先运行 uv venv --python 3.11 .venv。
只有配置读取、JSONL 读写和基本校验可用；其余入口均明确抛出未实现错误。
没有模型、奖励、采样、算法或参数更新实现，没有 GPU/verl 依赖或模型下载。
示例不是训练集，配置不包含未经验证的超参数，不代表训练就绪。
