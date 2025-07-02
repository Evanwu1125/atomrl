# AtomRL

一个原子化的强化学习算法库，专注于将复杂的RL算法分解为可独立调用和组合的原子函数。

## 🎯 项目理念

AtomRL的核心思想是将强化学习算法拆解为最小的、可复用的原子函数单元。每个函数都：
- 具有明确定义的输入输出接口
- 可通过 `--help` 参数查看详细信息
- 返回带有形状信息的张量
- 可独立测试和调试
- 支持灵活组合构建复杂算法

## 🚀 快速开始

### 安装依赖
```bash
pip install torch transformers datasets accelerate deepspeed tensorboard
```

### 运行GRPO训练示例
```bash
cd scripts/grpos/grpo
accelerate launch --config_file config.yaml grpo.py
```

## 📦 当前实现

### GRPO (Group Relative Policy Optimization)

基于PPO的改进算法，针对语言模型优化设计。

#### 主要组件

**核心原子函数：**
- `generate_samples()` - 样本生成
- `get_action_log_probs()` - 动作概率计算  
- `compute_advantages()` - 优势函数计算
- `compute_policy_loss()` - 策略损失计算
- `compute_kl_penalty()` - KL散度惩罚

**奖励函数模块：**
- `correctness_reward()` - 答案正确性奖励
- `digit_reward()` - 数字格式奖励
- `hard_format_reward()` - 严格格式奖励
- `mark_reward()` - 标记完整性奖励

#### 使用示例

```python
from atomrl.grpo import GRPOTrainer, GRPOArgs
from atomrl.rewards import correctness_reward, digit_reward

# 配置参数
args = GRPOArgs()
args.lr = 1e-6
args.num_generations = 4

# 定义奖励函数组合
reward_functions = [correctness_reward, digit_reward]

# 创建训练器
trainer = GRPOTrainer(
    model="your-model-path",
    tokenizer="your-tokenizer-path", 
    reward_funcs=reward_functions,
    args=args
)

# 开始训练
trainer.train()
```

## 🔧 原子函数设计规范

### 函数签名标准
```python
def atomic_function(
    inputs: torch.Tensor,  # 输入张量
    mask: Optional[torch.Tensor] = None,  # 可选掩码
    **kwargs
) -> Dict[str, torch.Tensor]:  # 返回字典包含所有输出
    """
    原子函数的标准模板
    
    Args:
        inputs: 输入张量 [batch_size, seq_len, hidden_dim]
        mask: 注意力掩码 [batch_size, seq_len]
        
    Returns:
        {
            'output': 主要输出张量 [batch_size, output_dim],
            'intermediate': 中间结果 [batch_size, seq_len],
            'metadata': 元数据信息
        }
    """
    pass
```

### 帮助信息标准
每个原子函数都支持：
```bash
python -c "from atomrl.grpo import get_action_log_probs; help(get_action_log_probs)"
```

## 📊 数据流可视化

```
输入Prompts → generate_samples() → 采样结果
     ↓
计算概率 → get_action_log_probs() → 动作概率
     ↓  
计算奖励 → reward_functions() → 奖励值
     ↓
计算优势 → compute_advantages() → 优势函数
     ↓
计算损失 → compute_policy_loss() → 策略损失
     ↓
反向传播 → optimizer.step() → 参数更新
```

## 📁 项目结构

```
atomrl/
├── LICENSE                    # MIT许可证
├── README.md                 # 项目说明
├── scripts/                  # 算法实现脚本
│   └── grpos/               # GRPO算法
│       └── grpo/
│           ├── config.yaml  # 训练配置
│           ├── grpo.py      # 主要实现
│           ├── prompt.txt   # 系统提示
│           └── reward_func.py # 奖励函数
├── datasets/                # 数据集相关
└── weights/                 # 模型权重存储
```

## 🔄 扩展计划

### 即将支持的算法
- [ ] PPO (Proximal Policy Optimization)
- [x] DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization)
- [ ] DPO (Direct Preference Optimization)  
- [ ] RLHF (Reinforcement Learning from Human Feedback)
- [ ] A2C (Advantage Actor-Critic)
- [ ] TRPO (Trust Region Policy Optimization)

### 功能路线图
- [ ] 命令行工具支持
- [ ] 可视化界面
- [ ] 实验记录和对比
- [ ] 自动超参数调优
- [x] 分布式训练优化

## 🤝 贡献指南

欢迎为AtomRL贡献代码！请确保：

1. 遵循原子函数设计规范
2. 添加完整的文档字符串
3. 包含单元测试
4. 更新相关README

## 📄 许可证

本项目使用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

感谢开源社区为强化学习算法发展做出的贡献。

---

**让RL算法像搭积木一样简单！** 🧱✨