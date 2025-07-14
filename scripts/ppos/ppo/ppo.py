import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
from typing import Optional, Union, Tuple
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from copy import deepcopy
from datasets import load_dataset
from reward_func import *
import torch.nn.functional as F
import os
import json
from reward_func import *
import datetime
import deepspeed
from accelerate import Accelerator
from trl import PPOTrainer

@dataclass
class Samples:
    prompt_response_ids: torch.Tensor # prompt + response的长度
    attention_mask: Optional[torch.LongTensor] # 把padding token给跳过
    action_mask: Optional[torch.BoolTensor] #哪些token是response token
    num_actions: Union[int, torch.Tensor] #有多少response token
    response_ids: torch.Tensor # response token ids
    total_length: torch.Tensor # prompt + response的长度


class PPOArgs:
    output_dir = './output'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 0.000001
    save_steps = 100
    epoch = 3
    max_prompt_length = 256 # 最长输出的prompt
    max_generate_length = 512 # 最长的模型回复
    reward_weights : List[float] = None # 奖励的权重（多个奖励函数）
    beta = 0.1 # KL散度的系数，为0则忽略KL散度，即不使用参考模型
    clip_eps = 0.2
    gradient_accumulation_steps = 2 # gradient accumulation steps
    num_iterations = 3 # number of iterations
    batch_size = 1 # batch size

class PPOTrainer:
    def __init__(self,
        model = None,
        reward_funcs: Union[List[str], List[Callable]] = None,
        args = None,
        train_dataset: Optional[Union[Dataset]] = None,
        eval_dataset: Optional[Union[Dataset]] = None,
        tokenizer = None,
        accelerator=None):
        '''
        Args:
            model: 模型
            reward_funcs: 奖励函数
            args: 参数
            train_dataset: 训练数据集
            eval_dataset: 验证数据集
            tokenizer: 分词器
        '''
        self.args = args
        self.accelerator = accelerator
        # 加载模型
        assert model is not None, "model is required"
        model = AutoModelForCausalLM.from_pretrained(model)
        self.model = model.to(self.args.device)
        
        # 启动flash attention
        if hasattr(self.model.config, 'attn_implementation'):
            self.model.config.attn_implementation = "flash_attention_2"
        
        # 加载参考模型
        self.ref_model = deepcopy(self.model)
        self.ref_model.eval()
        
        # 加载分词器
        assert tokenizer is not None, "tokenizer is required"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'left'

        # 确保pad_token设置正确
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载system prompt
        with open('/workspace/minimind/reproduce/algorithms/PPO/prompt.txt', 'r', encoding='utf-8') as f:
            self.system_prompt = f.read().strip()
        print(f"[INFO] Loaded system prompt: {self.system_prompt}")

        # load the reward funcs
        if reward_funcs is not None:
            self.reward_funcs = []
            for reward_func in reward_funcs:
                self.reward_funcs.append(reward_func)
        else:
            ### TODO: load the default format reward and accuray reward
            pass

        # load the ref model if beta is not 0
        if self.args.beta != 0.0:
            self.ref_model = deepcopy(self.model)
            self.ref_model.eval()

        # load the train dataset
        assert train_dataset is not None, "train_dataset is required"
        self.train_dataset = DataLoader(
            self.train_dataset,
            batch_size = self.args.batch_size,
            shuffle = True,
        )

        # load the eval dataset
        assert eval_dataset is not None, "eval_dataset is required"
        self.eval_dataset = DataLoader(
            self.eval_dataset,
            batch_size = self.args.batch_size,
            shuffle = False,
        )

        self.model, self.optimizer, self.train_dataloader, self.eval_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.eval_dataloader
        )

    def 

        