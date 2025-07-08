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
    num_generations = 4 # number of generations
    max_prompt_length = 256 # max prompt length
    max_generate_length = 512 # max generate length

class PPOTrainer: