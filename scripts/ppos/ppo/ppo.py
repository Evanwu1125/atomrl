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

        # 设置stop token id来让模型输出正常的长度
        if hasattr(self.args, 'stop_token_id') and self.args.stop_token_id is not None:
            self.stop_token_id = self.args.stop_token_id
        elif hasattr(self.args, 'stop_token') and self.args.stop_token == "eos":
            self.stop_token_id = self.tokenizer.eos_token_id
        else:
            # 默认使用eos token作为stop token
            self.stop_token_id = self.tokenizer.eos_token_id

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
            train_dataset,
            batch_size = self.args.batch_size,
            shuffle = True,
        )

        # load the eval dataset
        assert eval_dataset is not None, "eval_dataset is required"
        self.eval_dataset = DataLoader(
            eval_dataset,
            batch_size = self.args.batch_size,
            shuffle = False,
        )
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr = self.args.lr,
            weight_decay=0.01
        )

        self.model, self.optimizer, self.train_dataloader, self.eval_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.eval_dataloader
        )

    def generate_samples(self, batch):
        """
        传统ppo的方式生成样本，每个提示词只生成一个响应
        Args:
            batch: 批次数据
        Returns:
            samples: 一个batch的样本
        get_action_log_probs: ✅
        prepare_prompts: ✅
        generate_responses: ✅
        postprocess_response: ✅
        create_masks: ✅
        batch_process_samples: ✅
        """
        self.model.eval()

        # 1. 准备prompt
        inputs, prompts = self._prepare_prompts(batch)

        # 2. 生成response
        """
        response_ids是指模型返回的response
        response_length是指模型返回的response的有效长度
        context_length是指prompt的最大长度
        """
        prompt_response_ids, response_ids, context_length = self._generate_responses(inputs)

        # 3. postprocess_response
        """
        postprocessed_responses是指做出后处理的response
        （后处理的方式是把第一个eos token后面的所有token都换成padding token的处理）
        sequence_lengths是指postprocessed_responses的有效长度
        （这个长度是记录了第一个pad token的索引，这个索引的数值刚好可以对应有效长度）
        """
        postprocessed_responses, sequence_lengths = self._postprocess_response(response=response_ids)
        
        # 4. 创建 mask
        """
        attention_mask - > prompt_response_ids
        action_mask - > reponse_ids
        """
        attention_mask, action_mask, padding_mask, padding_mask_p1 = self._create_masks(prompt_response_ids, response_ids, sequence_lengths)

        # 5. 分批处理计算概率和分数
        logprobs, ref_logprobs, values, scores = self._batch_process_samples(self, )

        # action_mask的形状是(batch_size, response_length)

        # 计算response长度
        response_length = action_mask.float().sum(dim=-1)
        total_length = attention_mask.float().sum(dim=-1)

        samples = Samples(
            prompt_response_ids = prompt_response_ids,
            attention_mask = attention_mask,
            action_mask = action_mask,
            num_actions = action_mask.size(1),
            response_ids = response_ids,
            total_length = total_length,
        )

        return samples

    def _prepare_prompts(self, batch):
        """
        准备prompt文本并进行tokenize
        args:
            batch: 一批数据
        returns:
            inputs: tokenize后的结果
            prompts: 原始的prompt列表
        """
        prompts = batch.get('question')

        prompt_text = []
        for prompt in prompts:
            # 应用聊天摸版
            input_text = self.tokenizer.apply_chat_template(
                [
                    {"role" : "system", "content":self.system_prompt},
                    {"role" : "user", "content": prompt}
                ],
                add_generation_prompt = True,
                tokenize = False # 这里不做tokenize是可以留到后面做批处理的tokenize
            )
            prompt_text.append(input_text)

        inputs = self.tokenizer(
            prompt_text,
            return_tensors = 'pt',
            padding = 'max_length', # 按照设置的最大句子长度做padding
            max_length = self.args.max_prompt_length,
            truncation = True,
        ).to(self.args.device)

        return inputs, prompts

    def _get_action_log_probs(self, model, prompt_response_ids,
                             attention_mask, action_mask):
        """
        计算模型输出token的log概率
        args:
            model: 要计算的模型
            prompt_response_ids: prompt + response的token_ids
            attention_mask: prompt + response中所有有效token的掩码
            action_mask: 动作掩码
        """
        with torch.no_grad():
            output = model(
                input_ids = prompt_response_ids,
                attention_mask = attention_mask,
            )
            logits = output.logits 
            # logits的形状是(batch_size, sequence_length, vocab_size)

            response_logits = logits[:, :-1, :]
            # 获取response部分的logits（去掉最后一个token的logits)

            '''
            位置:     0   1   2   3   4   5   6
            tokens:  [Q1, Q2, Q3, R1, R2, R3, PAD]
            logits:  [L0, L1, L2, L3, L4, L5, L6]
            logits[:, :-1, :] = [L0, L1, L2, L3, L4, L5]
            logits[:, 1:, :] = [L1, L2, L3, L4, L5, L6]
            '''
            log_probs = F.log_softmax(response_logits, dim=-1)
            log_probs_labels = log_probs.gather(dim=-1, index=prompt_response_ids[:, 1:].unsqueeze(-1))
            action_log_probs = log_probs_labels.squeeze(-1)[:, -(action_mask.size(1)-1):]

        return action_log_probs
    
    def _postprocess_response(self, response):
        """
        对响应进行后处理，截断stop token
        Args:
            response: 原始响应, 形状: (batch_size, actual_length)
        Returns:
            postprocessed_response: 后处理的响应, 形状: (batch_size, actual_length)
            sequence_lengths: 每个序列的有效长度, 彩色: (batch_size,)
        """
        postprocessed_response = response.clone() 
        # postprocessed_response的形状是（batch_size, max_generate_length)

        if hasattr(self, 'stop_token_id') and self.stop_token_id is not None:
            for i in range(postprocessed_response.shape[0]):
                eos_indices = (postprocessed_response[i] == self.stop_token_id).nonzero(as_tuple = True)[0]
                # nonzero(as_tuple = True)[0] 是以元组的形式返回所有的eos token对应的索引
                if len(eos_indices) > 0:
                    first_eos = eos_indices[0].item() #取出第一个eos token的位置
                    postprocessed_response[i, first_eos+1:] = self.tokenizer.pad_token_id
                    
        sequence_lengths = [] # 列表，长度为batch_size
        for i in range(postprocessed_response.shape[0]):
            pad_indices = (postprocessed_response[i] == self.tokenizer.pad_token_id).nonzero(as_tuple = True)[0]
            if len(pad_indices) > 0:
                sequence_lengths.append(pad_indices[0].item()) # 取出第一个pad token的位置
            else:
                sequence_lengths.append(postprocessed_response.shape[1]) # 没有pad token，则整个序列都是有效的
        sequence_lengths = torch.tensor(sequence_lengths, dtype=torch.long, device=self.args.device)
        return postprocessed_response, sequence_lengths

    def _generate_responses(self, inputs):
        """
        生成模型响应
        args:
            inputs: tokenize后的所有prompt的长度
        returns:
            prompt_reponse_ids: prompt+response的长度, 形状: (batch_size, total_length)
            response_ids: 模型的回答长度
            response_length: 每个响应的有效长度, 形状: (batch_size,)
        """
        context_length = inputs['input_ids'].shape[1]

        with torch.no_grad():
            prompt_response_ids = self.model.generate(
                **inputs,
                max_new_tokens = self.args.max_generate_length,
                pad_token_id = self.tokenizer.pad_token_id, # 设置pad_token_id虽然不会自动填充，但是可以保证模型不去采样padding token
                eos_token_id = self.tokenizer.eos_token_id,
                do_sample = True,
                temperature = 1.0,
                top_p = 0.95,
            )
        
        #response_ids是指模型生成的回答长度
        response_ids = prompt_response_ids[:, context_length:]
        
        return prompt_response_ids, response_ids, context_length
    
    def _create_masks(self, prompt_response_ids, responses_ids, sequence_lengths):
        """
        创建各种掩码
        Args:
            prompt_response_ids: prompt + response的内容
            responses_ids: 模型的回答
            sequence_lengths: 每个响应的有效长度(做过postprocess之后的长度，就是到第一个pad token的长度)
        Returns:
            attention_mask: prompt + response中所有有效token的掩码
            action_mask: 动作掩码
            padding_mask: padding掩码
            padding_mask_p1: padding掩码 + 1
        """
        
        attention_mask = (prompt_response_ids != self.tokenizer.pad_token_id).long()

        # 创建动作掩码
        is_not_pad = responses_ids != self.tokenizer.pad_token_id
        is_not_eos = responses_ids != self.tokenizer.eos_token_id
        action_mask = (is_not_pad & is_not_eos).long()

        # 创建padding掩码
        response_idxs = torch.arange(responses_ids.shape[1], device=responses_ids.device).repeat(responses_ids.shape[0],1)
        # 形状是 (batch_size, max_generate_length)

        padding_mask = response_idxs >= sequence_lengths.unsqueeze(1)
        # 形状是 (batch_size, max_generate_length)

        padding_mask_p1 = response_idxs >= (sequence_lengths + 1).unsqueeze(1)
        # 形状是 (batch_size, max_generate_length)
        # 之所以这里 + 1是因为价值函数预测需要预测t+1位置的价值

        return attention_mask, action_mask, padding_mask, padding_mask_p1
    
    def _compute_policy_log_probs(self, prompt_response_ids_subbatch, attention_mask_subbatch, response_ids_subbatch, context_length):
        """
        计算策略模型输出token的log概率
        Args:
            prompt_response_ids_subbatch: prompt + response的总输出
            attention_mask_subbatch: prompt + response中所有有效token的掩码
            response_ids_subbatch: 模型的回答
            context_length: max_prompt的大小
        Returns:
            logprobs: 模型输出token的log概率
        """
        with torch.no_grad():
            output = self.model(
                input_ids = prompt_response_ids_subbatch,
                attention_mask = attention_mask_subbatch,
            )
            logits = output.logits 
            # logits的形状是(batch_size, sequence_length, vocab_size)

            predict_logits = logits[:, context_length-1 : 1]
            predict_logits /= (1.0+1e-7)
            log_probs = F.log_softmax(predict_logits, dim=-1)
            log_probs = log_probs.gather(2, response_ids_subbatch.unsqueeze(-1)).squeeze(-1)

            return log_probs
        
    def _compute_ref_log_probs(self, prompt_response_ids_subbatch, attention_mask_subbatch, response_ids_subbatch, context_length):
        """
        计算参考模型的log模型
        Args:
            prompt_response_ids_subbatch: prompt + response的总输出
            attention_mask_subbatch: prompt + response中所有有效token的掩码
            response_ids_subbatch: 模型的回答
            context_length: max_prompt的大小
        Returns:
            ref_logprobs: 参考模型输出token的log概率    
        """
        # 先判断是否有ref model
        if self.args.beta == 0.0 or self.ref_model is None:
            return None
        
        with torch.no_grad():
            ref_output = self.ref_model(
                input_ids = prompt_response_ids_subbatch,
                attention_mask = attention_mask_subbatch,
            )
            ref_logits = ref_output.logits

            ref_predict_logits = ref_logits[:, context_length-1 : 1]
            ref_predict_logits /= (1.0+1e-7)
            ref_log_probs = F.log_softmax(ref_predict_logits, dim=-1)
            ref_log_probs = ref_log_probs.gather(2, response_ids_subbatch.unsqueeze(-1)).squeeze(-1)

            return ref_log_probs

    def _compute_values(self, prompt_response_ids_subbatch, attention_mask_subbatch, context_length):
        """
        计算价值函数的输出
        Args:
            prompt_response_ids_subbatch: prompt + response的总输出
            attention_mask_subbatch: prompt + response中所有有效token的掩码
            context_length: max_prompt的大小
        Returns:
            values: 价值函数的输出
        """
    def _bacth_process_samples(self, prompt_response_ids, responses_ids, postprocessed_responses_ids, attention_mask, context_length):
        """
        批量处理样本
        Args:
            prompt_response_ids: prompt + response的总输出
            response_ids: 光是模型的输出
            postprocessed_responses: 经过后处理的response
            attention_mask: prompt + response中所有有效token的掩码
            context_length: max_prompt的大小
        Returns:
            logprobs: 模型输出token的log概率
            ref_logprobs: 引用模型输出token的log概率
            values: 价值函数的输出
            scores: 奖励函数的输出
        """
        logprobs_list = []
        ref_logprobs_list = []
        values_list = []
        scores_list = []
        for i in range(0, prompt_response_ids.shape[0]):
            prompt_response_ids_subbatch = prompt_response_ids[i].unsqueeze(0)
            responses_ids_subbatch = responses_ids[i].unsqueeze(0)
            postprocessed_responses_ids_subbatch = postprocessed_responses_ids[i].unsqueeze(0)
            attention_mask_subbatch = attention_mask[i].unsqueeze(0)

            # 计算各种概率和分数
            logprobs = self._compute_policy_log_probs() # to be updated.    
            ref_logprobs = self._compute_ref_log_probs()
            values = self._compute_values()

            # 计算reward
            postprocessed_prompt_response_ids = torch.cat([
                prompt_response_ids_subbatch[:, :context_length],
                postprocessed_responses_ids_subbatch,
            ], dim = 1)
            scores = self._compute_reward_scores()

            logprobs_list.append(logprobs)
            if ref_logprobs is not None:
                ref_logprobs_list.append(ref_logprobs)
            values_list.append(values)
            scores_list.append(scores)

        return logprobs_list, ref_logprobs_list, values_list, scores_list
    
    def _compute_reward_scores(self, postprocessed_prompt_response_ids, context_length):
        