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
    logprobs: List[torch.Tensor] # 动作log概率列表
    ref_logprobs: List[torch.Tensor]
    values: List[torch.Tensor]
    rewards: List[torch.Tensor]
    advantages: List[torch.Tensor]
    returns: List[torch.Tensor]

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
    # ppo_epochs = 4

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
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size = self.args.batch_size,
            shuffle = True,
        )

        # load the eval dataset
        assert eval_dataset is not None, "eval_dataset is required"
        self.eval_dataloader = DataLoader(
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

        answers = batch.get('answer', [None] * len(prompts))
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
        logprobs, ref_logprobs, values, scores = self._batch_process_samples(prompt_response_ids, response_ids, postprocessed_responses, attention_mask, context_length, answers)

        # action_mask的形状是(batch_size, response_length)

        # 6. 计算advantage和回报
        advantages, returns = self._compute_advantages_and_returns(
            values_list = values,
            scores_list = scores,
            action_mask = action_mask
        )

        # 计算response长度
        response_length = action_mask.float().sum(dim=-1)
        total_length = attention_mask.float().sum(dim=-1)

        # 构建样本对象
        samples = Samples(
            prompt_response_ids = prompt_response_ids,
            attention_mask = attention_mask,
            action_mask = action_mask,
            num_actions = action_mask.size(1),
            response_ids = response_ids,
            total_length = total_length,
            logprobs = logprobs,
            ref_logprobs = ref_logprobs,
            values=values,
            rewards=scores,
            advantages = advantages,
            returns=returns,
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

            predict_logits = logits[:, context_length-1 : -1]
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

            ref_predict_logits = ref_logits[:, context_length-1 : -1]
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
        if not hasattr(self.model, 'value_head'):
            self.model.value_head = nn.Linear(self.model.config.hidden_size, 1).to(self.args.device)
        
        with torch.no_grad():
            output = self.model(
                input_ids = prompt_response_ids_subbatch,
                attention_mask = attention_mask_subbatch,
                output_hidden_states=True
            
            )
            hidden_states = output.hidden_states[-1]  # (batch_size, seq_len, hidden_size)
            value_output = self.model.value_head(hidden_states)  # (batch_size, seq_len, 1)
            values = value_output.squeeze(-1)  # (batch_size, seq_len)
            response_values = values[:, context_length:]
            return response_values
        
    def _batch_process_samples(self, prompt_response_ids, responses_ids, postprocessed_responses_ids, attention_mask, context_length, answers):
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
            current_answer = [answers[i] if answers and i < len(answers) else None]

            # 计算各种概率和分数
            logprobs = self._compute_policy_log_probs(
                prompt_response_ids_subbatch,
                attention_mask_subbatch,
                responses_ids_subbatch,
                context_length,
            ) # to be updated.    
            ref_logprobs = self._compute_ref_log_probs(
                prompt_response_ids_subbatch,
                attention_mask_subbatch,
                responses_ids_subbatch,
                context_length,
            )
            values = self._compute_values(
                prompt_response_ids_subbatch,
                attention_mask_subbatch,
                context_length,
            )

            # 计算reward
            postprocessed_prompt_response_ids = torch.cat([
                prompt_response_ids_subbatch[:, :context_length],
                postprocessed_responses_ids_subbatch,
            ], dim = 1)
            
            scores = self._compute_reward_scores(
                postprocessed_prompt_response_ids,
                context_length,
                current_answer,
            )

            logprobs_list.append(logprobs)
            if ref_logprobs is not None:
                ref_logprobs_list.append(ref_logprobs)
            values_list.append(values)
            scores_list.append(scores)

        return logprobs_list, ref_logprobs_list, values_list, scores_list
    
    def _compute_reward_scores(self, postprocessed_prompt_response_ids, context_length, answers=None):
        """
        计算奖励分数
        Args:
            postprocessed_prompt_response_ids: 后处理的prompt + response
            context_length: prompt的长度
        Returns:
            scores: 奖励分数
        """
        response_texts = []
        prompt_texts = []

        for i in range(postprocessed_prompt_response_ids.shape[0]):
            # 提取prompt部分
            prompt_ids = postprocessed_prompt_response_ids[i, :context_length]
            prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
            prompt_texts.append(prompt_text)

            # 提取response部分
            response_ids = postprocessed_prompt_response_ids[i, context_length:]
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            response_texts.append(response_text)
        
        if answers is None:
            answers = [None] * len(prompt_texts)
        # 计算总奖励
        total_rewards = torch.zeros(len(response_texts), device=self.args.device)

        # 如果有奖励函数，则计算奖励
        if hasattr(self, 'reward_funcs') and self.reward_funcs:
            for i, reward_func in enumerate(self.reward_funcs):

                reward_values = reward_func(
                    prompts = prompt_texts,
                    responses=response_texts,
                    answers=answers
                )

                reward_tensor = torch.tensor(reward_values, device = self.args.device, dtype=torch.float)

                if (hasattr(self.args, 'reward_weights') and self.args.reward_weights and len(self.args.reward_weights) == len(self.reward_funcs)):
                    weights = self.args.reward_weights[i]
                else:
                    weights = [1.0] * len(self.reward_funcs)    
                
                total_rewards += weights * reward_tensor

        return total_rewards
    
    def _compute_advantages_and_returns(self, values_list, scores_list, action_mask, gamma=0.99, lambd=0.95):
        """
        计算advantage和回报
        Args:
            values_list: 价值函数的输出
            scores_list: 奖励分数
            action_mask: 动作掩码
        Returns:
            advantages: 优势
            returns: 回报
        """
        batch_advantages = []
        batch_returns = []

        for i in range(len(values_list)):
            values = values_list[i] # (1, response_length)
            rewards = scores_list[i]
            current_action_mask = action_mask[i:i+1] 

            response_length = current_action_mask.sum().item()
            reward_sequence = torch.zeros_like(values)
            
            if response_length > 0:
                reward_sequence[0, response_length - 1] = rewards

            # 计算gae优势
            advantages = torch.zeros_like(values)
            returns = torch.zeros_like(values)

            gae = 0
            for t in reversed(range(response_length)):
                if t == response_length - 1:
                    next_value = 0
                else:
                    next_value = values[0, t + 1]

                delta = reward_sequence[0, t] + gamma * next_value - values[0, t]
                gae = delta + gamma * lambd * gae
                advantages[0, t] = gae
                returns[0, t] = gae + values[0, t]

            batch_advantages.append(advantages)
            batch_returns.append(returns)
    
        return batch_advantages, batch_returns

    def _compute_policy_loss(self, current_logprobs, old_logprobs, advantages, action_mask):
        """
        计算策略损失
        Args:
            current_logprobs: 当前策略的log概率
            old_logprobs: 旧策略的log概率
            advantages: 优势
            action_mask: 动作掩码
        Returns:
            policy_loss: 策略损失
        """
        # 计算重要性采样比率
        log_ratio = current_logprobs - old_logprobs
        ratio = torch.exp(log_ratio)

        #ppo clipped objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.args.clip_eps, 1.0 + self.args.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2)
        policy_loss = policy_loss * action_mask
        
        policy_loss = (policy_loss.sum(dim=1) / action_mask.sum(dim=1).clamp(min=1)).mean()
        return policy_loss

    def _compute_value_loss(self, current_values, old_values, returns, action_mask):
        """
        计算价值损失
        Args:
            current_values: 当前策略的价值
            old_values: 旧策略的价值
            returns: 回报
            action_mask: 动作掩码
        Returns:
            value_loss: 价值损失
        """
        #clipped value loss
        value_pred_clipped = old_values + (current_values - old_values).clamp(-self.args.clip_eps, self.args.clip_eps)

        value_losses = (current_values - returns) ** 2 # mse loss
        value_losses_clipped = (value_pred_clipped - returns) ** 2 # mse loss
        value_loss = torch.max(value_losses, value_losses_clipped)
        value_loss = value_loss * action_mask.float()
        value_loss = (value_loss.sum(dim=1) / action_mask.sum(dim=1).clamp(min=1)).mean()
       
        return value_loss

    def _compute_kl_divergence(self, current_logprobs, ref_logprobs, action_mask):
        """
        计算KL散度
        Args:
            current_logprobs: 当前策略的log概率
            old_logprobs: 旧策略的log概率
            action_mask: 动作掩码
        Returns:
            kl_divergence: KL散度
        """
        if ref_logprobs is None:
            return 0.0
        
        log_ratio = current_logprobs - ref_logprobs
        kl_divergence = log_ratio.exp() - 1 - log_ratio
        kl_divergence = kl_divergence * action_mask.float()
        kl_divergence = (kl_divergence.sum(dim=1) / action_mask.sum(dim=1).clamp(min=1)).mean()
        return kl_divergence

    def train(self):
        for epoch in range(self.args.epoch):
            for iteration in range(self.args.num_iterations):
                print(f"Epoch {epoch}, Iteration {iteration}")
                
                # 收集经验
                all_samples = []
                for batch_idx, batch in enumerate(self.train_dataloader):
                    print(f"Batch {batch_idx}")
                    samples = self.generate_samples(batch)
                    all_samples.append(samples)
                
                # PPO更新 - 直接对收集的样本进行一次更新
                total_policy_loss = 0
                total_value_loss = 0
                total_kl_penalty = 0
                
                for samples in all_samples:
                    self.model.train()
                    
                    # 重新计算当前策略的logprobs和values
                    current_logprobs_list = []
                    current_values_list = []
                    
                    for i in range(len(samples.logprobs)):
                        # 重新计算当前logprobs
                        current_logprobs = self._compute_policy_log_probs(
                            samples.prompt_response_ids[i:i+1],
                            samples.attention_mask[i:i+1],
                            samples.response_ids[i:i+1],
                            samples.prompt_response_ids.shape[1] - samples.response_ids.shape[1]
                        )
                        current_logprobs_list.append(current_logprobs)
                        
                        # 重新计算当前values
                        current_values = self._compute_values(
                            samples.prompt_response_ids[i:i+1],
                            samples.attention_mask[i:i+1],
                            samples.prompt_response_ids.shape[1] - samples.response_ids.shape[1]
                        )
                        current_values_list.append(current_values)
                    
                    # 计算损失
                    policy_loss = 0
                    value_loss = 0
                    kl_penalty = 0
                    
                    for i in range(len(samples.logprobs)):
                        # 策略损失
                        policy_loss += self._compute_policy_loss(
                            current_logprobs_list[i],
                            samples.logprobs[i],
                            samples.advantages[i],
                            samples.action_mask[i:i+1]
                        )
                        
                        # 价值损失
                        value_loss += self._compute_value_loss(
                            current_values_list[i],
                            samples.values[i],
                            samples.returns[i],
                            samples.action_mask[i:i+1]
                        )
                        
                        # KL惩罚
                        if samples.ref_logprobs and samples.ref_logprobs[i] is not None:
                            kl_penalty += self._compute_kl_penalty(
                                current_logprobs_list[i],
                                samples.ref_logprobs[i],
                                samples.action_mask[i:i+1]
                            )
                    
                    # 总损失
                    total_loss = policy_loss + 0.5 * value_loss + kl_penalty
                    
                    # 反向传播
                    self.accelerator.backward(total_loss)
                    
                    if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    
                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_kl_penalty += kl_penalty if isinstance(kl_penalty, float) else kl_penalty.item()
                
                print(f"Iteration {iteration}: Policy Loss: {total_policy_loss:.4f}, "
                      f"Value Loss: {total_value_loss:.4f}, KL Penalty: {total_kl_penalty:.4f}")
