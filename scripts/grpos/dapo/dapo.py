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
    prompt_response_ids: torch.Tensor # 包含了prompt和模型
    response_ids: torch.Tensor # 包含了模型输出的response
    prompt: Any # 包含了prompt，每个samples只会储存一个prompt
    answer: Any # 包含了answer
    attention_mask: Optional[torch.LongTensor] # 把padding的部分给mask掉
    action_mask: Optional[torch.BoolTensor] # 把padding的部分给mask掉

class DAPOArgs:
    output_dir = './output'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 0.000001
    save_steps = 100
    epoch = 3
    num_generations = 4 # number of generations
    max_prompt_length = 256 # max prompt length
    max_generate_length = 512 # max generate length
    reward_weights : List[float] = None # reward weights
    beta = 0.1 # KL divergence coefficient
    low_clip_eps = 0.2
    high_clip_eps = 0.28
    # gradient_accumulation_steps = 2 # gradient accumulation steps
    num_iterations = 1 # number of iterations
    batch_size = 1 # batch size

class DAPOTrainer:
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
        在Trainer中，我们首先需要初始化模型、优化器、数据集和奖励函数。
        然后，我们使用Accelerator来准备模型、优化器和数据集，以便在多GPU上进行训练。
        '''
        self.args = args
        self.accelerator = accelerator
        
        assert model is not None, "model is required"
        assert tokenizer is not None, "tokenizer is required"
        assert train_dataset is not None, "train_dataset is required"
        assert eval_dataset is not None, "eval_dataset is required"
        assert accelerator is not None, "accelerator is required"
        
        model = AutoModelForCausalLM.from_pretrained(model)
        self.model = model.to(self.args.device)
        
        if hasattr(self.model.config, 'attn_implementation'):
            self.model.config.attn_implementation = "flash_attention_2"
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'left'
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        with open('/workspace/minimind/reproduce/algorithms/DAPO/prompt.txt', 'r', encoding='utf-8') as f:
            self.system_prompt = f.read().strip()
        
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
        self.eval_dataloader = DataLoader(self.eval_dataset, batch_size=self.args.batch_size, shuffle=False)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)

        if reward_funcs is not None:
            self.reward_funcs = []
            for reward_func in reward_funcs:
                self.reward_funcs.append(reward_func)
        else:
            ### TODO: load the default format reward and accuray reward
            pass
        
        if self.args.beta != 0.0: # kl散度
            self.ref_model = deepcopy(model)
            self.ref_model.eval()

        self.model, self.optimizer, self.train_dataloader, self.eval_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.eval_dataloader
        )

    def generate_samples(self, batch):  # 生成样本
        self.model.eval()
        sample_list = [] # reserve all the results
        prompts = batch.get('question') # 获取prompt
        answers = batch.get('answer') # 获取answer

        for i in range(len(prompts)):
            prompt_text = prompts[i]
            answer_text = answers[i]

            input_text = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": self.system_prompt}, 
                    {"role": "user", "content": prompt_text}
                ],
                add_generation_prompt = True,
                tokenize = False
            )

            # a. 将prompt复制成N份，这一步是模拟rollout
            prompts_repeated = [input_text] * self.args.num_generations

            # b. 将prompt进行tokenize
            inputs = self.tokenizer(
                prompts_repeated,
                return_tensors = 'pt',
                padding = 'max_length',
                max_length = self.args.max_prompt_length,
                truncation = True,
            ).to(self.args.device)

            prompt_len = inputs['input_ids'].shape[1]

            # c. generate
            with torch.no_grad():
                prompt_response_ids = self.model.generate(
                    **inputs,
                    max_new_tokens = self.args.max_generate_length,
                    pad_token_id = self.tokenizer.pad_token_id,
                    eos_token_id = self.tokenizer.eos_token_id,
                    do_sample = True,
                    temperature = 1.0,
                    top_p = 0.95,
                )

            # d. 将response进行tokenize
            response_ids = prompt_response_ids[:, prompt_len:]
            attention_mask = (prompt_response_ids != self.tokenizer.pad_token_id).long() 
            # attention_mask指的是prompt + response长度的mask

            # e. 将response进行tokenize
            is_not_pad = response_ids != self.tokenizer.pad_token_id
            is_not_eos = response_ids != self.tokenizer.eos_token_id
            action_mask = (is_not_pad & is_not_eos).long() # action_mask是把模型的输出的response的special token都去掉之后得到的mask

            sample_list.append(
                Samples(
                    prompt_response_ids=prompt_response_ids,
                    response_ids=response_ids,
                    prompt=prompt_text,
                    answer=answer_text,
                    attention_mask=attention_mask,
                    action_mask=action_mask,
                )
            )
        return sample_list
        # sample_list的形状是由batch_size, num_generations组成的。
        # 比如现在batch_size是2，num_generations是4，prompt_response_ids的长度是300，response的长度是100
        # 那么sample_list里面就有两个sample，其中每个sample有4个rollout结果
    

    def generate_experience(self, batch):
        self.model.eval()
        samples_list = self.generate_samples(batch)
        batch_prompt_reponse_list = []
        batch_attention_mask = []
        batch_action_mask = []
        batch_advantage = []
        batch_old_log_probs = []
        batch_ref_log_probs = []
        all_step_rewards = []
        all_step_named_rewards = []
        all_step_responses = []  # 新增

        for samples in samples_list:
            prompt_response_ids = samples.prompt_response_ids.to(self.args.device)
            response_ids = samples.response_ids.to(self.args.device)
            attention_mask = samples.attention_mask.to(self.args.device)
            action_mask = samples.action_mask.to(self.args.device)
            response_texts = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
            all_step_responses.append(response_texts)  # 新增
            num_responses = len(response_texts)
            prompt_texts = [samples.prompt] * num_responses
            answer_texts = [samples.answer] * num_responses

            rewards = torch.zeros(num_responses, device=self.args.device)
            named_rewards = {}  # 这里主要是为了把reward_func的名字给打印出来
            ## TODO: 把这个reward_func打印的部分给加在utils函数里面
            for reward_func in self.reward_funcs:
                reward_values = reward_func(
                    prompts=prompt_texts,
                    responses=response_texts,
                    answers=answer_texts,
                )
                if hasattr(reward_func, '__name__'):
                    func_name = reward_func.__name__
                elif hasattr(reward_func, '__class__'):
                    func_name = reward_func.__class__.__name__
                else:
                    func_name = str(reward_func)
                named_rewards[func_name] = [float(x) for x in reward_values]
                rewards += torch.tensor(reward_values, device=self.args.device)
                
            all_step_rewards.append(rewards.detach().cpu().tolist())
            all_step_named_rewards.append(named_rewards)

            mean_rewards = rewards.mean()
            std_rewards = rewards.std()
            advantages = (rewards - mean_rewards) / (std_rewards + 1e-8) # + 1e-8是为了防止分母为0

            old_action_log_probs = self._get_action_log_probs(
                self.model,
                prompt_response_ids,
                attention_mask,
                action_mask,
            )
            if self.args.beta != 0.0 and self.ref_model is not None:
                ref_action_log_probs = self._get_action_log_probs(
                    self.ref_model,
                    prompt_response_ids,
                    attention_mask,
                    action_mask,
                )
                batch_ref_log_probs.append(ref_action_log_probs)
            batch_old_log_probs.append(old_action_log_probs)
            batch_prompt_reponse_list.append(prompt_response_ids)
            batch_attention_mask.append(attention_mask)
            batch_action_mask.append(action_mask)
            batch_advantage.append(advantages)

        experience_dict = {
            'prompt_response_ids': torch.cat(batch_prompt_reponse_list, dim=0),
            'attention_mask': torch.cat(batch_attention_mask, dim=0),
            'action_mask': torch.cat(batch_action_mask, dim=0),
            'advantages': torch.cat(batch_advantage, dim=0),
            'old_log_probs': torch.cat(batch_old_log_probs, dim=0),
            'step_rewards': all_step_rewards, # 存储每一个rollout的reward
            'step_named_rewards': all_step_named_rewards,  # 存储每一个rollout的reward_func的名字和对应的reward
            'step_responses': all_step_responses,  # 存储每一个rollout对应的response
        }

        if self.args.beta != 0.0 and batch_ref_log_probs:
            experience_dict['ref_log_probs'] = torch.cat(batch_ref_log_probs, dim=0)
        return experience_dict
    
    def _get_action_log_probs(
            self,
            model: PreTrainedModel,
            prompt_response_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            response_mask: torch.Tensor,
    ):
        '''
        计算给定模型下，一个序列中 response 部分的 token 对数概率
        '''

        outputs = model(
            input_ids = prompt_response_ids,
            attention_mask = attention_mask,
        )

        logits = outputs.logits # logits的形状是(batch_size, sequence_length, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1) # log_probs的形状是(batch_size, sequence_length, vocab_size)
        
        # 对齐并采集目标概率
        log_probs_for_labels = torch.gather(
            log_probs[:, :-1, :],
            dim = -1,
            index = prompt_response_ids[:, 1:].unsqueeze(-1),
        ).squeeze(-1) 
        # 这里torch.gather的作用是：在log_probs[:, :-1, :]的最后一维（vocab_size）上
        # 根据prompt_response_ids[:, 1:]的token_id，从log_probs[:, :-1, :]中采集对应token_id的概率
        # 最终的结果形状是(batch_size, sequence_length - 1, 1)
        # 在使用了.squeeze(-1)之后，形状变成了(batch_size, sequence_length - 1)

        num_actions = response_mask.shape[1]
        action_log_probs = log_probs_for_labels[:, -num_actions:]

        return action_log_probs
    
    def _compute_loss(self, experience):
        prompt_response_ids = experience['prompt_response_ids']
        attention_mask = experience['attention_mask']
        action_mask = experience['action_mask']
        advantages = experience['advantages']
        old_action_log_probs = experience['old_log_probs']

        # 1. 获取当前策略的 log_probs
        current_action_log_probs = self._get_action_log_probs(
            self.model,
            prompt_response_ids,
            attention_mask,
            action_mask,
        )

        # 2. 计算重要性采样的比率
        # 在grpo原来的公式中是将现有策略的probs和旧策略的probs直接相除
        # 但是如果分母的概率是0的话很有可能出现问题，所以这里使用log_ratio来避免这个问题
        # 我们在对数空间里有对数的减法代替除法 - > log(p_new / p_old) = log(p_new) - log(p_old)
        # 然后我们再使用exp来将log_ratio转换回原来的概率比率
    
        log_ratio = current_action_log_probs - old_action_log_probs
        ratio = torch.exp(log_ratio) # 这个比例实际上在说明新策略有多大可能性会重复旧策略的行为
        
        # 3. 计算优势函数
        # 这里advantage_expand的形状是(batch_size, sequence_length, 1)
        # advantage
        advantages_expand = advantages.unsqueeze(-1)

        # 计算不同情况的损失函数
        loss1 = ratio * advantages_expand
        loss2 = torch.clamp(
            ratio,
            1.0 - self.args.low_clip_eps,
            1.0 + self.args.high_clip_eps,
        ) * advantages_expand

        policy_loss = -torch.min(loss1, loss2) 
        # 实际上在rl里面，我觉得不能叫做loss
        # 我们实际上就是在最大化选择最优路径的概率 - > 岂不是很容易就很过拟合
        policy_loss = policy_loss * action_mask

        # 4. 计算KL散度损失
        if self.args.beta != 0.0 and 'ref_log_probs' in experience:
            ref_action_log_probs = experience['ref_log_probs']
            log_kl_ratio = ref_action_log_probs - current_action_log_probs
            kl_penalty = log_kl_ratio.exp() - 1 - log_kl_ratio

            policy_loss += self.args.beta * (kl_penalty * action_mask)

        loss = (policy_loss.sum(dim=1) / action_mask.sum(dim=1)).mean()
        return loss
    
    def train_step(self, batch, global_step=None, rollout_log_file=None):
        '''
        单步训练
        '''
        experience = self.generate_experience(batch)
        loss = self._compute_loss(experience)
        self.accelerator.backward(loss)
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # 打印当前step、学习率等
        # TODO: 把这个函数放到utils里面
        if global_step is not None:
            lr = self.optimizer.param_groups[0]['lr']
            for i, named_rewards in enumerate(experience['step_named_rewards']):
                reward_str = ' '.join([f'{k}:{v}' for k, v in named_rewards.items()])
                print(f'[Step {global_step}] lr={lr:.8f} prompt_idx={i} {reward_str}')
        # 记录rollout到txt
        if rollout_log_file is not None:
            # 确保output目录存在
            os.makedirs(os.path.dirname(rollout_log_file), exist_ok=True)
            with open(rollout_log_file, 'a', encoding='utf-8') as f:
                for i, named_rewards in enumerate(experience['step_named_rewards']):
                    prompt = None
                    responses = None
                    
                    # 从experience中提取prompt_response_ids并解码
                    if 'prompt_response_ids' in experience:
                        # 计算当前prompt的起始和结束索引
                        num_generations = self.args.num_generations
                        start_idx = i * num_generations
                        end_idx = start_idx + num_generations
                        
                        # 提取当前prompt的所有response_ids
                        current_prompt_response_ids = experience['prompt_response_ids'][start_idx:end_idx]
                        
                        # 解码得到完整的prompt（包含system prompt和user question）
                        prompt_texts = self.tokenizer.batch_decode(
                            current_prompt_response_ids, 
                            skip_special_tokens=True
                        )
                        # 取第一个作为prompt（因为同一个prompt的多个生成应该是一样的）
                        prompt = prompt_texts[0] if prompt_texts else None
                    
                    if 'step_responses' in experience:
                        responses = experience['step_responses'][i]
                    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    f.write(f'{now} [Step {global_step}] prompt_idx={i}\n')
                    if prompt is not None:
                        f.write(f'  prompt: {prompt}\n')
                    if responses is not None:
                        for idx, resp in enumerate(responses):
                            f.write(f'  response[{idx}]: {resp}\n')
                    for k, v in named_rewards.items():
                        f.write(f'  {k}: {v}\n')
                    f.write('\n')
        

        return loss.item()
    
    def train(self):
        '''
        训练主循环
        '''
        self.model.train()
        dataloader = DataLoader(
            self.train_dataset,
            batch_size = self.args.batch_size,
            shuffle = True,
        )
        global_step = 0
        total_steps = len(dataloader) * self.args.epoch
        print(f"开始训练，总共{total_steps}步")
        rollout_log_file = os.path.join(self.args.output_dir, 'rollout_log.txt')
        print(f"[INFO] Rollout log file will be saved at: {os.path.abspath(rollout_log_file)}")
        # 清空旧日志
        # 确保output目录存在
        os.makedirs(os.path.dirname(rollout_log_file), exist_ok=True)
        with open(rollout_log_file, 'w', encoding='utf-8') as f:
            f.write('')
        for epoch in range(self.args.epoch):
            epoch_losses = []
            for batch_idx, batch in enumerate(dataloader):
                loss = self.train_step(batch, global_step=global_step, rollout_log_file=rollout_log_file)
                epoch_losses.append(loss)
                global_step += 1
                if global_step % self.args.save_steps == 0:
                    self.save_checkpoint(global_step)
            self.save_checkpoint(global_step)
        self.save_final_model()
        print(f"训练完成")


    def save_checkpoint(self, global_step = None):
        '''
        保存模型
        '''
        checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint_{global_step}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

    def save_final_model(self):
        '''
        保存最终模型
        '''
        checkpoint_dir = os.path.join(self.args.output_dir, "final_model")
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

def main():
    args = DAPOArgs()
    accelerator = Accelerator()

    model_path = '/workspace/minimind/reproduce/weights/qwen2.5-1.5b-instruct'
    tokenizer_path = '/workspace/minimind/reproduce/weights/qwen2.5-1.5b-instruct'

    gsm8k_train_dataset = load_dataset('/workspace/minimind/reproduce/datasets/gsm8k/main', split='train')
    gsm8k_eval_dataset = load_dataset('/workspace/minimind/reproduce/datasets/gsm8k/main', split='test')

    reward_functions = [
        correctness_reward,
        digit_reward,
        hard_format_reward,
        mark_reward,
    ]

    os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
    trainer = DAPOTrainer(
        model = model_path,
        tokenizer = tokenizer_path,
        reward_funcs=reward_functions,
        args = args,
        train_dataset = gsm8k_train_dataset,
        eval_dataset = gsm8k_eval_dataset,
        accelerator = accelerator,
    )
    rollout_log_file = os.path.join(args.output_dir, 'rollout_log.txt')
    print(f"[INFO] Rollout log file saved at: {os.path.abspath(rollout_log_file)}")
    trainer.train()
    # 打印log文件路径


if __name__ == "__main__":
    main()