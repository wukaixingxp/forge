# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
)
from reward_func import *
import os


class GSM8KDataset(Dataset):
    def __init__(
        self, data_path, tokenizer, split: str = "train", test_size: int = 100
    ):
        self.tokenizer = tokenizer
        data = load_dataset(data_path)
        self.data = data[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        answer = sample["answer_only"]
        prompt = sample["question_zh-cn"]
        return {"prompt": prompt, "answer": answer}


@dataclass
class Samples:
    prompt_response_ids: torch.Tensor
    response_ids: torch.Tensor
    prompt: Any
    answer: Any
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    response_length: int


class DapoArguments:
    output_dir = "./output"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 0.000001
    save_steps = 100
    epoch = 3
    num_generations = 4  # 组内样本数
    max_prompt_length = 256  # 最大输入长度
    max_generate_length = 128  # 最大输出长度
    reward_weights: List[float] = None  # 奖励的权重（多个奖励函数）
    beta = 0.0  # KL散度的系数，为0则忽略KL散度，即不使用参考模型
    clip_eps_high = 0.28
    clip_eps_low = 0.2
    gradient_accumulation_steps = 2  # 梯度累加
    num_iterations = 1  # 采样一次样本训练模型轮数
    batch_size = 1


class DapoTrainer:
    def __init__(
        self,
        model=None,
        reward_funcs: Union[List[str], List[Callable]] = None,
        args=None,
        train_dataset: Optional[Union[Dataset]] = None,
        eval_dataset: Optional[Union[Dataset]] = None,
        tokenizer=None,
        reward_tokenizers=None,
    ):

        self.args = args
        # 加载模型
        if isinstance(model, str):
            model = AutoModelForCausalLM.from_pretrained(model)
        self.model = model.to(self.args.device)

        # 是否使用参考模型
        self.ref_model = None
        if self.args.beta != 0.0:
            self.ref_model = deepcopy(model)
            self.ref_model.eval()

        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        self.tokenizer = self.get_tokenizer(tokenizer)

        if isinstance(reward_funcs, str):
            reward_funcs = [reward_funcs]

        for i, reward_func in enumerate(reward_funcs):
            # 如果奖励函数为字符串，表示使用的是奖励模型，则加载模型
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1
                ).to(self.args.device)

        self.reward_funcs = reward_funcs

        if reward_tokenizers is None:
            reward_tokenizers = [None] * len(reward_funcs)

        elif isinstance(reward_tokenizers, str):
            reward_tokenizers = [reward_tokenizers]

        else:
            if len(reward_tokenizers) != len(reward_funcs):
                raise ValueError(
                    "Length of reward_tokenizers must be equal to the number of reward_funcs."
                )

        for i, (reward_tokenizer, reward_func) in enumerate(
            zip(reward_tokenizers, reward_funcs)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if reward_tokenizer is None:
                    reward_tokenizer = AutoTokenizer.from_pretrained(
                        reward_func.config._name_or_path
                    )
                if reward_tokenizer.pad_token_id is None:
                    reward_tokenizer.pad_token = reward_tokenizer.eos_token

                reward_func.config.pad_token_id = reward_tokenizer.pad_token_id
                reward_tokenizers[i] = reward_tokenizer
        self.reward_tokenizers = reward_tokenizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        # 缓存已经生成的数据的一个批次的数据，可供模型多次训练迭代，无需重新生成
        self.input_buffer = [None] * self.args.gradient_accumulation_steps

        # 模型更新的次数
        self.update_steps = 0

    def get_tokenizer(self, tokenizer):
        tokenizer.padding_side = "left"
        return tokenizer

    # 生成样本，以组为单位
    def generate_samples(self, inputs):
        samples_list = []
        self.model.eval()
        prompts = [prompt for prompt in inputs["prompt"]]
        answers = [None] * len(prompts)

        if "answer" in inputs:
            answers = [answer for answer in inputs["answer"]]

        max_length = self.args.max_generate_length + self.args.max_prompt_length
        for prompt, answer in zip(prompts, answers):
            # 应用聊天模板，加入系统提示词
            input_text = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                add_generation_prompt=True,
                tokenize=False,
            )

            # 生成一个group的输入数据
            inputs = self.tokenizer(
                [input_text] * self.args.num_generations,
                padding="max_length",
                max_length=self.args.max_prompt_length,
                truncation=True,
                return_tensors="pt",
            )
            prompt_ids = inputs["input_ids"]
            with torch.no_grad():
                prompt_response_ids = self.model.generate(
                    **inputs.to(self.args.device),
                    max_new_tokens=self.args.max_generate_length,
                    temperature=0.9,
                    top_p=1,
                    top_k=50,
                )

            if prompt_response_ids.size(1) >= max_length:
                prompt_response_ids = prompt_response_ids[:, :max_length]
            else:
                prompt_response_ids = torch.cat(
                    [
                        prompt_response_ids,
                        torch.full(
                            (
                                prompt_response_ids.size(0),
                                max_length - prompt_response_ids.size(1),
                            ),
                            fill_value=self.tokenizer.pad_token_id,
                            device=prompt_response_ids.device,
                        ),
                    ],
                    dim=1,
                )

            attention_mask = (prompt_response_ids.ne(self.tokenizer.pad_token_id)).to(
                dtype=torch.long
            )
            response_ids = prompt_response_ids[:, prompt_ids.size(1) :]
            action_mask = (
                response_ids.ne(self.tokenizer.eos_token_id)
                & response_ids.ne(self.tokenizer.pad_token_id)
            ).to(dtype=torch.long)

            # 存储的是一个group的数据
            samples = Samples(
                prompt_response_ids=prompt_response_ids,
                response_ids=response_ids,
                prompt=prompt,
                answer=answer,
                attention_mask=attention_mask,
                action_mask=action_mask,
                num_actions=action_mask.size(1),
                response_length=action_mask.float().sum(dim=-1),
            )
            samples_list.append(samples)

        return samples_list

    # 生成经验(优势、token的概率分布)
    def generate_experiences(self, inputs):
        self.model.eval()
        samples_list = self.generate_samples(inputs)

        batch_prompt_response_ids = []
        batch_attention_mask = []
        batch_action_mask = []
        batch_advantages = []
        batch_old_action_log_probs = []
        batch_ref_action_log_probs = []

        for samples in samples_list:
            prompt_response_ids = (
                samples.prompt_response_ids
            )  # shape: (num_generations, seq_len)
            response_ids = samples.response_ids  # shape: (num_generations, seq_len)
            answer = samples.answer
            attention_mask = samples.attention_mask  # shape: (num_generations, seq_len)
            action_mask = samples.action_mask  # shape: (num_generations, seq_len)
            num_actions = samples.num_actions
            prompt = samples.prompt

            with torch.no_grad():
                # 存储各个奖励函数在一个group内各个响应的奖励
                rewards_per_func = torch.zeros(
                    len(self.reward_funcs),
                    self.args.num_generations,
                    device=self.args.device,
                )

                # 将输出转换成文本
                response_texts = self.tokenizer.batch_decode(
                    response_ids, skip_special_tokens=True
                )
                prompt_texts = [prompt] * len(response_texts)
                prompt_response_texts = [
                    prompt + response
                    for prompt, response in zip(prompt_texts, response_texts)
                ]

                for i, (reward_func, reward_tokenizer) in enumerate(
                    zip(self.reward_funcs, self.reward_tokenizers)
                ):
                    if isinstance(reward_func, PreTrainedModel):
                        with torch.inference_mode():
                            reward_model_inputs = reward_tokenizer(
                                prompt_response_texts, return_tensors="pt", padding=True
                            )
                            rewards_per_func[i] = reward_func(
                                **reward_model_inputs.to(self.args.device)
                            ).logits.squeeze(-1)

                    else:
                        answers = [answer] * len(prompt_texts)
                        output_reward_func = reward_func(
                            prompts=prompt_texts,
                            responses=response_texts,
                            answers=answers,
                        )
                        output_reward_func = [
                            reward if reward is not None else torch.nan
                            for reward in output_reward_func
                        ]
                        rewards_per_func[i] = torch.tensor(
                            output_reward_func,
                            dtype=torch.float32,
                            device=self.args.device,
                        )

                # rewards_per_func: [num_funcs, num_generations]
                if not self.args.reward_weights:
                    self.args.reward_weights = [1.0] * len(self.reward_funcs)
                if len(self.args.reward_weights) != len(self.reward_funcs):
                    raise ValueError(
                        "The number of reward weights must be equal to the number of reward functions."
                    )
                # 乘以各个奖励函数的权重
                rewards = rewards_per_func * torch.tensor(
                    self.args.reward_weights,
                    dtype=torch.float32,
                    device=rewards_per_func.device,
                ).unsqueeze(1)
                # rewards: [num_funcs, num_generations]
                rewards = rewards.sum(dim=0)  # shape: [num_generations]

                mean_group_rewards = rewards.mean()
                std_group_rewards = rewards.std()

                # GRPO的优势是句子粒度的，而非token粒度的
                advantages = (rewards - mean_group_rewards) / (
                    std_group_rewards + 1e-8
                )  # shape: [num_generations]
                # 统计优势中非零元素的数量，如果为0，则说明该组中的优势全为0，舍弃该组数据(对更新模型没有用)
                nonzero_num = advantages.count_nonzero().item()
                if nonzero_num == 0:
                    print(f"组内优势为0, 跳过")
                    continue
                print(f"rewards: {rewards}")

                batch_advantages.append(advantages)

                # 计算策略模型输出token的概率
                old_action_log_probs = self.get_action_log_probs(
                    self.model, prompt_response_ids, attention_mask, num_actions
                )
                batch_old_action_log_probs.append(old_action_log_probs)

                # 是否使用参考模型
                if self.ref_model:
                    # 计算参考模型输出token的概率
                    ref_action_log_probs = self.get_action_log_probs(
                        self.ref_model, prompt_response_ids, attention_mask, num_actions
                    )
                    batch_ref_action_log_probs.append(ref_action_log_probs)

                batch_prompt_response_ids.append(prompt_response_ids)
                batch_attention_mask.append(attention_mask)
                batch_action_mask.append(action_mask)

        return {
            "prompt_response_ids": batch_prompt_response_ids,
            "attention_mask": batch_attention_mask,
            "action_mask": batch_action_mask,
            "old_action_log_probs": batch_old_action_log_probs,
            "ref_action_log_probs": (
                batch_ref_action_log_probs if self.ref_model else None
            ),
            "advantages": batch_advantages,
        }

    def compute_loss(self, model, inputs):
        prompt_response_ids = inputs["prompt_response_ids"]
        attention_mask = inputs["attention_mask"]
        action_mask = inputs["action_mask"]
        num_actions = action_mask.size(1)
        action_log_probs = self.get_action_log_probs(
            model, prompt_response_ids, attention_mask, num_actions
        )

        if self.args.beta != 0.0:
            ref_action_log_probs = inputs["ref_action_log_probs"]
            log_ratio = ref_action_log_probs - action_log_probs
            log_ratio = log_ratio * action_mask

            k3 = log_ratio.exp() - 1 - log_ratio

        advantages = inputs["advantages"]

        old_action_log_probs = (
            inputs["old_action_log_probs"]
            if self.args.num_iterations > 1
            else action_log_probs.detach()
        )
        coef_1 = torch.exp(
            action_log_probs - old_action_log_probs
        )  # 重要性采样 shape: [batch_size * num_generations, num_actions]
        coef_2 = torch.clamp(
            coef_1, 1 - self.args.clip_eps_low, 1 + self.args.clip_eps_high
        )
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)  # 一个序列中每个token的优势是一样的
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(
            per_token_loss1, per_token_loss2
        )  # shape: [batch_size * num_generations, num_actions]
        per_token_loss = per_token_loss * action_mask
        if self.args.beta != 0.0:
            per_token_loss = per_token_loss + self.args.beta * k3

        per_token_loss = per_token_loss.view(
            -1, self.args.num_generations, num_actions
        )  #  shape: [batch_size, num_generations, num_actions]
        action_mask = action_mask.view(-1, self.args.num_generations, num_actions)
        loss = per_token_loss.sum(-1).sum(-1) / action_mask.sum(-1).sum(
            -1
        )  # shape: [batch_size]
        loss = loss.mean()

        return loss

    def get_action_log_probs(self, model, input_ids, attention_mask, num_actions):
        # 计算策略模型输出token的概率
        output = model(input_ids, attention_mask=attention_mask)
        logits = output.logits
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        log_probs_labels = log_probs.gather(
            dim=-1, index=input_ids[:, 1:].unsqueeze(-1)
        )
        action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]
        return action_log_probs

    def train_step(self, model, inputs, optimizer, step):
        model.train()
        # scaler = torch.amp.GradScaler()
        # with torch.amp.autocast(device_type='cuda'):
        loss = self.compute_loss(model, inputs)
        loss = loss / self.args.gradient_accumulation_steps
        # loss = scaler.scale(loss)
        loss.backward()
        if (step + 1) % self.args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

            print(
                f"step: {self.update_steps}/{self.global_steps}  dapo_loss: {loss.item():.8f}"
            )
            # 将loss保存到文件中
            loss_file_path = os.path.join(self.args.output_dir, "training_losses.txt")
            os.makedirs(self.args.output_dir, exist_ok=True)

            with open(loss_file_path, "a", encoding="utf-8") as f:
                f.write(f"{self.update_steps},{loss.item():.8f}\n")
        torch.cuda.empty_cache()

    def train(self):
        print(f"\n第 {self.update_steps} 步: === 开始评估模型 ===")
        accuracy = self.evaluate(num_samples=100, batch_size=20)
        print(f"第 {self.update_steps} 步: 模型准确率: {accuracy:.2f}")
        # 将accuracy保存到文件中
        accuracy_file_path = os.path.join(self.args.output_dir, "accuracy_losses.txt")
        os.makedirs(self.args.output_dir, exist_ok=True)
        with open(accuracy_file_path, "a", encoding="utf-8") as f:
            f.write(f"{self.update_steps},{accuracy:.2f}\n")

        self.global_steps = (
            self.args.num_iterations
            * self.args.epoch
            * len(self.train_dataset)
            // (self.args.batch_size * self.args.gradient_accumulation_steps)
        )
        for _ in range(self.args.epoch):
            dataloader = DataLoader(
                self.train_dataset, batch_size=self.args.batch_size, shuffle=True
            )
            buffer = {
                "prompt_response_ids": [],
                "attention_mask": [],
                "action_mask": [],
                "old_action_log_probs": [],
                "ref_action_log_probs": [],
                "advantages": [],
            }
            idx = 0
            for batch in dataloader:
                inputs = self.generate_experiences(batch)
                buffer["prompt_response_ids"] += inputs["prompt_response_ids"]
                buffer["attention_mask"] += inputs["attention_mask"]
                buffer["action_mask"] += inputs["action_mask"]
                buffer["old_action_log_probs"] += inputs["old_action_log_probs"]
                if self.ref_model is not None:
                    buffer["ref_action_log_probs"] += inputs["ref_action_log_probs"]
                else:
                    buffer["ref_action_log_probs"] = None

                buffer["advantages"] += inputs["advantages"]

                # 如果生成的样本batch_size小于设定的batch_size，说明生成数据过程中有舍弃数据，需要继续采样，凑够一个完整的batch_size
                if len(buffer["prompt_response_ids"]) < self.args.batch_size:
                    continue

                if self.ref_model is not None:
                    inputs = {k: v[: self.args.batch_size] for k, v in buffer.items()}
                    inputs = {k: torch.cat(v, dim=0) for k, v in inputs.items()}
                    buffer = {k: v[self.args.batch_size :] for k, v in buffer.items()}
                else:
                    inputs = {
                        k: v[: self.args.batch_size]
                        for k, v in buffer.items()
                        if k != "ref_action_log_probs"
                    }
                    inputs = {k: torch.cat(v, dim=0) for k, v in inputs.items()}
                    inputs["ref_action_log_probs"] = None
                    buffer = {
                        k: v[self.args.batch_size :]
                        for k, v in buffer.items()
                        if k != "ref_action_log_probs"
                    }
                    buffer["ref_action_log_probs"] = None
                self.input_buffer[idx % self.args.gradient_accumulation_steps] = inputs

                if (idx + 1) % self.args.gradient_accumulation_steps == 0:
                    for _ in range(self.args.num_iterations):
                        for step, inputs in enumerate(self.input_buffer):
                            self.train_step(self.model, inputs, self.optimizer, step)
                        self.update_steps += 1
                        if self.update_steps % 10 == 0:
                            print(f"\n第 {self.update_steps} 步: === 开始评估模型 ===")
                            accuracy = self.evaluate(num_samples=100, batch_size=25)
                            print(f"第 {self.update_steps} 步: 模型准确率: {accuracy:.2f}")
                            # 将accuracy保存到文件中
                            accuracy_file_path = os.path.join(
                                self.args.output_dir, "accuracy_losses.txt"
                            )
                            os.makedirs(self.args.output_dir, exist_ok=True)
                            with open(accuracy_file_path, "a", encoding="utf-8") as f:
                                f.write(f"{self.update_steps},{accuracy:.2f}\n")

                        if self.update_steps % self.args.save_steps == 0:
                            self.model.save_pretrained(
                                self.args.output_dir
                                + f"/checkpoint_{self.update_steps}"
                            )
                            self.tokenizer.save_pretrained(
                                self.args.output_dir
                                + f"/checkpoint_{self.update_steps}"
                            )
                idx += 1
                del inputs

    def evaluate(self, num_samples=100, batch_size=20):
        # 限制评估样本数量
        if len(self.eval_dataset) > num_samples:
            indices = torch.randperm(len(self.eval_dataset))[:num_samples].tolist()
            eval_subset = torch.utils.data.Subset(self.eval_dataset, indices)
        else:
            eval_subset = self.eval_dataset
            num_samples = len(self.eval_dataset)

        self.model.eval()
        correct_count = 0
        total_count = 0

        with torch.no_grad():
            dataloader = DataLoader(eval_subset, batch_size=batch_size, shuffle=False)

            for i, batch in enumerate(dataloader):
                # 计算当前批次实际样本数量
                current_batch_size = len(batch["prompt"])
                batch_start_idx = i * batch_size
                batch_end_idx = min(batch_start_idx + current_batch_size, num_samples)

                if batch_start_idx >= num_samples:
                    break

                prompts = batch["prompt"]
                answers = batch["answer"]

                # 批量应用聊天模板
                input_texts = []
                for prompt in prompts:
                    input_text = self.tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                    input_texts.append(input_text)

                # 批量编码输入
                inputs = self.tokenizer(
                    input_texts,
                    padding="max_length",
                    max_length=self.args.max_prompt_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.args.device)

                # 批量生成回复
                prompt_response_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.args.max_generate_length,
                    temperature=0.9,
                    top_p=1,
                    top_k=50,
                )

                # 批量解码回复
                response_texts = []
                for j in range(current_batch_size):
                    response_ids = prompt_response_ids[j, len(inputs[j].ids) :]
                    response_text = self.tokenizer.decode(
                        response_ids, skip_special_tokens=True
                    )
                    response_texts.append(response_text)

                # 批量提取答案
                from reward_func import extract_answer

                for j in range(current_batch_size):
                    prompt = prompts[j]
                    answer = answers[j]
                    response_text = response_texts[j]

                    predicted_answer = extract_answer(response_text)

                    # 标准化比较
                    pred_normalized = str(predicted_answer).strip()
                    true_normalized = str(answer.item()).strip()

                    is_correct = pred_normalized == true_normalized

                    if is_correct:
                        correct_count += 1
                    total_count += 1

                # 如果已经处理了足够多的样本，提前退出
                if total_count >= num_samples:
                    break

        accuracy = correct_count / total_count if total_count > 0 else 0.0

        self.model.train()  # 返回训练模式
        return accuracy

    def save_model(self):
        self.model.save_pretrained(self.args.output_dir)
        self.tokenizer.save_pretrained(self.args.output_dir)


if __name__ == "__main__":
    import os

    SYSTEM_PROMPT = """
按照如下格式回答问题：
<think>
你的思考过程
</think>
<answer>
你的回答
</answer>
"""

    args = DapoArguments()

    # 策略模型
    tokenizer = AutoTokenizer.from_pretrained("./Qwen2.5-1.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("./Qwen2.5-1.5B-Instruct")

    # 加载训练集和测试集
    prompts_dataset = GSM8KDataset("./gsm8k_chinese", tokenizer, split="train")
    test_dataset = GSM8KDataset("./gsm8k_chinese", tokenizer, split="test")

    trainer = DapoTrainer(
        model=model,
        reward_funcs=[
            correctness_reward,
            digit_reward,
            hard_format_reward,
            mark_reward,
        ],
        args=args,
        train_dataset=prompts_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model()
