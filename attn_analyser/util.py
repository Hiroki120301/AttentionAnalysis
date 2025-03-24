import argparse
import os
import math
import random
import numpy as np
from tqdm import tqdm

import torch
import matplotlib.pyplot as plt
import seaborn as sns

def plot_heatmap(attention_scores, model_id, plot_figs_per_head, save_fig_path, tokens_list=None, ignore_first_token=False, num_figs_per_row=4):
    """
    attention_scores: a list containing 32 layers' attention scores, each is a tensor with shape [1, num_heads, seq_len, seq_len]
    tokens_list: act as xticks and yticks of the figure, eg. ['<s>', 'Hi', ',', 'how', 'are', 'you', '?']
    """
    save_fig_path_model = os.path.join(save_fig_path, model_id) # the model's results are saved under this dir 
    os.makedirs(save_fig_path_model, exist_ok=True)

    if ignore_first_token:
        attention_scores = [attention_scores[i][:, :, 1: , 1: ] for i in range(len(attention_scores))]
        tokens_list = tokens_list[1: ]

    # a figure for all
    print(f'plotting a figure for all layers ...')
    num_heads = len(attention_scores)
    num_rows = math.ceil(num_heads / num_figs_per_row) 
    fig, axes = plt.subplots(num_rows, num_figs_per_row, figsize=(len(tokens_list) * 2, 0.5 * num_rows * len(tokens_list)))
    for layer_idx in tqdm(range(len(attention_scores))):
        row, col = layer_idx // num_figs_per_row, layer_idx % num_figs_per_row
        avg_attention_scores = attention_scores[layer_idx][0].mean(dim=0)    # [ seq_len, seq_len]
        mask = torch.triu(torch.ones_like(avg_attention_scores, dtype=torch.bool), diagonal=1)
        sns.heatmap(avg_attention_scores.numpy(), mask=mask.numpy(), cmap='RdBu_r', square=True, xticklabels=tokens_list, yticklabels=tokens_list, ax=axes[row, col])
        axes[row, col].set_title(f'layer {layer_idx}')

    plt.suptitle(f'all layers avg') 
    plt.savefig(os.path.join(save_fig_path_model, f'all_layers_avg.jpg'))
    plt.close()   

    if not plot_figs_per_head:
        return

    # a figure for each layer
    for layer_idx in range(len(attention_scores)):
        print(f'plotting layer {layer_idx} ...')
        num_heads = attention_scores[layer_idx].shape[1]
        num_rows = math.ceil(num_heads / num_figs_per_row)
        fig, axes = plt.subplots(num_rows, num_figs_per_row, figsize=(len(tokens_list) * 2, 0.5 * num_rows * len(tokens_list)))
        for head_idx in tqdm(range(num_heads)):
            row, col = head_idx // num_figs_per_row, head_idx % num_figs_per_row
            head_attention_scores = attention_scores[layer_idx][0][head_idx]    # [seq_len, seq_len]
            mask = torch.triu(torch.ones_like(head_attention_scores, dtype=torch.bool), diagonal=1)
            sns.heatmap(head_attention_scores.numpy(), mask=mask.numpy(), cmap='RdBu_r', square=True, xticklabels=tokens_list, yticklabels=tokens_list, ax=axes[row, col])
            axes[row, col].set_title(f'head {head_idx}')

        plt.suptitle(f'layer_{layer_idx}') 
        plt.savefig(os.path.join(save_fig_path_model, f'layer_{layer_idx}.jpg'))
        plt.close()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument('--model', type=str, default=None, choices=["llama2-7b-chat-4k", "longchat-v1.5-7b-32k", "xgen-7b-8k",
                                                                    "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k",
                                                                    "chatglm3-6b-32k", "vicuna-v1.5-7b-16k", "LLaMA-2-7B-32K",
                                                                    "LWM-Text-Chat-1M"])
    return parser.parse_args(args)


# This is the customized building prompt for chat models
def build_chat(prompt, model_name, noquery=False):
    if noquery:
        if "LWM" in model_name:
            prompt = f"You are a helpful assistant. USER: {prompt}"
        elif "longchat" in model_name:
            from fastchat.model import get_conversation_template
            conv = get_conversation_template("vicuna")
            conv.append_message(conv.roles[0], prompt)
            prompt = conv.get_prompt()

    else:
        if "LWM" in model_name:
            prompt = f"You are a helpful assistant. USER: {prompt} ASSISTANT: "
        elif "longchat" in model_name:
            from fastchat.model import get_conversation_template
            conv = get_conversation_template("vicuna")
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
    return prompt


def truncate_fn(prompt, prompt_noquery, tokenizer, max_length, dataset, device, model_name=' '):
    # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
    tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    tokenized_prompt_noquery = tokenizer(prompt_noquery, truncation=False, return_tensors="pt").input_ids[0]

    # truncate based on length of prompt with query
    len_tokenized_prompt = len(tokenized_prompt)
    if len(tokenized_prompt) > max_length:
        half = int(max_length/2)

        # compute num tokens removed and subtract from sp_len
        tokens_removed = len(tokenized_prompt) - 2*half

        prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
    else:
        tokens_removed = 0

    # incorporate chat template for shared prefix length
    if dataset not in ["trec", "triviaqa", "samsum", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
        prompt = build_chat(prompt, model_name)
        prompt_noquery = build_chat(prompt_noquery, model_name, noquery=True)

    # compute shared prefix length
    input_ids_prompt_only = tokenizer(prompt_noquery, truncation=False, return_tensors="pt").input_ids.to(device)
    shared_prefix_length = input_ids_prompt_only.shape[1]

    return prompt, shared_prefix_length - tokens_removed
