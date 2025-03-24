import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig, AutoConfig, AutoModelForCausalLM
from attn_analyser.util import *
import torch.multiprocessing as mp

def analyze(rank, world_size, data, max_length, prompt_format, prompt_only_format, dataset, device, model_path, model_name):
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(rank)

    # Load the model and tokenizer
    config = AutoConfig.from_pretrained(model_path)
    config._attn_implementation = "eager"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16)
    model = model.to(device)
    model = model.eval()

    torch.cuda.empty_cache()

    for i, d in enumerate(tqdm(data)):
        if i == 3 or i == 10:
            continue
        prompt = prompt_format.format(**d)
        prompt_only = prompt_only_format.format(**d)

        prompt, _ = truncate_fn(prompt, prompt_only, tokenizer, max_length, dataset, device, model_name)
        input_ids = tokenizer(prompt, truncation=True, return_tensors="pt").input_ids.to(device)

        # Clear cache before inference
        torch.cuda.empty_cache()
        
        # Perform inference with no_grad
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=False)
        print(outputs.attentions)
        # Free memory and clear variables
        del input_ids, outputs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    seed_everything(42)
    mp.set_start_method('spawn', force=True)
    args = parse_args()
    world_size = torch.cuda.device_count()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model2path = json.load(open("config/model2path.json", "r"))
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))

    max_length = model2maxlen[args.model]
    model_path = model2path[args.model]

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.set_default_dtype(torch.bfloat16)
    dataset = args.task
    data = load_dataset('THUDM/LongBench', dataset, split='test[:10%]')
    prompt_format = dataset2prompt[dataset]
    prompt_only_format = dataset2prompt[dataset + '_prompt']
    max_gen = dataset2maxlen[dataset]

    data_all = [data_sample for data_sample in data]

    for i in range(len(data_all)):
        data_all[i]['different_prefix_index'] = i

    data_subsets = [data_all[i::world_size] for i in range(world_size)]

    processes = []
    for rank in range(world_size):
        p = mp.Process(target=analyze, args=(rank, world_size, data_subsets[rank], max_length, prompt_format, prompt_only_format, dataset, device, model_path, args.model))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
