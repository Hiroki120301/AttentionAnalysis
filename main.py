import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig
from attn_analyser.util import *

if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    world_size = torch.cuda.device_count()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model2path = json.load(open("config/model2path.json", "r"))
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))

    max_length = model2maxlen[args.model]
    model_path = model2path[args.model]

    config = LlamaConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = LlamaForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16)
    config.return_qkv_states = True
    model.eval()
    model = model.to(device)

    if args.task is not None:
        datasets = [args.task]
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
                    "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", \
                    "lcc", "repobench-p"]
    
    for dataset in datasets:
        data = load_dataset('THUDM/LongBench', dataset, split='test')
        prompt_format = dataset2prompt[dataset]
        prompt_only_format = dataset2prompt[dataset + '_prompt']

        for i, d in enumerate(tqdm(data)):
            prompt = prompt_format.format(**d)
            prompt_only = prompt_only_format.format(**d)
            # get truncated input prompt
            prompt, _ = truncate_fn(prompt, prompt_only, tokenizer, max_length, dataset, device, args.model)
            input_ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids.to(device)

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_ids, output_attentions=True)
            attention_scores = outputs.attentions
