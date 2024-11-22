import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class GSM8KDataset(Dataset):
    def __init__(self, path):
        # Load data and preprocess
        splits = {'train': 'main/train-00000-of-00001.parquet', 'test': 'main/test-00000-of-00001.parquet'}
        self.data = pd.read_parquet("hf://datasets/openai/gsm8k/" + splits["train"])  # Load GSM8K from 'pandas'
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

class MMEDataset(Dataset):
    def __init__(self, path):
        # Load data and preprocess
        self.data = ... # Load MME from 'path'
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
    

from transformers import AutoModelForCausalLM, AutoTokenizer

# Load Qwen2-1.5b (LLM)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5b")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-1.5b").cuda()

def run_inference_llm(dataset):
    results = []
    model.eval()
    for data in DataLoader(dataset, batch_size=1):
        input_ids = tokenizer(data["input"], return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            output = model.generate(input_ids)
        results.append(tokenizer.decode(output[0]))
    return results

def run_inference_vlm(dataset):
    # VLM inference logic for MiniGPT-4
    results = []
    model.eval()
    for data in DataLoader(dataset, batch_size=1):
        with torch.no_grad():
            # Assumed processing for VLM
            output = model(data)
        results.append(output)
    return results


def create_results_table(results, model_name):
    table = pd.DataFrame(results, columns=[f"{model_name} Output"])
    print(table)
    return table