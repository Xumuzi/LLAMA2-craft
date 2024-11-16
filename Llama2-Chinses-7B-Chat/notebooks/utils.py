from config import *

import gpt_neox
import llama
from conversation import StopOnTokens, get_prompt

import argparse
import json
import torch.nn as nn
import numpy as np
import os
import pandas as pd
import pickle
import random
import shutil
import torch
from itertools import product
from os.path import join
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, classification_report
from transformers import AutoConfig
from transformers import AutoTokenizer, LlamaTokenizer, PreTrainedTokenizer
from transformers import GenerationConfig, StoppingCriteriaList
from typing import Any, List, Tuple, Union


def cuda_available() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def seed_all(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    if torch.cuda.is_available():
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_tokenizer(model_name: str) -> PreTrainedTokenizer:
    model_path = join(MODEL, model_name)
    if model_name.lower().startswith("vicuna"):
        tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)
    elif model_name.lower().startswith("open_llama"):
        tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)
    elif model_name.lower().startswith("baichuan"):
        tokenizer = tokenization_baichuan.BaiChuanTokenizer.from_pretrained(model_path, trust_remote_code=True)
    elif model_name.lower().startswith("llama-2"):
        tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    elif model_name.lower().startswith("llama2"):
        tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"{model_name}: Tokenizer loaded.")
    return tokenizer

def load_model(model_name: str) -> nn.Module:
    model_path = join(MODEL, model_name)
    if model_name.lower().startswith("open_llama"):
        model = llama.LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    elif model_name.lower().startswith("baichuan"):
        model = baichuan.BaiChuanForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    elif model_name.lower().startswith("llama-2"):
        model = llama.LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    elif model_name.lower().startswith("llama2"):
        model = llama.LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    elif model_name.lower().startswith("stablelm-tuned-alpha"):
        model = gpt_neox.GPTNeoXForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
        model.half()
    elif model_name.lower().startswith("dolly-v2"):
        model = gpt_neox.GPTNeoXForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    elif model_name.lower().startswith("vicuna"):
        model = llama.LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    elif model_name.lower().startswith("redpajama"):
        model = gpt_neox.GPTNeoXForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    elif model_name.lower().startswith("chatglm2"):
        raise NotImplementedError
    else:
        raise NotImplementedError
    print(f"{model_name}: Model loaded.")
    return model

def set_generation_config_(decode_method: str) -> GenerationConfig:
    """
    We refer to the following link for more details:
    https://huggingface.co/docs/transformers/generation_strategies
    """
    if decode_method == "greedy":
        config = GenerationConfig(do_sample=False)
    elif decode_method == "contrastive_search":
        config = GenerationConfig(penalty_alpha=0.6, top_k=4)
    elif decode_method == "multinomial_sampling":
        config = GenerationConfig(do_sample=True)
    elif decode_method == "beam_search":
        config = GenerationConfig(num_beams=5)
    elif decode_method == "beam_search_multinomial_sampling":
        config = GenerationConfig(num_beams=5, do_sample=True)
    else:
        raise NotImplementedError
    return config

def qa(
        question: str,
        model_name: str,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        device: str='cuda',
        max_tokens: int=900,
        temperature: float=0.8,
        hide_prompt: bool=True,
    ) -> str:
    results  = get_prompt(question, MODEL_MAPPING[model_name])
    prompt   = results['prompt']
    if results['stop_ids'] is None and results['stop_str'] is None:
        stop_ids = []
    else:
        if results['stop_ids'] is not None:
            stop_ids = results['stop_ids']
        else:
            stop_ids = tokenizer(results['stop_str']).input_ids[1:] # the first token is `<s>``
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids
    input_ids = input_ids.to(device)
    generation_output = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_tokens,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=StoppingCriteriaList([StopOnTokens(stop_ids=stop_ids)])
    )
    answer = tokenizer.decode(generation_output[0])
    if hide_prompt:
        answer = answer[len(prompt):]
    return answer

def save_json(data: dict, file_path:str, indent: int=4):
    """
    Save data to a JSON file.

    Parameters:
    data (dict): Data to be saved.
    file_path (str): Path where the JSON file will be saved.
    indent (int): Indentation level for pretty-printing. Default is 4.
    """

    def default(o):  # Define a new function to handle numpy numbers
        if isinstance(o, np.int64):
            return int(o)
        if isinstance(o, np.float64):
            return float(o)
        raise TypeError

    with open(file_path, 'w') as f:
        json.dump(data, f, default=default, indent=indent)

def get_layer_names(model: nn.Module, show_in_terminal: bool=False) -> List[str]:
    names = []
    for name, _ in model.named_modules():
        if show_in_terminal:
            print(name)
        names.append(name)
    return names

def align_show_in_terminal(
        *inputs: List[List[Union[int, str]]],
        header: Union[str, List[str]]=None,
        column_width: Union[int, List[int], str]='auto',
        sep: str='',
        truncate: bool=True,
        truncated_size: int=10,
        truncated_pad_token: str="...") -> None:
    '''
    For example:
    --------
    >>> align_show_in_terminal(
    ...     ['Tom', 'Jerry', 'Spike'], 
    ...     [25, 22, 30], 
    ...     ['Engineer', 'Doctor', 'Lawyer'], 
    ...     header=['Name', 'Age', 'Profession'],
    ...     column_width='auto',
    ...     sep=' | ',
    ...     truncate=True,
    ...     truncated_size=10,
    ...     truncated_pad_token="..."
    ... )
    '''
    num_cols = len(inputs)
    inputs = [[str(i) for i in inp] for inp in inputs]
    if header:
        if isinstance(header, str):
            header = [header] * num_cols
        else:
            assert len(header) == num_cols
        inputs = [[header[i]] + inp for i, inp in enumerate(inputs)]
    if isinstance(column_width, list):
        assert num_cols == len(column_width), f"Column size mismatched, {num_cols} != {len(column_width)}."
    elif isinstance(column_width, int):
        column_width = [column_width] * num_cols
    elif isinstance(column_width, str):
        column_width = [max(len(i) for i in inp) + 5 for inp in inputs]
    if truncate:
        inputs = [col[:truncated_size] for col in inputs]
        inputs = [col + [truncated_pad_token] for col in inputs]
    aligned_inputs = []
    for size, inp in zip(column_width, inputs):
        aligned_inputs.append([i + " " * (size - len(i)) for i in inp])
    num_rows = len(aligned_inputs[0])
    for i in range(num_rows):
        print(sep.join([aligned_inputs[j][i] for j in range(num_cols)]))

def save_sk_model(probes: List[Any], path: str): 
    ''' takes in a list of sklearn lr probes and saves them to path '''
    with open(path, 'wb') as f: 
        pickle.dump(probes, f)

def load_probes(path: str) -> List[Any]: 
    ''' loads a list of sklearn lr probes from path '''
    with open(path, 'rb') as f: 
        probes = pickle.load(f)
    return probes

def save_to_txt(texts: np.ndarray, path: str):
    np.savetxt(path, texts, delimiter=',')

def analyze_layer_wise_accs(accs: np.ndarray, show_details: bool=True, show_summary: bool=True) -> dict:
    ''' Present layer-wise and model-wise analysis results. '''
    layer_wise_avgs = np.mean(accs, axis=1).tolist()
    layer_wise_maxs = np.max(accs, axis=1).tolist()
    layer_wise_mins = np.min(accs, axis=1).tolist()
    layer_wise_medians = np.median(accs, axis=1).tolist()
    layer_wise_indices = np.arange(len(layer_wise_avgs)).tolist()
    if show_details:
        align_show_in_terminal(
            layer_wise_indices,
            layer_wise_avgs,
            layer_wise_maxs,
            layer_wise_mins,
            layer_wise_medians,
            header=['layer', 'mean', 'max', 'min', 'median'],
            truncate=False)
        model_wise_avg = np.mean(accs).tolist()
        model_wise_max = np.max(accs).tolist()
        model_wise_min = np.min(accs).tolist()
        model_wise_median = np.median(accs).tolist()
    results = {
        'mean': layer_wise_avgs,
        'max': layer_wise_maxs,
        'min': layer_wise_mins,
        'median': layer_wise_medians
    }
    if show_summary:
        print("=" * 20)
        print("Summary:")
        align_show_in_terminal(
            ['mean', 'max', 'min', 'median'],
            [model_wise_avg, model_wise_max, model_wise_min, model_wise_median],
            truncate=False)
    return results

def load_json(file_path: str):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def load_jsonl(input_path: str) -> List[dict]:
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def jsonl_to_dict(jsonl: List[dict]) -> dict:
    result = {}
    for d in jsonl:
        keys = list(d.keys())
        for k in keys:
            if k in result:
                result[k].append(d[k])
            else:
                result[k] = [d[k]]
    return result

def columns_to_dict(df: pd.DataFrame, left: str, right: str) -> dict:
    """ Select two columns from a dataframe and map them into
        a dictionary.

    Args:
        df (pd.DataFrame): a dataframe.
        left (str): the left column name; will be mapped into the keys of the dictionary.
        right (str): the right column name; will be mapped into the values of the dictionary.
    Returns:
        dict: the transformed dictionary.
    """
    left_col = df[left].tolist()
    right_col = df[right].tolist()
    size = len(left_col)
    return {left_col[i]: right_col[i] for i in range(size)}

def text_normalize_(text: str) -> str:
    text = text.strip()
    text = text.strip('\t')
    text = text.replace('\n', '\\n')
    text = text.replace('\t', '\\t')
    return text

def present_args(args: argparse.Namespace):
    print("=" * 30)
    print("Settings:")
    align_show_in_terminal(
        [i + ":" for i in args.__dict__.keys()],
        args.__dict__.values(),
        truncate=False)
    print("=" * 30)

def mkdir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def deldir(path: str):
    if os.path.exists(path) and os.path.isdir(path):
        try:
            shutil.rmtree(path)
            print(f"Folder '{path}' has been deleted successfully.")
        except OSError as e:
            print(f"Error: {e.filename} - {e.strerror}.")
    else:
        print("The folder does not exist or the path is not a directory.")

def random_choose(seq: np.ndarray, size: int, seed: int=42) -> np.ndarray:
    np.random.seed(seed)
    random_chosed = np.random.choice(seq, size, replace=False)
    return random_chosed

def frequency_matrix(matrix: np.ndarray) -> np.ndarray:
    unique_elements, counts = np.unique(matrix, return_counts=True)
    frequency_matrix = np.zeros_like(matrix, dtype=int)
    for i, unique_element in enumerate(unique_elements):
        frequency_matrix[matrix == unique_element] = counts[i]
    return frequency_matrix

def get_model_layers_and_heads(model_path: str) -> Tuple[int, int]:
    """ Load number of layers and number of heads for a transformer-based
        model from its configuration.

    Args:
        model_path (str): Local path to the model. 

    Returns:
        Tuple[int, int]: Number of layers and number of heads.
    """
    config = AutoConfig.from_pretrained(model_path)
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    return (num_layers, num_heads)

def produce_script(py_name: str, shell_name: str, **kwargs):
    '''
    Example:
    --------
    >>> produce_script(
    ...     "get_activations.py",
    ...     "get_activations.sh",
    ...     model_name=['open_llama_3b', 'open_llama_7b'],
    ...     dataset_name=['lahate', 'toxicn'],
    ...     split=['train', 'test', 'val'],
    ...     method=['all_ct_single_emb', 'concat_ct_embs', 'avg_over_ct_embs', 'baseline'])
    '''
    kws = []
    args = []
    for k, v in kwargs.items():
        kws.append(k)
        args.append(v)
    arg_combinations = list(product(*args))
    script_lines = ['cd ..', 'cd src']
    for ac in arg_combinations:
        arg_string = ' '.join([f"--{kw} {arg}" for kw, arg in zip(kws, ac)])
        line = f"python {py_name} {arg_string}"
        script_lines.append(line)
    script_lines = [l + '\n' for l in script_lines]
    with open(os.path.join(SCRIPTS, shell_name), 'w') as writer:
        writer.writelines(script_lines)

def init_weights(m: nn.Module):
    if type(m) == nn.Linear:
        nn.init.uniform_(m.weight)
        m.bias.data.fill_(0.0)

def eval(labels, preds):
    acc = accuracy_score(labels, preds)
    rec = recall_score(labels, preds)
    pre = precision_score(labels, preds)
    f1  = f1_score(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    print(classification_report(labels, preds))
    results = {
        'accuracy': acc,
        'recall': rec,
        'precision': pre,
        'f1-score': f1,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp
    }
    return results

def save_results(folder_path, config: argparse.Namespace, results: dict, model: Union[Any, None]):
    config_dict = config.__dict__
    mkdir(folder_path)
    save_json(config_dict, join(folder_path, 'config.json'))
    save_json(results, join(folder_path, 'results.json'))
    if model is not None:
        if isinstance(model, nn.Module):
            pass
            #raise NotImplementedError
        else:
            save_sk_model(model, join(folder_path, 'model.pkl'))


produce_script(
    "emb_exp.py",
    "emb_exp.sh",
    model_name=['Llama2-Chinese-7b-Chat'],
    dataset_name=['lahate', 'toxicn'],
    method=['all_ct_single_emb', 'concat_ct_embs', 'avg_over_ct_embs', 'baseline'],
    seed=[999])

# produce_script(
#     "get_activations.py",
#     "get_activations.sh",
#     model_name=['Llama2-Chinese-7b-Chat'],
#     dataset_name=['lahate', 'toxicn'],
#     split=['train', 'test', 'val'],
#     method=[ 'concat_ct_embs', 'avg_over_ct_embs'])

# produce_script(
#     "get_activations.py",
#     "get_activations.sh",
#     model_name=['stablelm-tuned-alpha-7b', 'dolly-v2-7b'],
#     dataset_name=['lahate'],
#     split=['train', 'test', 'val'],
#     method=['all_ct_single_emb', 'concat_ct_embs', 'avg_over_ct_embs', 'baseline'])