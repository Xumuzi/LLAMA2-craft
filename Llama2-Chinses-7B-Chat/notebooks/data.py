from config import *
from conversation import get_prompt
from utils import *

import pandas as pd
from os.path import join, exists
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import List, Union, Tuple

def load_lahate(split='val', seed=42):
    ''' Load latent-hatred dataset. '''
    assert split in ['train', 'test', 'val'], f"{split} not existing."
    def preprocess(src_path):
        src_data = pd.read_csv(src_path, header=0, sep='\t')
        src_data = src_data[~(src_data['class'] == 'explicit_hate')]
        src_data['label'] = src_data['class'].apply(lambda x: x == 'implicit_hate')
        src_data = src_data.rename(columns={'post': 'sentence'})
        return src_data
    data_path = join(LATENT_HATRED_FOLDER, "all.tsv")
    if not exists(data_path):
        data = preprocess(join(LATENT_HATRED_FOLDER, "implicit_hate_v1_stg1_posts.tsv"))
        data.to_csv(data_path, sep='\t', index=False)
        train, val_test = train_test_split(data, test_size=0.2, train_size=0.8, random_state=seed, shuffle=True)
        val, test = train_test_split(val_test, test_size=0.5, train_size=0.5, random_state=seed, shuffle=True)
        train.to_csv(join(LATENT_HATRED_FOLDER, "train.tsv"), sep='\t', index=False)
        val.to_csv(join(LATENT_HATRED_FOLDER, "val.tsv"), sep='\t', index=False)
        test.to_csv(join(LATENT_HATRED_FOLDER, "test.tsv"), sep='\t', index=False)
    return pd.read_csv(join(LATENT_HATRED_FOLDER, f"{split}.tsv"), sep='\t')

def load_toxicn(split='val', seed=42):
    assert split in ['train', 'test', 'val'], f"{split} not existing."
    def preprocess(src_path):
        src_data = pd.read_excel(src_path, header=0)
        src_data = src_data[~(src_data['toxic_type'] == 1)]
        src_data['label'] = src_data['toxic_type'].apply(lambda x: x == 2)
        src_data = src_data.rename(columns={'content': 'sentence'})
        src_data = src_data[['sentence', 'toxic_type', 'label']]
        return src_data
    data_path = join(TOXICN_FOLDER, "all.tsv")
    if not exists(data_path):
        data = preprocess(join(TOXICN_FOLDER, "ToxiCN_1.0.xlsx"))
        data.to_csv(data_path, sep='\t', index=False)
        train, val_test = train_test_split(data, test_size=0.2, train_size=0.8, random_state=seed, shuffle=True)
        val, test = train_test_split(val_test, test_size=0.5, train_size=0.5, random_state=seed, shuffle=True)
        train.to_csv(join(TOXICN_FOLDER, "train.tsv"), sep='\t', index=False)
        val.to_csv(join(TOXICN_FOLDER, "val.tsv"), sep='\t', index=False)
        test.to_csv(join(TOXICN_FOLDER, "test.tsv"), sep='\t', index=False)
    return pd.read_csv(join(TOXICN_FOLDER, f"{split}.tsv"), sep='\t')

def load_data(dataset_name: str, split: str) -> Tuple[pd.DataFrame, str]:
    assert split in ['train', 'test', 'val'], f"{split} not existing."
    if dataset_name == 'toxicn':
        folder = TOXICN_FOLDER
        lang = 'zh'
    elif dataset_name == 'lahate':
        folder = LATENT_HATRED_FOLDER
        lang = 'en'
    else:
        raise NotImplementedError
    data_path = join(folder, f"{split}.tsv")
    data = pd.read_csv(data_path, sep='\t', header=0)
    return data, lang

class LlmDataset(Dataset):
    def __init__(
            self,
            data: pd.DataFrame,
            model_name: str,
            lang: str,
            method: str,
            tokenizer: Union[None, PreTrainedTokenizer]=None):
        self.tokenizer: PreTrainedTokenizer = tokenizer
        if tokenizer is None:
            self.tokenizer = load_tokenizer(model_name)
        self.data: pd.DataFrame = data
        self.model_name: str = model_name
        self.lang: str = lang
        self.method: str = method
        self.sentences: List[str] = self.data['sentence'].tolist()
        self.labels: List[bool] = self.data['label'].tolist()

    def __len__(self) -> int:
        return len(self.data)

    def load_codetyped_question(self, question, lang: str, method: str) -> Union[str, List]:
        assert lang in ['en', 'zh'], f"{lang} not found."
        assert method in ['all_ct_single_emb', 'concat_ct_embs', 'avg_over_ct_embs', 'baseline'], f"{method} not found."
        codetypes: list = CODETYPES[lang]
        if method == 'all_ct_single_emb':
            prefix = '\n'.join(codetypes)
            prompt = prefix + question
        elif method == 'concat_ct_embs' or method == 'avg_over_ct_embs':
            prompt = []
            for ct in codetypes:
                prompt.append(ct + question)
        elif method == 'baseline':
            prompt = question
        else:
            raise NotImplementedError
        return prompt

    def __getitem__(self, index: int) -> dict:
        question = self.sentences[index]
        label = self.labels[index]
        processed_question = get_prompt(question, MODEL_MAPPING[self.model_name])['prompt']
        processed_question_ids = self.tokenizer(processed_question, return_tensors='pt').input_ids
        prompt = self.load_codetyped_question(processed_question, self.lang, self.method)
        if isinstance(prompt, list):
            prompt_ids = [self.tokenizer(p, return_tensors='pt').input_ids for p in prompt]
        else:
            prompt_ids = self.tokenizer(prompt, return_tensors='pt').input_ids
        return {
            'prompt_ids'            : prompt_ids,
            'prompt'                : prompt,
            'label'                 : label,
            'processed_question'    : processed_question, # processed q
            'processed_question_ids': processed_question_ids,
            'original_question'     : question}

    
    # def __getitem__(self, index: int) -> dict:
    #     question = self.sentences[index]
    #     label = self.labels[index]
    #     prompt_question = self.load_codetyped_question(question, self.lang, self.method)
    #     processed_question = get_prompt(prompt_question, MODEL_MAPPING[self.model_name])['prompt']
    #     processed_question_ids = self.tokenizer(processed_question, return_tensors='pt').input_ids

    #     if isinstance(processed_question, list):
    #         prompt_ids = [self.tokenizer(p, return_tensors='pt').input_ids for p in processed_question]
    #     else:
    #         prompt_ids = self.tokenizer(processed_question, return_tensors='pt').input_ids
    #     return {
    #         'prompt_ids'            : prompt_ids,
    #         'label'                 : label,
    #         'processed_question'    : processed_question, # processed q
    #         'processed_question_ids': processed_question_ids,
    #         'original_question'     : question}