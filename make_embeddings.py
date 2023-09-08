import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from torch import Tensor
import os

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def create_embeddings(string_dict, use_cuda=True):
    embeddings_raw_name = 'embeddings.npy'
    embeddings_text_name = 'embeddings.txt'
    embeddings_answer_name = 'embeddings_answer.txt'

    if not os.path.exists(embeddings_raw_name):
        tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-base')
        if use_cuda:
            model = AutoModel.from_pretrained('intfloat/multilingual-e5-base').to('cuda')
        else:
            model = AutoModel.from_pretrained('intfloat/multilingual-e5-base')

        embeddings = []

        # пока складываем основную причину с подпричиной
        string_list = []
        answer_list = []
        for key in string_dict.keys():
            for subkey in string_dict[key]:
                temp_str = f'{key}. {subkey}.'
                string_list.append(str.replace(temp_str, '..', '.'))
                answer_list.append(string_dict[key][subkey])

        # медленно (лучше батчами), но просто и исполняется один раз
        for line in string_list:
            if use_cuda:
                batch_dict = tokenizer(line, max_length=512, padding=True, truncation=True, return_tensors='pt').to('cuda')
            else:
                batch_dict = tokenizer(line, max_length=512, padding=True, truncation=True, return_tensors='pt')
            outputs = model(**batch_dict)
            embedding = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embeddings.append(embedding[0])
        embeddings = torch.stack(embeddings).cpu().detach().numpy()

        np.save(embeddings_raw_name, embeddings)
        with open(embeddings_text_name, 'w', encoding='utf-8') as f:
            for line in string_list:
                f.write(line + '\n')
        with open(embeddings_answer_name, 'w', encoding='utf-8') as f:
            for line in answer_list:
                f.write(line + '\n')

        embeddings_raw = embeddings
        embeddings_text = string_list
        embeddings_answer = answer_list
    else:
        embeddings_raw = np.load(embeddings_raw_name)
        with open(embeddings_text_name, 'r', encoding='utf-8') as f:
            embeddings_text = f.readlines()
        with open(embeddings_answer_name, 'r', encoding='utf-8') as f:
            embeddings_answer = f.readlines()

    return embeddings_raw, embeddings_text, embeddings_answer

# Достаточно быстро работает и на процессоре для одиночных вводов текста
# Желательно держать модель в памяти, ну пока сойдет
def get_embedding(text):
    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-base')
    model = AutoModel.from_pretrained('intfloat/multilingual-e5-base')
    batch_dict = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**batch_dict)
    embedding = average_pool(outputs.last_hidden_state, batch_dict['attention_mask']).cpu().detach().numpy()
    return embedding

