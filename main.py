import torch
import numpy as np

# Данные - чтение документа и разбивка на читаемые куски
import test_loco
from document_reader import get_text_from_document

# Эмбеддинги
from make_embeddings import create_embeddings, get_embedding

# Вход - звуковой файл
from whisper_speech_recognition import speech_to_text

# Языковая модель
from text_llm import get_reply

# Озвучка текста
from tts_model import get_wav_from_text

# Gradio main thread
import gradio as gr

def respond(text):
    print(f'user question: {text}')
    embedding = get_embedding(text)

    # Поиск cosine simularity
    scores = []
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    for embed in embeddings_raw:
        score = cos(torch.Tensor(embedding), torch.Tensor(embed))
        scores.append(score[0])
    #print(scores)
    sorted_index = np.argsort(scores)[::-1]

    # запрос к LLM
    # vicuna 1.1 as stated on a model card
    llm_prompt = f'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions.\nUSER: '
    # sub-instruction for specific task
    llm_prompt = llm_prompt + 'Answer the question using only provided context below. Answer using your own understanding of the context. Be concise and precise. Answer in Russian.\n'
    # User question
    llm_prompt = llm_prompt + f'Question: {text}\n'
    # Retrieval-Augmented Generation, 4 nearest examples from embedding database + solutions
    llm_prompt = llm_prompt + 'Context: '
    for i in range(0, 4):
        #llm_prompt = llm_prompt + 'Issue: ' + embeddings_text[sorted_index[i]] + '.'
        #llm_prompt = llm_prompt + 'Solution: ' + embeddings_answer[sorted_index[i]] + '.\n'
        llm_prompt = llm_prompt + embeddings_answer[sorted_index[i]]

    # LLM answer
    llm_prompt = str.replace(llm_prompt, '..', '.')
    llm_prompt = llm_prompt + '\nASSISTANT: '

    full_llm_answer = get_reply(llm_prompt)
    print(f'llm answer before prune: {full_llm_answer}\n')

    #llm_prompt = llm_prompt + str(len(llm_prompt))
    full_llm_answer = full_llm_answer[len(llm_prompt):]
    print(f'llm answer after prune: {full_llm_answer}')
    print('------------')
    #full_llm_answer = full_llm_answer + str(len(full_llm_answer))

    solution_text = ''
    answer_string = ''
    for i in range(0, 4):
        solution_text = f'{solution_text}{embeddings_answer[sorted_index[i]]}'
        answer_string = f'{answer_string}{embeddings_text[sorted_index[i]]}'

    return answer_string, solution_text, full_llm_answer, llm_prompt

with gr.Blocks() as demo:
    text = gr.Textbox(label="Запрос")
    problem = gr.Textbox(label="Проблема")
    solution = gr.Textbox(label="Метод устранения", lines=4)
    llm_answer = gr.Textbox(label="Ответ помощника", lines=7)
    raw_prompt = gr.Textbox(label="Raw Prompt", lines=10)

    text.submit(fn=respond, inputs=text, outputs=[problem, solution, llm_answer, raw_prompt])

    send_button = gr.Button(value="Отправить")
    send_button.click(fn=respond, inputs=text, outputs=[problem, solution, llm_answer, raw_prompt])

embeddings_raw, embeddings_text, embeddings_answer = create_embeddings(test_loco.dict, use_cuda=True)

demo.launch(share=True)