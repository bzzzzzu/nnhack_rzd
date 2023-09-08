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
    embedding = get_embedding(text)

    # Поиск cosine simularity
    scores = []
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    for embed in embeddings_raw:
        score = cos(torch.Tensor(embedding), torch.Tensor(embed))
        scores.append(score[0])
    print(scores)
    sorted_index = np.argsort(scores)[::-1]

    solution_text = ''
    answer_string = ''
    for i in range(0, 4):
        solution_text = f'{solution_text}{embeddings_answer[sorted_index[i]]}'
        answer_string = f'{answer_string}{embeddings_text[sorted_index[i]]}'

    return answer_string, text, solution_text

with gr.Blocks() as demo:
    text = gr.Textbox(label="Запрос")
    output = gr.Textbox(label="Проблема")
    solution = gr.Textbox(label="Метод устранения", lines=4)
    raw_prompt = gr.Textbox(label="Raw Prompt", lines=10)

    text.submit(fn=respond, inputs=text, outputs=[output, raw_prompt, solution])

    send_button = gr.Button(value="Отправить")
    send_button.click(fn=respond, inputs=text, outputs=[output, raw_prompt, solution])

embeddings_raw, embeddings_text, embeddings_answer = create_embeddings(test_loco.dict, use_cuda=True)

demo.launch()