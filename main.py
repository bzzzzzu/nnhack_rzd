import torch
import numpy as np
import os
import pandas as pd

use_llm = True
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
global_problem = ""

# Данные - чтение документа и разбивка на читаемые куски
import test_loco

# Эмбеддинги
from make_embeddings import create_embeddings, get_embedding

# Вход - звуковой файл
from whisper_speech_recognition import speech_to_text

if use_llm:
    # Языковая модель
    from text_llm import get_reply

# Озвучка текста
from tts_model import Processor

tts = Processor()

# Gradio main thread
import gradio as gr

types_of_trains = [
    "2М62",
    "2ТЭ10М",
    "2ТЭ10МК",
    "2ТЭ10У",
    "2ТЭ10УК",
    "2ТЭ25А",
    "2ТЭ25КМ",
    "2ТЭ70",
    "2ТЭ116",
    "2ТЭ116УД",
    "2ЭС4К",
    "2ЭС5К",
    "3ЭС5К",
    "2ЭС6",
    "2ЭС7",
    "2ЭС10",
    "Все",
]
selected_train_type = None


def respond(text, to_text=False):
    global global_problem
    print(f"user question: {text}")
    embedding = get_embedding(text)

    # Поиск cosine simularity
    scores = []
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    for embed in embeddings_raw:
        score = cos(torch.Tensor(embedding), torch.Tensor(embed))
        scores.append(score[0])
    sorted_index = np.argsort(scores)[::-1]

    # запрос к LLM
    # vicuna 1.1 as stated on a model card
    llm_prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nUSER: "
    # sub-instruction for specific task
    # llm_prompt = llm_prompt + 'Answer the question or give the solution to the problem using only provided context below. Answer using your own understanding of the context. Be concise and precise. Answer in Russian.\n'
    llm_prompt = (
        llm_prompt
        + "Answer the question or give the solution to the problem. Answer using your own understanding of the context. Be concise and precise. Answer in Russian.\n"
    )
    # User question
    llm_prompt = llm_prompt + f"Запрос пользователя: {text}\n"
    # Retrieval-Augmented Generation, 4 nearest examples from embedding database + solutions
    llm_prompt = llm_prompt + "Context: "
    for i in range(0, 3):
        problem_text = str.replace(embeddings_text[sorted_index[i]], "\n", "")
        solution_text = str.replace(embeddings_answer[sorted_index[i]], "\n", "")
        story_text = f"Проблема: {problem_text}, Решение: {solution_text}\n"
        # llm_prompt = llm_prompt + 'Issue: ' + embeddings_text[sorted_index[i]] + '.'
        # llm_prompt = llm_prompt + 'Solution: ' + embeddings_answer[sorted_index[i]] + '.\n'
        # llm_prompt = llm_prompt + embeddings_answer[sorted_index[i]]
        llm_prompt = llm_prompt + story_text

    # LLM answer
    llm_prompt = str.replace(llm_prompt, "..", ".")
    llm_prompt = llm_prompt + "\nASSISTANT: "

    if use_llm:
        full_llm_answer = get_reply(llm_prompt)
    else:
        full_llm_answer = ""
    print(f"llm answer before prune: {full_llm_answer}\n")

    # llm_prompt = llm_prompt + str(len(llm_prompt))
    full_llm_answer = full_llm_answer[len(llm_prompt) :]
    print(f"llm answer after prune: {full_llm_answer}")
    print("------------")
    # full_llm_answer = full_llm_answer + str(len(full_llm_answer))

    solution_text = "".join([f"{i + 1}) {embeddings_answer[sorted_index[i]]}" for i in range(3)])
    answer_string = "".join([f"{i + 1}) {embeddings_text[sorted_index[i]]}" for i in range(3)])

    if not global_problem:
        sol_1 = embeddings_answer[sorted_index[0]][0].lower() + embeddings_answer[sorted_index[0]][1:]
        problem_1 = embeddings_text[sorted_index[0]][0].lower() + embeddings_text[sorted_index[0]][1:]
        global_problem = f"\n Ваша предыдущая проблема это: {problem_1}" + f" Мой совет: {sol_1}"
    else:
        answer_string += global_problem 

    if to_text == False:
        return answer_string, solution_text, full_llm_answer, llm_prompt
    else:
        return answer_string, solution_text, full_llm_answer, llm_prompt, text


def text_wrapper(input_text):
    answer_string, solution_text, full_llm_answer, llm_prompt = respond(input_text)
    audio = tts.va_speak(full_llm_answer)
    return answer_string, solution_text, full_llm_answer, llm_prompt, audio


def asr_wrapper(wav):
    whisper_text = speech_to_text(wav, device)
    answer_string, solution_text, full_llm_answer, llm_prompt, text = respond(
        whisper_text, to_text=True
    )
    audio = tts.va_speak(full_llm_answer)
    return answer_string, solution_text, full_llm_answer, llm_prompt, text, audio


def tts_speak(input_text):
    audio = tts.va_speak(input_text)
    return audio


def type_to_global(input_type):
    global selected_train_type
    selected_train_type = input_type
    return selected_train_type


def save_file(audio_input, problem, solution, llm_answer, name, type_of_train):
    file_name = f"./samples/{name}_{type_of_train}.csv"
    data_user.loc[len(data_user)] = [audio_input, llm_answer, problem, solution]
    data_user.to_csv(file_name, index=False)

def pass_logo():
    pass

# my_theme = gr.Theme.from_hub('rottenlittlecreature/Moon_Goblin')
with gr.Blocks(title="RZD Voice Assistant fttftf") as demo:
    if not os.path.exists("./samples"):
        os.makedirs("./samples")

    # with gr.Column(scale=2, min_width=600):
    #     img1 = gr.Image("./src/RZD.png")
    with gr.Row():
        type_of_train = gr.Dropdown(
            choices=types_of_trains,
            label="Выберите серию тепловоза / электровоза:",
            value="Все",
        )
        name = gr.Textbox(label="Имя машиниста")
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            audio_input = gr.Audio(source="microphone", type="numpy")
            clear = gr.Button("Очистить")
        audio_output = gr.Audio(type="numpy", autoplay=True)
    with gr.Row():
        text = gr.Textbox(label="Запрос")
    with gr.Row():
        problem = gr.Textbox(label="Проблема")
    with gr.Row():
        solution = gr.Textbox(label="Метод устранения", lines=4)
    with gr.Row():
        llm_answer = gr.Textbox(label="Ответ помощника", lines=7)
    with gr.Row():
        raw_prompt = gr.Textbox(label="Raw Prompt", lines=10, visible=False)

    data_user = pd.DataFrame(columns=["user_data", "llm_answer", "problem", "solution"])

    audio_input.stop_recording(
        fn=asr_wrapper,
        inputs=audio_input,
        outputs=[problem, solution, llm_answer, raw_prompt, text, audio_output],
    )

    text.submit(
        fn=text_wrapper,
        inputs=text,
        outputs=[problem, solution, llm_answer, raw_prompt, audio_output],
    )

    send_button = gr.Button(value="Отправить")
    send_button.click(
        fn=text_wrapper,
        inputs=text,
        outputs=[problem, solution, llm_answer, raw_prompt, audio_output],
    )

    type_of_train.change(fn=type_to_global, inputs=type_of_train, outputs=None)
    name.change(fn=lambda x: x, inputs=name)
    text.change(
        fn=save_file,
        inputs=[text, problem, solution, llm_answer, name, type_of_train],
        outputs=None,
    )
    clear.click(lambda :None, None, audio_input)
    

embeddings_raw, embeddings_text, embeddings_answer = create_embeddings(
    test_loco.dict, device
)


# demo.launch()
demo.launch(server_name="0.0.0.0", share=True)
