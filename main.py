# Данные - чтение документа и разбивка на читаемые куски
from document_reader import get_text_from_document

# Эмбеддинги
from make_embeddings import create_embeddings

# Вход - звуковой файл
from whisper_speech_recognition import speech_to_text

# Языковая модель
from text_llm import get_reply

# Озвучка текста
from tts_model import get_wav_from_text

# Gradio main thread
import gradio as gr

with gr.Blocks() as demo:
    text = gr.Textbox(label="Text")

demo.launch()