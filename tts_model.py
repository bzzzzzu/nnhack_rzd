from RUTTS import TTS
from ruaccent import RUAccent
from datetime import datetime
import time
from pathlib import Path
import tts_preprocessor


dir_path = Path.cwd()
model_name = 'TeraTTS/natasha-g2p-vits'
model_path = dir_path / 'model' / model_name
output_dir = dir_path / 'saved_tts_audio'


class Processor:
    def __init__(self,
                 model_path=model_path,
                 model_name=model_name):

        self.model_path = model_path
        self.model_name = model_name
        if self.model_path.exists():
            self.tts = TTS(model_path, add_time_to_end=0.8)      # add_time_to_end продолжительность аудио
        else:
            print('Скачивание модели...')
            self.tts = TTS(model_name, add_time_to_end=0.8)
        
        self.accentizer = RUAccent(workdir="./model")
        self.accentizer.load(omograph_model_size='medium', dict_load_startup=False)


    def va_speak(self, text: str, play: bool=False, save: bool=False):
        time_stamp = str(int(time.time()))
        text = tts_preprocessor.preprocess(text)
        text = self.accentizer.process_all(text)
        audio = self.tts(text, play)
        if save:
            self.tts.save_wav(audio, f'{output_dir}/audio_{time_stamp}.wav')
        return audio


if __name__ == '__main__':
    sample = Processor()
    text = """Неисправна плата ПВАД или БОАД УОИ. Проверить наличие тока через мотор-вентиляторы и резисторы ЭДТ.
    Если есть неисправность, то не пользоваться ЭДТ. Если возникает в режиме "Тяги", то отключить ОМ1, ОМ2, ОМ3."""
    text = sample.va_speak(text, play=False, save=True)