import whisper
import torch
import torchaudio
import soundfile
import numpy as np

def speech_to_text(wav):
    #wav_2, sr = soundfile.read(wav, dtype='float32')
    wav_2 = wav[1]
    sr = wav[0]
    print(sr)
    print(wav_2)
    print(f'mean: {np.mean(wav_2)}, std: {np.std(wav_2)}, 90per: {np.percentile(wav_2, 90)}')
    resampler = torchaudio.transforms.Resample(sr, 16000)

    wav_2 = resampler(torch.Tensor(wav_2))
    #wav_2 = wav_2 * 32768.0
    wav_2 = wav_2 / 32768.0 / 32768.0
    print(wav_2)
    print(f'mean: {torch.mean(wav_2)}, std: {torch.std(wav_2)}, 90per: {torch.quantile(wav_2, 0.9)}')
    print(wav_2.dtype)

    #exit()
    if torch.cuda.is_available():
        model_whisper = whisper.load_model("small", device=torch.device("cuda"))
    else:
        model_whisper = whisper.load_model("tiny", device=torch.device("cpu"))
    predictions = model_whisper.transcribe(wav_2, language='ru')
    print(predictions)
    return predictions['text']

#result = speech_to_text('m.ogg')
#print(result)