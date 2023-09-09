import whisper
import torch
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio
import torchaudio.transforms as T
import numpy as np

def speech_to_text(wav):
    audio = wav[1]
    sr = wav[0]

    resampler = torchaudio.transforms.Resample(sr, 16000)
    model_denoiser = pretrained.dns64()
    if torch.cuda.is_available():
        model_whisper = whisper.load_model("small", device=torch.device("cuda"))
    else:
        model_whisper = whisper.load_model("tiny", device=torch.device("cpu"))

    audio = resampler(torch.Tensor(audio))
    # wav_2 = wav_2 * 32768.0
    audio = audio / 32768.0 / 32768.0
    audio = convert_audio(
        audio.unsqueeze(0),
        model_denoiser.sample_rate,
        model_denoiser.sample_rate,
        model_denoiser.chin,
    )
    with torch.no_grad():
        denoised_audio = model_denoiser(audio[None])[0].data.cpu().numpy()
    denoised_audio = np.squeeze(denoised_audio)

    predictions = model_whisper.transcribe(denoised_audio, language="ru")["text"]
    return predictions
