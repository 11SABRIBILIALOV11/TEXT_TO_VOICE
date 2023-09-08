import nltk
from transformers import BarkModel, AutoProcessor
import numpy as np
import torch
import scipy
from bark.generation import (
    generate_text_semantic,
    preload_models,
)
from bark.api import semantic_to_waveform
from bark import generate_audio, SAMPLE_RATE


def text_to_audio(voice_preset='v2/ru_speaker_6'):
    text = """
    Сайт рыбатекст поможет дизайнеру, верстальщику, вебмастеру сгенерировать несколько абзацев более менее осмысленного текста рыбы на 
    русском языке, а начинающему оратору отточить навык публичных выступлений в домашних условиях. При создании генератора мы использовали небезизвестный универсальный код речей. 
    Текст генерируется абзацами случайным образом от двух до десяти предложений в абзаце, что позволяет сделать текст более привлекательным и живым для визуально-слухового восприятия.
    """.replace("\n", " "). strip()

    sentences = nltk.sent_tokenize(text)
    silence = np.zeros(int(0.25 * SAMPLE_RATE))

    pieces = []

    for sentence in sentences:
        semantic_tokens = generate_text_semantic(
            sentence,
            history_prompt=voice_preset,
            min_eos_p=0.05,
        )
        audio_array = semantic_to_waveform(semantic_tokens, history_prompt=voice_preset)
        pieces += [audio_array, silence.copy()]

    scipy.io.wavfile.write(f'{voice_preset.split("/")[1]}_long.wav', rate=SAMPLE_RATE, data=np.concatenate(pieces))


def main():
    text_to_audio()


if __name__ == '__main__':
    main()

