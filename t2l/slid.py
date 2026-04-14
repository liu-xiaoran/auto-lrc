import logging
import os

import numpy as np
import torch
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions, collect_chunks, get_speech_timestamps

from . import t2l

logger = logging.getLogger("t2l")


def detect_language(audio_tensor, model='tiny', vad=True, no_voice_ratio=0.01):
    """
    :param audio_tensor: [..., time], and sr=16000
    :param vad: voice activity detection, remove silence from audio
    :return: [(lang, prob), ...]
    """
    if isinstance(model, str):
        if torch.cuda.is_available():
            model = WhisperModel(
                model, device="cuda", compute_type="int8_float16")
        else:
            model = WhisperModel(
                model, device="cpu", compute_type="int8")

    mono = audio_tensor[tuple(0 for _ in range(
        len(audio_tensor.shape)-1)) + (slice(None),)].cpu()
    # 过滤静音处 vad
    ratio_silenced = 1
    if vad:
        duration = mono.shape[0]
        # audio_tensor = audio = remove_silence(
        #     audio_tensor, silence_threshold, sample_rate=16000)
        # ratio_silenced = audio.shape[-1]/duration

        vad_parameters = VadOptions()
        speech_chunks = get_speech_timestamps(mono, vad_parameters)
        mono = collect_chunks(mono, speech_chunks)
        duration_after_vad = mono.shape[0]
        ratio_silenced = duration_after_vad/duration
        logger.info("silence VAD ratio: %s", ratio_silenced)
        # 无人声
        if no_voice_ratio > 0 and ratio_silenced < no_voice_ratio:
            return [(None, 1 - ratio_silenced)]

        # results is a list of tuple[lang, prob]
    features = model.feature_extractor(mono)
    segment = features[:, : model.feature_extractor.nb_max_frames]
    encoded = model.encode(segment)
    probs = model.model.detect_language(encoded)[0]
    probs = [(token[2:-2], prob) for (token, prob) in probs]
    logger.info(f"Detected language: {probs[:3]}, ratio_silenced={ratio_silenced}")

    return probs


def test_slid(audio_file_path='./test/japanese.mp3', demucs=True):
    file_size = os.path.getsize(audio_file_path)/1024//1024
    print('file_size=', file_size, audio_file_path)

    MODEL_SIZE = 'tiny'
    if torch.cuda.is_available():
        lyrics_model = WhisperModel(
            MODEL_SIZE, device="cuda", compute_type="int8_float16")
        print("WhisperModel cuda int8_float16")
    else:
        lyrics_model = WhisperModel(
            MODEL_SIZE, device="cpu", compute_type="int8")
        print("WhisperModel cpu int8")

    # demucs
    if demucs:
        # load demucs model
        demucs_model = t2l.load_demucs_model('mdx_extra', 1)
        print('demucs model loaded. cuda=', torch.cuda.is_available())
        vocals, _ = t2l.vocalize(audio_file_path, 16000, demucs_model)
    else:
        vocals, _ = t2l.preprocess_audio(audio_file_path, 16000)
    print('vocals shape=', vocals.shape)
    return detect_language(vocals, lyrics_model)
