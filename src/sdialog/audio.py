"""
util: Utility Functions for sdialog

This module provides helper functions for the sdialog package, including serialization utilities to ensure
objects can be safely converted to JSON for storage or transmission.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>, Yanis Labrak <yanis.labrak@univ-avignon.fr>
# SPDX-License-Identifier: MIT
import json
import numpy as np

# Audio processing
import soundfile as sf
from kokoro import KPipeline

from sdialog import Dialog

pipeline = KPipeline(lang_code='a')


def generate_utterance(text: str, persona: dict, voice: str = "af_heart") -> None:
    """
    Generates an audio recording of a text utterance based on the speaker persona.
    """

    generator = pipeline(text, voice=voice)

    gs, ps, audio = next(iter(generator))

    return audio


def master_audio(dialogue_audios: list) -> None:
    """
    Combines multiple audio segments into a single master audio track.
    """
    master_audio = np.concatenate(dialogue_audios)
    return master_audio


def to_wav(audio, output_file, sampling_rate=24000) -> None:
    """
    Combines multiple audio segments into a single master audio track.
    """
    sf.write(output_file, audio, sampling_rate)


def generate_audios(dialog: Dialog):

    dialogue_audios = []

    for turn in dialog.turns:

        utterance_audio = generate_utterance(turn.text, dialog.personas[turn.speaker])
        dialogue_audios.append(utterance_audio)

    return dialogue_audios


def dialog_to_audio(dialog: Dialog):

    dialogue_audios = generate_audios(dialog)

    combined_audio = master_audio(dialogue_audios)

    return combined_audio


def make_serializable(data: dict) -> dict:
    """
    Converts non-serializable values in a dictionary to strings so the dictionary can be safely serialized to JSON.

    :param data: The dictionary to process.
    :type data: dict
    :return: The dictionary with all values JSON-serializable.
    :rtype: dict
    """

    if type(data) is not dict:
        raise TypeError("Input must be a dictionary")

    for key, value in data.items():
        try:
            json.dumps(value)
        except (TypeError, OverflowError):
            data[key] = str(value)

    return data
