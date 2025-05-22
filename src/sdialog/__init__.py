# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import os
import json

from pydantic import BaseModel
from typing import List, Union, Optional
from print_color import print

from .util import make_serializable


class Turn(BaseModel):
    speaker: Optional[str]
    text: str


class Event(BaseModel):
    agent: Optional[str] = None # "user", "system"
    action: str  # "utter", "instruct"
    actionLabel: Optional[str] = None # action label (e.g. type of instruct)
    text: str  # the content of the event
    timestamp: int  # timestemp


class Dialog(BaseModel):
    formatVersion: Optional[str] = "0.0.5"  # Version of the format
    model: Optional[str] = None  # the model used to generate the dialogue
    seed: Optional[int] = None  # the seed used to generated
    dialogId: Optional[int] = None
    complete: Optional[bool] = None
    scenario: Optional[Union[dict, str]] = None  # the scenario used to generated the dialogue
    turns: List[Turn]  # the list of turns of the conversation
    events: Optional[List[Event]] = None

    def __len__(self):
        return len(self.turns)

    def description(self, turn_template: str = "{speaker}: {text}"):
        return "\n".join(turn_template.format(speaker=turn.speaker, text=turn.text.replace("\n", " "))
                         for turn in self.turns)

    def json(self, string: bool = False, indent: int = None):
        data = self.model_dump()
        make_serializable(data)
        return json.dumps(data, indent=indent) if string else data

    def print(self, *a, **kw):
        print_dialog(self, *a, **kw)

    def to_file(self, path: str, type: str = "auto", makedir: bool = True):  # type = "txt", "json" or "auto" which get's the type from the file extention
        if type == "auto":
            type = "json" if path.endswith(".json") else "txt"

        if makedir:
            os.makedirs(os.path.split(path)[0], exist_ok=True)

        with open(path, "w") as writer:
            if type == "json":
                writer.write(self.json(string=True))
            else:
                writer.write(self.description())

    @staticmethod
    def from_file(path: str, type: str = "auto"):  # type = "txt", "json" or "auto" which get's the type from the file extention
        if type == "auto":
            type = "json" if path.endswith(".json") else "txt"

        with open(path) as reader:
            if type == "json":
                return Dialog.model_validate(json.load(reader))

            lines = reader.read().split("\n")

        return Dialog(turns=[Turn(speaker=line[:line.index(":")].strip(),
                                  text=line[line.index(":") + 1:].strip())
                             for line in lines if line])

    # TODO: add from_dict as an alias of (so we don't have to use .model_validate())

    __str__ = description


class Instruction(BaseModel):
    text: str = None
    events: Optional[Union[Event, List[Event]]] = None  # extra events


def print_dialog(dialog: Union[Dialog, dict], scenario: bool = False, orchestration: bool = False):
    if type(dialog) == dict:
        dialog = Dialog.model_validate(dialog)

    speaker_tag_colors = ["red", "blue", "yellow", "cyan", "green", "magenta", "purple"]
    speaker_utt_colors = ["grey", "white"]
    # speaker_utt_colors = ["black", "grey"]

    if dialog.dialogId:
        print(dialog.dialogId, tag="dialog_id", tag_color="purple", color="magenta", format="bold")
    if dialog.complete:
        print(dialog.complete, tag="complete", tag_color="purple", color="magenta", format="bold")
    if dialog.model:
        print(dialog.model, tag="model", tag_color="purple", color="magenta", format="bold")
    if dialog.seed:
        print(dialog.seed, tag="seed", tag_color="purple", color="magenta", format="bold")
    if scenario and dialog.scenario:
        print("", tag="scenario", tag_color="purple", color="magenta", format="bold")
        if type(dialog.scenario) == str:
            print(dialog.scenario, color="magenta")
        else:
            print(json.dumps(dialog.scenario, indent=2), color="magenta")

    print("--- Dialogue Begins ---", color="magenta", format="bold")
    speakers = sorted(list(set(turn.speaker for turn in dialog.turns)))
    if orchestration:
        dialog = dialog.model_copy()
        dialog.turns = [Turn(speaker=e.agent, text=e.text) if e.action == "utter"
                        else (
                            Turn(speaker=e.agent, text=f"[pick_suggestion] {e.text}") if e.action == "pick_suggestion"
                            else
                            Turn(speaker=e.action, text=f"({e.agent}) {e.text}"))
                        for e in dialog.events]

    for ix, turn in enumerate(dialog.turns):
        speaker = turn.speaker

        if speaker not in speakers:
            tag_color = "yellow"
            color = "purple"
        else:
            tag_color = speaker_tag_colors[speakers.index(speaker) % len(speaker_tag_colors)]
            color = speaker_utt_colors[speakers.index(speaker) % len(speaker_utt_colors)]

        print(turn.text,
              tag=speaker,
              tag_color=tag_color,
              color=color)
    print("--- Dialogue Ends ---", color="magenta", format="bold")
