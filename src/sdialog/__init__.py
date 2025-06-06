"""
sdialog: Synthetic Dialogue Generation Toolkit

This package provides utilities for generating synthetic dialogues using instruction-tuned large language models (LLMs).
Dialogues are generated primarily via role-playing, where each agent is defined by a Persona object. The package supports
dialogue orchestration, scenario management, and flexible serialization for downstream tasks.

Main components:
- Dialog, Turn, Event: Data structures for representing dialogues and their events.
- Persona and PersonaAgent: For defining and simulating role-played agents.
- Orchestrators: For controlling agent behavior during dialogue generation.
- Utility functions for serialization, pretty-printing, and file I/O.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import os
import json

from pydantic import BaseModel
from typing import List, Union, Optional
from print_color import print

from .util import make_serializable


__version__ = "0.0.2"


class Turn(BaseModel):
    """
    Represents a single turn in a dialogue.

    Attributes:
        speaker (Optional[str]): The name or role of the speaker.
        text (str): The utterance text for this turn.
    """
    speaker: Optional[str]
    text: str


class Event(BaseModel):
    """
    Represents an event in a dialogue, which may be an utterance, instruction, or other action.

    Attributes:
        agent (Optional[str]): The agent responsible for the event (e.g., "user", "system").
        action (str): The type of event (e.g., "utter", "instruct").
        actionLabel (Optional[str]): A label describing the action (e.g., type of instruction).
        text (str): The content of the event.
        timestamp (int): The Unix timestamp of the event.
    """
    agent: Optional[str] = None # "user", "system"
    action: str  # "utter", "instruct"
    actionLabel: Optional[str] = None # action label (e.g. type of instruct)
    text: str  # the content of the event
    timestamp: int  # timestemp


class Dialog(BaseModel):
    """
    Represents a full dialogue, including turns, events, and scenario metadata.

    Attributes:
        formatVersion (Optional[str]): Version of the dialogue format.
        model (Optional[str]): The model used to generate the dialogue.
        seed (Optional[int]): The random seed used for generation.
        dialogId (Optional[int]): Unique identifier for the dialogue.
        complete (Optional[bool]): Whether the dialogue is complete.
        scenario (Optional[Union[dict, str]]): Scenario description or metadata.
        turns (List[Turn]): List of dialogue turns.
        events (Optional[List[Event]]): List of dialogue events (optional).
    """
    formatVersion: Optional[str] = "0.0.5"  # Version of the format
    model: Optional[str] = None  # the model used to generate the dialogue
    seed: Optional[int] = None  # the seed used to generated
    dialogId: Optional[int] = None
    complete: Optional[bool] = None
    scenario: Optional[Union[dict, str]] = None  # the scenario used to generated the dialogue
    turns: List[Turn]  # the list of turns of the conversation
    events: Optional[List[Event]] = None

    def __len__(self):
        """Returns the number of turns in the dialogue."""
        return len(self.turns)

    def description(self, turn_template: str = "{speaker}: {text}"):
        """
        Returns a human-readable string representation of the dialogue.

        Args:
            turn_template (str): Template for formatting each turn.

        Returns:
            str: The formatted dialogue.
        """
        return "\n".join(turn_template.format(speaker=turn.speaker, text=turn.text.replace("\n", " "))
                         for turn in self.turns)

    def json(self, string: bool = False, indent: int = None):
        """
        Serializes the dialogue to JSON.

        Args:
            string (bool): If True, returns a JSON string; otherwise, returns a dict.
            indent (int): Indentation level for pretty-printing.

        Returns:
            Union[str, dict]: The serialized dialogue.
        """
        data = self.model_dump()
        make_serializable(data)
        return json.dumps(data, indent=indent) if string else data

    def print(self, *a, **kw):
        """
        Pretty-prints the dialogue to the console.

        Args:
            scenario (bool): If True, prints scenario information.
            orchestration (bool): If True, prints orchestration events.
        """
        print_dialog(self, *a, **kw)

    def to_file(self, path: str, type: str = "auto", makedir: bool = True):
        """
        Saves the dialogue to a file in either JSON or plain text format.

        Args:
            path (str): Output file path.
            type (str): "json", "txt", or "auto" (determined by file extension).
            makedir (bool): If True, creates parent directories as needed.
        """
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
    def from_file(path: str, type: str = "auto"):
        """
        Loads a dialogue from a file.

        Args:
            path (str): Path to the dialogue file.
            type (str): "json", "txt", or "auto" (determined by file extension).

        Returns:
            Dialog: The loaded dialogue object.
        """
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
    """
    Represents an instruction to an agent, optionally with associated events.

    Attributes:
        text (str): The instruction text.
        events (Optional[Union[Event, List[Event]]]): Associated events (optional).
    """
    text: str = None
    events: Optional[Union[Event, List[Event]]] = None  # extra events


def print_dialog(dialog: Union[Dialog, dict], scenario: bool = False, orchestration: bool = False):
    """
    Pretty-prints a dialogue to the console, with optional scenario and orchestration details.

    Args:
        dialog (Union[Dialog, dict]): The dialogue to print.
        scenario (bool): If True, prints scenario information.
        orchestration (bool): If True, prints orchestration events instead of turns.
    """
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
