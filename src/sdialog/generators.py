# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import json
import random

from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from pydantic import BaseModel

from print_color import print
from typing import Union, List
from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from . import Dialog, Turn


class LLMDialogOutput(BaseModel):
  dialog: List[Turn]


# TODO: create a BaseDialogGenerator, and also PersonaDialogGenerator that takes personas objects as in multi-agent.
class DialogGenerator:
    def __init__(self, model: Union[ChatOllama, str], dialogue_details: str, output_format: Union[dict, BaseModel] = LLMDialogOutput, scenario: dict = None):
        """Optional `scenario` to populate the "scenario" field of the output, if not provided, `dialogue_details` content will be used."""

        if not output_format or type(output_format) == dict:
            output_format_schema = output_format
            self.output_format = None
        else:
            output_format_schema = output_format.model_json_schema()
            self.output_format = output_format

        if type(model) == str:
            self.llm = ChatOllama(model=model,
                                  format=output_format_schema,
                                  temperature=0.8,
                                  seed=13)
        else:
            self.llm = model
            if output_format:
                self.llm.format = output_format

        self.model_name = model
        self.set(dialogue_details, scenario)

    def generate(self, seed: int = None, id: int = None):
        self.llm.seed = seed if seed is not None else random.getrandbits(32)

        # hack to avoid seed bug in prompt cache (to force a new cache, related to https://github.com/ollama/ollama/issues/5321)
        _ = self.llm.num_predict
        self.llm.num_predict = 1
        self.llm.invoke(self.messages)
        self.llm.num_predict = _

        dialogue = self.llm.invoke(self.messages).content

        if not self.output_format:
            return dialogue
        else:
            llm_output = self.output_format.model_validate(json.loads(dialogue))

            if self.output_format is LLMDialogOutput:
                return Dialog(dialogId=id if id else None,
                            model=self.model_name,
                            seed=self.llm.seed,
                            scenario=self.scenario if self.scenario else self.dialogue_details,
                            turns=llm_output.dialog)
            else:
                return llm_output

    def set(self, dialogue_details: str, scenario:dict=None):
        self.scenario = scenario
        self.dialogue_details = dialogue_details
        self.messages = [
            SystemMessage(
                "You are a knowledgeable and useful AI assistant that can write natural conversations by role paying different speakers."
                "The output should be a full dialogue, from begining (greetings) to end (bye bye messages)."
            ),
            HumanMessage(content=dialogue_details)
        ]

    __call__ = generate
