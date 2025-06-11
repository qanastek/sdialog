"""
personas: Persona and Agent Definitions for Synthetic Dialogue Generation

This module provides classes for defining personas (character profiles) and simulating agents that role-play these personas
in synthetic dialogue generation. Agents interact using LLMs and can be orchestrated for complex behaviors.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import json
import random

from time import time
from tqdm.auto import trange

from typing import List, Union
from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from . import Dialog, Turn, Event, Instruction
from .orchestrators import BaseOrchestrator
from .util import make_serializable


class __Meta__(type):
    """
    Metaclass for enabling automatic __init__ chaining in subclasses.
    """
    def __init__(cls, name, bases, dct):
        def auto__call__init__(self, *a, **kw):
            for base in cls.__bases__:
                base.__init__(self, *a, **kw)
            cls.__init__child_(self, *a, **kw)
            cls.__init__child_ = cls.__init__
            cls.__init__ = auto__call__init__


class BasePersona(metaclass=__Meta__):
    """
    Base class for defining a persona (character profile) for role-play.

    :param kwargs: Arbitrary keyword arguments are stored as persona attributes.
    """
    def __init__(self, **kwargs):
        """
        Initializes the persona with arbitrary attributes.

        :param kwargs: Arbitrary persona attributes.
        """
        self.__dict__.update(kwargs)

    def description(self) -> str:
        """
        Returns a string description of the persona's attributes.

        :return: Description of the persona.
        :rtype: str
        """
        return "\n".join(f"Your {key}: {value}" for key, value in self.__dict__.items())

    def __str__(self) -> str:
        """
        Returns the string representation of the persona.

        :return: Description of the persona.
        :rtype: str
        """
        return self.description()

    def json(self, string: bool = False, indent=None):
        """
        Serializes the persona to JSON.

        :param string: If True, returns a JSON string; otherwise, returns a dict.
        :type string: bool
        :param indent: Indentation level for pretty-printing.
        :type indent: int
        :return: The serialized persona.
        :rtype: Union[str, dict]
        """
        data = self.__dict__.copy()
        make_serializable(data)
        return json.dumps(data, indent=indent) if string else data


class Persona(BasePersona):
    """
    Standard persona class with common attributes for role-play.

    :ivar name: Name of the persona.
    :vartype name: str
    :ivar role: Role or occupation.
    :vartype role: str
    :ivar background: Background information.
    :vartype background: str
    :ivar personality: Personality traits.
    :vartype personality: str
    :ivar circumstances: Current circumstances.
    :vartype circumstances: str
    :ivar rules: Rules or constraints.
    :vartype rules: str
    :ivar language: Preferred language.
    :vartype language: str
    """
    name: str = ""
    role: str = ""
    background: str = ""
    personality: str = ""
    circumstances: str = ""
    rules: str = ""
    language: str = ""


class PersonaAgent:
    """
    Agent that simulates a persona in dialogue using an LLM.

    :cvar STOP_WORD: Special token to indicate end of conversation.
    :vartype STOP_WORD: str
    :cvar STOP_WORD_TEXT: Replacement text for STOP_WORD.
    :vartype STOP_WORD_TEXT: str
    """

    STOP_WORD = "STOP"
    STOP_WORD_TEXT = "(bye bye!)"

    def __init__(self,
                 model: Union[str, ChatOllama],
                 persona: BasePersona = Persona(),
                 name: str = None,
                 dialogue_details: str = "",
                 response_details: str = "responses SHOULD NOT be too long and wordy, should be "
                                         "approximately one utterance long",
                 system_prompt: str = None,
                 can_finish: bool = False,
                 orchestrators: Union[BaseOrchestrator, List[BaseOrchestrator]] = None,
                 scenario: Union[dict, str] = None):
        """
        Initializes a PersonaAgent for role-play dialogue.

        :param model: The LLM or model name to use.
        :type model: Union[str, ChatOllama]
        :param persona: The persona to role-play.
        :type persona: BasePersona
        :param name: Name of the agent.
        :type name: str
        :param dialogue_details: Additional details about the dialogue.
        :type dialogue_details: str
        :param response_details: Instructions for response style.
        :type response_details: str
        :param system_prompt: Custom system prompt (optional).
        :type system_prompt: str
        :param can_finish: If True, agent can end the conversation.
        :type can_finish: bool
        :param orchestrators: Orchestrators for agent behavior.
        :type orchestrators: Union[BaseOrchestrator, List[BaseOrchestrator]]
        :param scenario: Scenario metadata.
        :type scenario: Union[dict, str]
        """

        if not system_prompt:
            if can_finish:
                conversation_end_instructions = ("To finish the conversation you first have to say good bye and "
                                                 f"immediately after you **MUST** output '{self.STOP_WORD}' to "
                                                 "indicate it is the end of it.")
            else:
                conversation_end_instructions = ("When the user finish the conversation you should say good bye and "
                                                 "also finish the conversation")

            # system_prompt = prompt_template.format(role=role, ...)
            system_prompt = f"""Role play as a character that is described by the persona defined in the following lines. You always stay in character.
[[ ## BEGING PERSONA ## ]]
{persona}
[[ ## END PERSONA ## ]]
---
{"Details about the overall dialogue: " + dialogue_details if dialogue_details else ""}
{"Details about your responses: " + response_details if response_details else ""}
Finally, remember:
   1. You always stay on character. You are the character described above.
   2. Your first utterance / turn MUST always be a short generic greeting (e.g. "Hello, how are you?", "Hi!", "hey! what's up?", etc.), and nothing else, wait for a reply before start with the actual conversation.
   3. {conversation_end_instructions}."""

        if type(model) is str:
            # TODO: ChatHuggingFace
            self.llm = ChatOllama(model=model,
                                  temperature=0.8,
                                  seed=13)
        else:
            self.llm = model
        self.memory = [SystemMessage(system_prompt)]

        self.name = name if name else (persona.name if hasattr(persona, "name") else None)
        self.persona = persona
        self.model_name = str(self.llm)
        self.first_utterances = None
        self.finished = False
        self.scenario = scenario
        self.orchestrators = None
        self.add_orchestrators(orchestrators)

    def __call__(self, utterance: str = "", return_events: bool = False) -> str:
        """
        Processes an input utterance and generates a response.

        :param utterance: The input utterance from the other agent or user.
        :type utterance: str
        :param return_events: If True, returns a list of events instead of just the response string.
        :type return_events: bool
        :return: The agent's response or events, or None if finished.
        :rtype: Union[str, List[Event], None]
        """
        if self.finished:
            return None

        if utterance:
            self.memory.append(HumanMessage(content=utterance))

        if return_events:
            events = []
        if self.orchestrators:
            for orchestrator in self.orchestrators:
                instruction = orchestrator()
                if instruction:

                    if type(instruction) is Instruction:
                        if return_events and instruction.events:
                            if type(instruction.events) is Event:
                                events.append(instruction.events)
                            else:
                                events.extend(instruction.events)
                        instruction = instruction.text

                    persist = orchestrator.is_persistent()
                    self.instruct(instruction, persist=persist)
                    if return_events:
                        events.append(Event(agent=self.get_name(),
                                            action="instruct" + ("-persist" if persist else ""),
                                            actionLabel=orchestrator.get_event_label(),
                                            text=instruction,
                                            timestamp=int(time())))

        if len(self.memory) <= 1 and self.first_utterances:
            response = (random.choice(self.first_utterances) if type(self.first_utterances) is list
                        else self.first_utterances)
            response = AIMessage(content=response)
        else:
            response = self.llm.invoke(self.memory)

        if self.orchestrators:
            self.memory[:] = [msg for msg in self.memory
                              if not (msg.response_metadata
                                      and "persist" in msg.response_metadata
                                      and not msg.response_metadata["persist"])]
        self.memory.append(response)

        response = response.content
        if self.STOP_WORD in response:
            response = response.replace(self.STOP_WORD, self.STOP_WORD_TEXT).strip()
            self.memory[-1].content = self.memory[-1].content.replace(self.STOP_WORD, "").strip()
            self.finished = True

        if return_events:
            if response:
                events.append(Event(agent=self.get_name(),
                                    action="utter",
                                    text=response,
                                    timestamp=int(time())))
            return events
        else:
            return response if response else ""

    def __or__(self, orchestrator: Union[BaseOrchestrator, List[BaseOrchestrator]]):
        """
        Adds orchestrators to the agent using the | operator.

        :param orchestrator: Orchestrator(s) to add.
        :type orchestrator: Union[BaseOrchestrator, List[BaseOrchestrator]]
        :return: The agent with orchestrators added.
        :rtype: PersonaAgent
        """
        self.add_orchestrators(orchestrator)
        return self

    def response_lookahead(self, utterance: str = None):
        """
        Generates a response to a hypothetical next utterance without updating memory.

        :param utterance: The hypothetical next utterance.
        :type utterance: str
        :return: The predicted response.
        :rtype: str
        """
        if not utterance:
            return self.llm.invoke(self.memory).content
        return self.llm.invoke(self.memory + [HumanMessage(utterance)]).content

    def add_orchestrators(self, orchestrators):
        """
        Adds orchestrators to the agent.

        :param orchestrators: Orchestrator(s) to add.
        :type orchestrators: Union[BaseOrchestrator, List[BaseOrchestrator]]
        """
        if not orchestrators:
            return

        if self.orchestrators is None:
            self.orchestrators = []

        if isinstance(orchestrators, BaseOrchestrator):
            orchestrators = [orchestrators]

        self.orchestrators.extend(orchestrators)

        for orchestrator in orchestrators:
            orchestrator._set_target_agent(self)

    def clear_orchestrators(self):
        """
        Removes all orchestrators from the agent.
        """
        self.orchestrators = None

    def instruct(self, instruction: str, persist: bool = False):
        """
        Adds a system instruction to the agent's memory.

        :param instruction: The instruction text.
        :type instruction: str
        :param persist: If True, instruction persists across turns.
        :type persist: bool
        """
        self.memory.append(SystemMessage(instruction, response_metadata={"persist": persist}))

    def set_first_utterances(self, utterances: Union[str, List[str]]):
        """
        Sets the agent's first utterance(s) for dialogue initialization.

        :param utterances: The greeting(s) to use.
        :type utterances: Union[str, List[str]]
        """
        self.first_utterances = utterances

    def get_name(self):
        """
        Returns the agent's name.

        :return: The agent's name.
        :rtype: str
        """
        return self.name

    def get_prompt(self):
        """
        Returns the current system prompt.

        :return: The system prompt.
        :rtype: str
        """
        return self.memory[0].content

    def json(self, string: bool = False, indent=None):
        """
        Serializes the agent's configuration and persona to JSON.

        :param string: If True, returns a JSON string; otherwise, returns a dict.
        :type string: bool
        :param indent: Indentation level for pretty-printing.
        :type indent: int
        :return: The serialized agent.
        :rtype: Union[str, dict]
        """
        data = {}
        if self.name:
            data["name"] = self.name
        data["model_name"] = self.model_name
        if self.first_utterances:
            data["first_utterances"] = self.first_utterances
        data["persona"] = self.persona.json()
        if self.orchestrators:
            data["persona"]["orchestrators"] = [orc.json() for orc in self.orchestrators]
        return json.dumps(data, indent=indent) if string else data

    def reset(self, seed: int = None):
        """
        Resets the agent's memory and orchestrators, optionally reseeding the LLM.

        :param seed: Random seed for reproducibility.
        :type seed: int
        """
        self.memory[:] = self.memory[:1]
        self.finished = False
        self.llm.seed = seed

        if self.orchestrators:
            for orchestrator in self.orchestrators:
                orchestrator.reset()

        # hack to avoid seed bug in prompt cache
        # (to force a new cache, related to https://github.com/ollama/ollama/issues/5321)
        _ = self.llm.num_predict
        self.llm.num_predict = 1
        self.llm.invoke(self.memory)
        self.llm.num_predict = _

    def dialog_with(self,
                    persona: "PersonaAgent",
                    max_iterations: int = 20,
                    id: int = None,
                    seed: int = None,
                    keep_bar: bool = True):
        """
        Simulates a dialogue between this agent and another PersonaAgent.

        :param persona: The other agent to converse with.
        :type persona: PersonaAgent
        :param max_iterations: Maximum number of dialogue turns.
        :type max_iterations: int
        :param id: Dialogue ID.
        :type id: int
        :param seed: Random seed for reproducibility.
        :type seed: int
        :param keep_bar: If True, keeps the progress bar visible.
        :type keep_bar: bool
        :return: The generated dialogue object.
        :rtype: Dialog
        """
        seed = seed if seed is not None else random.getrandbits(32)

        random.seed(seed)
        self.reset(seed)
        persona.reset(seed)

        dialog = []
        events = []

        utter = None
        completion = False
        tqdm_iterator = trange(max_iterations, desc="Dialogue", leave=keep_bar)
        for _ in tqdm_iterator:
            utt_events = self(utter, return_events=True)

            if utt_events and utt_events[-1].action == "utter":
                utter = utt_events[-1].text
                utt_events[-1].text = utter.replace(self.STOP_WORD_TEXT, "").strip()
                if not utt_events[-1].text:
                    break
            else:
                completion = True
                break

            dialog.append(Turn(
                speaker=self.get_name() if self.get_name() else "Me",
                text=utt_events[-1].text
            ))
            events.extend(utt_events)

            utt_events = persona(utter, return_events=True)
            if utt_events and utt_events[-1].action == "utter":
                utter = utt_events[-1].text
                utt_events[-1].text = utter.replace(self.STOP_WORD_TEXT, "").strip()
                if not utt_events[-1].text:
                    break
            else:
                completion = True
                break

            dialog.append(Turn(
                speaker=persona.get_name() if persona.get_name() else "Other",
                text=utt_events[-1].text
            ))
            events.extend(utt_events)

        if not keep_bar:
            try:
                tqdm_iterator.container.close()
            except AttributeError:
                pass

        if self.scenario:
            scenario = self.scenario
        else:
            scenario = {
                "agents": [
                    self.json(),
                    persona.json()
                ]
            }

        return Dialog(
            dialogId=id if id else None,
            complete=completion,  # incomplete if ran out of iterations (reached max_iteration number)
            model=self.model_name,
            seed=seed,
            scenario=scenario,
            turns=dialog,
            events=events
        )

    talk_with = dialog_with
