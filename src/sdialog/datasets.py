# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import os
import re
import json

from tqdm.auto import tqdm

from . import Dialog, Turn, Event
from .personas import Persona, PersonaAgent
from .orchestrators import InstructionListOrchestrator, SimpleResponseOrchestrator

class STAR:
    _path = None
    _speakers = ["User", "Wizard"]

    @staticmethod
    def set_path(path):
        STAR._path = path

    @staticmethod
    def read_graph(task_name, as_dot: bool = True):
        with open(os.path.join(STAR._path, f"tasks/{task_name}/{task_name}.json")) as reader:
            if not as_dot:
                return json.load(reader)["graph"]
            dot_edges = ";\n".join(f"    {a} -> {b}" for a,b in json.load(reader)["graph"].items())

        return "digraph %s  {\n%s\n}" % (task_name, dot_edges)

    @staticmethod
    def read_graph_responses(task_name, as_dict: bool = False):
        with open(os.path.join(STAR._path, f"tasks/{task_name}/responses.json")) as reader:
            responses = json.load(reader)
            responses = {key:re.sub(r"{(.+?)(?::\w+?)?}", lambda m:m.group(1).upper(), value)
                         for key, value in responses.items()
                         if key != "out_of_scope"}
            return responses if as_dict else json.dumps(responses, indent=2)

    @staticmethod
    def get_dialog(id):
        with open(os.path.join(STAR._path, f"dialogues/{id}.json")) as reader:
            dialog = json.load(reader)

        for e in dialog["Events"]:
            if e["Agent"] == "Wizard":
                e["Agent"] = "System"

        return Dialog(
                dialogId=id,
                scenario=dialog["Scenario"],
                turns=[Turn(speaker=e["Agent"], text=e["Text"])
                    for e in dialog["Events"]
                    if e["Action"] in ["utter", "pick_suggestion"]],
                events=[Event(agent=e["Agent"],
                            action=e["Action"],
                            actionLabel=e["ActionLabel"] if "ActionLabel" in e else None,
                            text=e["Text"],
                            timestamp=e["UnixTime"])
                        for e in dialog["Events"]
                        if "Text" in e]
            )

    @staticmethod
    def get_dialogs(domain: str = None, task_name: str = None, happy: bool = None, multitask: bool = None):
        dialogs = []
        for fname in tqdm(os.listdir(os.path.join(STAR._path, f"dialogues/")), desc="Reading dialogs", leave=False):
            if not fname.endswith(".json"):
                continue
            dialog_id = int(os.path.splitext(fname)[0])
            scenario = STAR.get_dialog_scenario(dialog_id)

            if (domain is None or domain in scenario["Domains"]) and \
               (happy is None or scenario["Happy"] == happy) and \
               (multitask is None or scenario["MultiTask"] == multitask) and \
               (task_name is None or any(capability["Task"] == task_name for capability in scenario["WizardCapabilities"])):
                dialogs.append(STAR.get_dialog(dialog_id))
        return dialogs

    @staticmethod
    def get_dialog_scenario(id):
        with open(os.path.join(STAR._path, f"dialogues/{id}.json")) as reader:
            return json.load(reader)["Scenario"]

    @staticmethod
    def get_dialog_first_turn(id, speaker: str = None):
        with open(os.path.join(STAR._path, f"dialogues/{id}.json")) as reader:
            for event in json.load(reader)["Events"]:
                turn_speaker = event["Agent"]
                if speaker == None and turn_speaker in STAR._speakers:
                        return Turn(speaker=turn_speaker, text=event["Text"])
                elif turn_speaker == speaker:
                    return Turn(speaker=turn_speaker, text=event["Text"])

    @staticmethod
    def get_dialog_task_names(id):
        scenario = STAR.get_dialog_scenario(id)
        return [task["Task"] for task in scenario["WizardCapabilities"]]

    @staticmethod
    def get_dialog_responses(id):
        tasks = STAR.get_dialog_task_names(id)
        return [STAR.read_graph_responses(task, as_dict=True) for task in tasks]

    @staticmethod
    def get_dialog_graphs(id):
        tasks = STAR.get_dialog_task_names(id)
        return [STAR.read_graph(task, as_dot=False) for task in tasks]

    @staticmethod
    def get_dialog_events(id):
        with open(os.path.join(STAR._path, f"dialogues/{id}.json")) as reader:
            return json.load(reader)["Events"]

    @staticmethod
    def get_dialog_events(id):
            with open(os.path.join(STAR._path, f"dialogues/{id}.json")) as reader:
                return json.load(reader)["Events"]

    @staticmethod
    def get_dialog_user_instructions(id):
        def get_user_n_turns_before(turn_ix, events):
            return len([e for e in events[:turn_ix]
                        if e["Agent"] == "User" and e["Action"] == "utter"])
        events = STAR.get_dialog_events(id)
        return {get_user_n_turns_before(ix, events): e["Text"]
                for ix, e in enumerate(events)
                if e["Action"] == "instruct" and e["Agent"] == "UserGuide"}

    @staticmethod
    def get_dialog_graphs_and_responses(id):
        return STAR.get_dialog_graphs(id), STAR.get_dialog_responses(id)

    @staticmethod
    def get_scenario_description(scenario):
        # Let's generate the graph description for each task:
        flowcharts = ""
        for task in scenario["WizardCapabilities"]:
            task_name = task["Task"]
            flowcharts += f"""
The graph for the task '{task_name}' with domain '{task['Domain']}' is:
```dot
{STAR.read_graph(task_name)}
```
and one example responses for each node is provided in the following json:
```json
{STAR.read_graph_responses(task_name)}
```

---
"""
        # Finally, let's return the scenario object and natural language description for it.
        return f"""The conversation is between a User and a AI assistant in the following domains: {', '.join(scenario['Domains'])}.

The User instructions are: {scenario['UserTask']}
The AI assistant instructions are: {scenario['WizardTask']}

In addition, the AI assistant is instructed to follow specific flowcharts to address the tasks. Flowcharts are defined as graph described using DOT.
The actual DOT for the current tasks are:
{flowcharts}

Finally, the following should be considered regarding the conversation:
   1. {"The conversation follows the 'happy path', meaning the conversations goes according to what it is described in the flowcharts"
       if scenario['Happy'] else
       "The conversation does NOT follow a 'happy path', meaning something happend to the user to change its mind or something happend "
       "in the environment for the conversation to not flow as expected, as described in the flowchart"}.
   2. {"The user is calling to perform multiple tasks, involving all the tasks defined as flowcharts above (" + ', '.join(task['Task'] for task in scenario['WizardCapabilities']) + ")"
        if scenario['MultiTask'] else
        "The user is calling to perform only the defined task (" + scenario['WizardCapabilities'][0]['Task'] + "), nothing else"}.
"""

    @staticmethod
    def get_dialog_scenario_description(id):
        scenario = STAR.get_dialog_scenario(id)
        return scenario, STAR.get_scenario_description(scenario)

    @staticmethod
    def get_user_persona_for_scenario(scenario):
        dialogue_details = f"""
The following should be considered regarding the conversation:
   1. {"The conversation follows a 'happy path', meaning the conversations goes smoothly without any unexpected behavior"
       if scenario['Happy'] else
       "The conversation does NOT follow a 'happy path', meaning you have to simulate something happend in the middle of the conversation, "
       "perhaps you changed your mind at some point or something external happend in the environment for the conversation to not flow as expected"}.
   2. {"The conversation involves multiple tasks, that is, you want the assistant to perform multiple tasks (" + ', '.join(task['Task'] for task in scenario['WizardCapabilities']) + "), not just one."
        if scenario['MultiTask'] else
        "The conversation involves only one task you were instructed to (" + scenario['WizardCapabilities'][0]['Task'] + "), nothing else"}"""

        return Persona(
            role=f"user calling a AI assistant that can perform multiple tasks in the following domains: {', '.join(scenario['Domains'])}.\n" + dialogue_details,
            circumstances=scenario["UserTask"],
        )

    @staticmethod
    def get_flowchart_description_for_scenario(scenario):
        flowcharts = ""
        for task in scenario["WizardCapabilities"]:
            task_name = task["Task"]
            flowcharts += f"""
## {task_name} ({task['Domain']})

The flowchart described as an action transition graph for the task '{task_name}' with domain '{task['Domain']}' is:
```dot
{STAR.read_graph(task_name)}
```
Response example for each action is provided in the following json:
```json
{STAR.read_graph_responses(task_name)}
```
where UPPERCASE words above are just example placeholders. You MUST fill in those with any coherent values in the actual conversation.
"""
        return flowcharts

    @staticmethod
    def get_system_persona_for_scenario(scenario):


        dialogue_details = f"""In the conversation, the AI assistant is instructed to follow specific action flowcharts to address the tasks. Flowcharts are defined as graph described using DOT.
The actual DOT for the current tasks are:
{STAR.get_flowchart_description_for_scenario(scenario)}
"""
        return Persona(
            role="AI assistant.\n" + dialogue_details,
            circumstances=scenario['WizardTask'],
        )

    @staticmethod
    def get_agents_for_scenario(scenario, model_name):
        user = PersonaAgent(model_name,
                            STAR.get_user_persona_for_scenario(scenario),
                            name="User",
                            can_finish=True)

        system = PersonaAgent(model_name,
                              STAR.get_system_persona_for_scenario(scenario),
                              name="System")

        return system, user

    @staticmethod
    def get_agents_from_dialogue(id, model_name:str, set_first_utterance: bool = False):
        scenario = STAR.get_dialog_scenario(id)
        system, user = STAR.get_agents_for_scenario(scenario, model_name)

        if set_first_utterance:
            first_turn = STAR.get_dialog_first_turn(id)
            if first_turn.speaker == "Wizard":
                system.set_first_utterances(first_turn.text)
            else:
                system.set_first_utterances("Hello, how can I help?")

        return system, user

    @staticmethod
    def get_agents_from_dialogue_with_orchestration(id, model_name:str, set_first_utterance: bool = False):
        system, user = STAR.get_agents_from_dialogue(id, model_name, set_first_utterance)

        graphs, responses = STAR.get_dialog_graphs_and_responses(id)
        response_action_orchestrator = SimpleResponseOrchestrator(responses[0], graph=graphs[0])
        instr_list_orchestrator = InstructionListOrchestrator(
            STAR.get_dialog_user_instructions(id),
            persistent=True
        )

        return system | response_action_orchestrator, user | instr_list_orchestrator
