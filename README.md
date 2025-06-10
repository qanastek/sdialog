# SDialog
[![Documentation Status](https://app.readthedocs.org/projects/sdialog/badge/?version=latest)](https://sdialog.readthedocs.io)
[![PyPI version](https://badge.fury.io/py/sdialog.svg)](https://badge.fury.io/py/sdialog)
[![Downloads](https://static.pepy.tech/badge/sdialog)](https://pepy.tech/project/sdialog)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/idiap/sdialog/master?filepath=tutorials)
<!-- [![Build Status](https://api.travis-ci.com/sergioburdisso/sdialog.svg?branch=master)](https://app.travis-ci.com/github/idiap/sdialog) -->
<!-- [![codecov](https://codecov.io/gh/idiap/sdialog/branch/master/graph/badge.svg)](https://codecov.io/gh/idiap/sdialog) -->

**SDialog** is a modular, extensible Python toolkit for synthetic dialogue generation and analysis, designed for research and development with instruction-tuned Large Language Models (LLMs). It enables flexible, persona-driven, multi-agent dialogue simulation, orchestration, and scenario management, making it ideal for building, evaluating, and experimenting with conversational agents.

## 🚀 Motivation

Modern conversational AI research and applications increasingly require high-quality, flexible, and reproducible synthetic dialogues for training, evaluation, and benchmarking. SDialog addresses the need for:

- **Standardization:** Clear definitions for dialogue, persona, and event structures.
- **Abstraction:** Abstract interfaces for both single-agent and multi-agent dialogue generation.
- **Fine-grained Control:** Orchestration to inject instructions, simulate user behaviors, and enforce scenario constraints.
- **LLM Integration:** Seamless integration with instruction-tuned LLMs, prompt management, and memory handling.
- **Scenario and Dataset Management:** Tools for managing complex scenarios, flowcharts, and persona definitions.

## ✨ Features

- **Persona-based Role-Playing:** Define rich agent personas to simulate realistic conversations.
- **Multi-Agent Dialogue:** Generate dialogues between multiple agents, each with their own persona and behavior.
- **Dialogue Orchestration:** Control agent actions and inject instructions dynamically using orchestrators.
- **Scenario Management:** Easily describe and manage dialogue scenarios, including flowcharts and user/system goals.
- **Flexible Serialization:** Export dialogues and events in JSON or plain text for downstream tasks.
- **Integration with LLMs:** Out-of-the-box support for [Ollama](https://ollama.com/) and [LangChain](https://python.langchain.com/), with planned support for HuggingFace models.

## ⚡ Installation

```bash
pip install sdialog
```

> **Note:** You must have [Ollama](https://ollama.com/download) running on your system to use the default LLM integration.
> ```bash
> curl -fsSL https://ollama.com/install.sh | sh
> ```

## 🏁 Quick Start

Define personas, create agents, and generate a dialogue:

```python
from sdialog import Persona, PersonaAgent

# Define personas
alice = Persona(name="Alice", role="friendly barista", personality="cheerful and helpful")
bob = Persona(name="Bob", role="customer", personality="curious and polite")

# Create agents
alice_agent = PersonaAgent("llama2", persona=alice, name="Alice")
bob_agent = PersonaAgent("llama2", persona=bob, name="Bob")

# Generate a dialogue
dialog = alice_agent.dialog_with(bob_agent)
dialog.print()
```

## 🎛️ Orchestration Example

Add orchestration to control dialogue length or simulate agent behaviors:

```python
from sdialog.orchestrators import LengthOrchestrator, ChangeMindOrchestrator

length_orch = LengthOrchestrator(min=3, max=6)
mind_orch = ChangeMindOrchestrator(probability=0.5, reasons=["changed plans", "new information"], max_times=1)
alice_agent = alice_agent | length_orch | mind_orch
```

## 📚 STAR Dataset Integration

Work with the STAR dataset for scenario-driven dialogue generation:

```python
from sdialog.datasets import STAR

STAR.set_path("/path/to/star-dataset")
dialog = STAR.get_dialog(123)
dialog.print(scenario=True)
```

## 📖 Documentation

- **[Documentation](https://sdialog.readthedocs.io)** - Full package documentation, including installation, API reference, usage guides, and advanced examples available.
- **[API Reference](https://sdialog.readthedocs.io/en/latest/api/index.html):** See docstrings in the codebase for detailed documentation of all classes and functions.
- **[Tutorials](https://github.com/idiap/sdialog/tree/main/tutorials):** Tutorials for hands-on examples as Jupyter Notebooks.


## 🙏 Acknowledgments

This work was supported by the EU Horizon 2020 project [ELOQUENCE](https://eloquenceai.eu/) (grant number 101070558).

This work was also initially created in preparation for the 2025 Jelinek Memorial Summer Workshop on Speech and Language Technologies ([JSALT 2025](https://jsalt2025.fit.vut.cz/)) as part of the work done by the ["Play your Part" research group](https://jsalt2025.fit.vut.cz/play-your-part).


## 📝 License

MIT License  
Copyright (c) 2025 Idiap Research Institute
