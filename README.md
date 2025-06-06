# SDialog

**SDialog** is a Python toolkit for synthetic dialogue generation and analysis, designed for research and development with instruction-tuned Large Language Models (LLMs). It enables flexible, role-playing-based dialogue simulation, orchestration, and scenario management, making it ideal for building, evaluating, and experimenting with conversational agents.

## Features

- **Persona-based Role-Playing:** Define rich agent personas to simulate realistic conversations.
- **Multi-Agent Dialogue:** Generate dialogues between multiple agents, each with their own persona and behavior.
- **Dialogue Orchestration:** Control agent actions and inject instructions dynamically using orchestrators.
- **Scenario Management:** Easily describe and manage dialogue scenarios, including flowcharts and user/system goals.
- **Flexible Serialization:** Export dialogues and events in JSON or plain text for downstream tasks.
- **Integration with LLMs:** Out-of-the-box support for [Ollama](https://ollama.com/) and [LangChain](https://python.langchain.com/), with planned support for HuggingFace models.

## Installation

```bash
pip install sdialog
```

> **Note:** You must have [Ollama](https://ollama.com/download) running on your system to use the default LLM integration.

## Quick Start

```python
from sdialog import Persona, PersonaAgent

# Define personas
alice = Persona(name="Alice", role="friendly barista", personality="cheerful and helpful")
bob = Persona(name="Bob", role="customer", personality="curious and polite")

# Create agents
alice_agent = PersonaAgent("llama2", persona=alice, name="Alice")
bob_agent = PersonaAgent("llama2", persona=bob, name="Bob")

# Generate a dialogue
dialog = alice_agent.dialog_with(bob_agent, max_iterations=10)
dialog.print()
```

## Tutorials

Explore our Jupyter Notebooks for hands-on examples:

1. [Single-LLM Full Dialogue Generation](https://github.com/idiap/sdialog/blob/main/tutorials/1.single_llm_full_generation.ipynb)
2. [Role-Playing Multi-Agent Dialogue Generation](https://github.com/idiap/sdialog/blob/main/tutorials/2.multi-agent_generation.ipynb)
3. [Multi-Agent Dialogue Generation with Orchestration](https://github.com/idiap/sdialog/blob/main/tutorials/3.multi-agent+orchestrator_generation.ipynb)

## Documentation

- **API Reference:** See docstrings in the codebase for detailed documentation of all classes and functions.
- **Scenarios & Orchestration:** Easily define complex scenarios and control agent behavior using orchestrators.
- **Exporting Dialogues:** Save dialogues as JSON or text for further analysis or training.

## License

MIT License  
Copyright (c) 2025 Idiap Research Institute

---

For questions or contributions, please open an issue or pull request on [GitHub](https://github.com/idiap/sdialog).
