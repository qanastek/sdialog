# SDialog

Synthetic Dialogue Generation and Analysis.

## Tutorials

Make sure you have [`Ollama`](https://ollama.com/download) running is your system before going through our Jupyter Notebooks:
1. [Single-LLM Full Dialogue Generation.](1.single_llm_full_generation.ipynb)
2. [Role-Playing Multi-Agent Dialogue Generation.](2.multi-agent_generation.ipynb)
3. [Multi-Agent Dialogue Generation with Orchestration.](3.multi-agent+orchestrator_generation.ipynb)

## TODO

- [ ] Add initial API documentation comment to all the classes and functions
- [ ] Add initial documentation
- [ ] Write a good initial `README.md` (use JSALT [`README.md`](https://github.com/Play-Your-Part/tutorials) too)
- [ ] Add integration with LangChain’s [`ChatHuggingFace`](https://python.langchain.com/docs/integrations/chat/huggingface/) to allow access to low-level features (e.g. allowing integration with [`TransformerLens`](https://github.com/TransformerLensOrg/TransformerLens)).
- [ ] Allow dumping the full low-level messages sent to the LLM (prompt, messages, special symbols, etc. the raw LLM input)
- [ ] Add function to export/generate internal memory object for given `dialog.events` to a certain point (i.e. from events -> memory conversation).
- [ ] Add initial unit tests

## License

MIT License

Copyright (c) 2025 Idiap Research Institute
