# TODO

- [x] Add initial API documentation to all classes and functions
- [x] Expand user documentation and examples
- [x] Improve README and add usage guides
- [x] Generate API documentation
- [x] Add initial unit tests
- [x] Allow `PersonaGenerator` to also take as input agents (and call `agent_a.talk_with(agent_b)` under the hood)
- [x] Add library version used as part of the Dialog metadata.
- [x] Add `InstructionListOrchestrator` to provide instructions at given turns (`{3: "do this", 7: "do that"}`)
- [ ] Integrate with LangChain’s `ChatHuggingFace` for more LLM options
- [ ] Move default now hard-coded prompts to config files that support prompt template definition with optional fields as in Ollama templates using [jinja template](https://jinja.palletsprojects.com/en/stable/templates/) (e.g. [here](https://ollama.com/library/deepseek-r1:latest/blobs/c5ad996bda6e))
- [ ] Enable exporting raw LLM messages and internal memory states
- [ ] As with `Persona`, define a `BaseScenario` and a `Scenario` classes
- [ ] Add integration with Dialog2Flow for dialog flow visualization.
- [ ] Add (optional) TTS support.
- [ ] Improve the coverage of unit tests to be above 90%.
