# TODO

- [x] Add initial API documentation to all classes and functions
- [x] Expand user documentation and examples
- [x] Improve README and add usage guides
- [x] Generate API documentation
- [x] Add initial unit tests
- [x] Allow `PersonaGenerator` to also take as input agents (and call `agent_a.talk_with(agent_b)` under the hood)
- [x] Add library version used as part of the Dialog metadata.
- [ ] Move default now hard-coded prompts to config files that support prompt template definition with optional fields as in Ollama templates (e.g. [here](https://ollama.com/library/deepseek-r1:latest/blobs/c5ad996bda6e))
- [ ] Integrate with LangChain’s `ChatHuggingFace` for more LLM options
- [ ] Enable exporting raw LLM messages and internal memory states
- [ ] As with `Persona`, define a `BaseScenario` and a `Scenario` classes
- [ ] Add `InstructAtTurnsOrchestrator` to provide instructions at given turns (`{3: "do this", 7: "do that"}`)
- [ ] Add integration with Dialog2Flow for dialog flow visualization.
- [ ] Improve the coverage of unit tests to be above 90%.
