# TODO

- [x] Add initial API documentation to all classes and functions
- [x] Expand user documentation and examples
- [x] Improve README and add usage guides
- [x] Generate API documentation
- [x] Add initial unit tests
- [ ] Integrate with LangChain’s `ChatHuggingFace` for more LLM options
- [ ] Enable exporting raw LLM messages and internal memory states
- [ ] As with `Persona`, define a `BaseScenario` and a `Scenario` classes
- [ ] Add `InstructAtTurnsOrchestrator` to provide instructions at given turns (`{3: "do this", 7: "do that"}`)
- [ ] Allow `PersonaGenerator` to also take as input agents (and call `agent_a.talk_with(agent_b)` under the hood)
