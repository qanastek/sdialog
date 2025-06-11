from sdialog.personas import Persona, PersonaAgent, BasePersona
from sdialog.generators import LLMDialogOutput, Turn
from sdialog import Dialog

MODEL = "smollm:135m"


# Patch LLM call
class DummyLLM:
    seed = 0
    num_predict = 1

    def __init__(self, *a, **kw):
        pass

    def invoke(self, memory):
        return type(
            "Msg", (),
            {"content": LLMDialogOutput(
                dialog=[Turn(speaker="A", text="Hi")]).model_dump_json()}
        )()

    def __str__(self):
        return "dummy"


def test_base_persona_description_and_json():
    p = BasePersona(name="Test", role="tester")
    desc = p.description()
    assert "Test" in desc
    js = p.json()
    assert isinstance(js, dict)
    js_str = p.json(string=True)
    assert isinstance(js_str, str)


def test_persona_fields():
    p = Persona(name="Alice", role="barista", background="Cafe")
    assert p.name == "Alice"
    assert p.role == "barista"
    assert p.background == "Cafe"


def test_persona_agent_init(monkeypatch):
    persona = Persona(name="Alice")
    agent = PersonaAgent(DummyLLM(), persona=persona, name="Alice")
    assert agent.get_name() == "Alice"
    assert "Role play" in agent.get_prompt()
    agent.set_first_utterances("Hi!")
    assert agent.first_utterances == "Hi!"
    agent.clear_orchestrators()
    agent.reset(seed=42)


def test_persona_and_json():
    persona = Persona(name="Alice", role="barista", background="Works at a cafe")
    desc = persona.description()
    assert "Alice" in desc
    js = persona.json()
    assert isinstance(js, dict)
    js_str = persona.json(string=True)
    assert isinstance(js_str, str)
    assert "Alice" in js_str


def test_persona_agent_init_and_prompt():
    persona = Persona(name="Alice", role="barista")
    agent = PersonaAgent(MODEL, persona=persona, name="Alice")
    assert agent.get_name() == "Alice"
    prompt = agent.get_prompt()
    assert "Role play" in prompt


def test_persona_agent_dialog_with():
    persona1 = Persona(name="A")
    persona2 = Persona(name="B")
    agent1 = PersonaAgent(DummyLLM(), persona=persona1, name="A")
    agent2 = PersonaAgent(DummyLLM(), persona=persona2, name="B")
    dialog = agent1.dialog_with(agent2, max_iterations=2, keep_bar=False)
    assert isinstance(dialog, Dialog)
    assert len(dialog.turns) > 0
