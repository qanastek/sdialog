from sdialog.generators import DialogGenerator, PersonaDialogGenerator, LLMDialogOutput, Turn
from sdialog.personas import Persona, PersonaAgent


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


def test_dialog_generator(monkeypatch):
    monkeypatch.setattr("sdialog.generators.ChatOllama", DummyLLM)
    gen = DialogGenerator(MODEL, dialogue_details="test")
    dialog = gen()
    assert hasattr(dialog, "turns")


def test_persona_dialog_generator(monkeypatch):
    monkeypatch.setattr("sdialog.generators.ChatOllama", DummyLLM)
    persona_a = Persona(name="A")
    persona_b = Persona(name="B")
    gen = PersonaDialogGenerator(MODEL, persona_a, persona_b)
    dialog = gen()
    assert hasattr(dialog, "turns")


def test_persona_dialog_generator_personas(monkeypatch):
    monkeypatch.setattr("sdialog.generators.ChatOllama", DummyLLM)
    persona_a = Persona(name="A")
    persona_b = Persona(name="B")
    gen = PersonaDialogGenerator(MODEL, persona_a, persona_b)
    dialog = gen()
    assert "A" in dialog.personas
    assert "B" in dialog.personas


def test_persona_dialog_generator_with_agents(monkeypatch):
    monkeypatch.setattr("sdialog.generators.ChatOllama", DummyLLM)
    persona_a = PersonaAgent(DummyLLM(), name="A")
    persona_b = PersonaAgent(DummyLLM(), name="B")
    gen = PersonaDialogGenerator(MODEL, persona_a, persona_b)
    dialog = gen()
    assert hasattr(dialog, "turns")
    assert "A" in dialog.personas
    assert "B" in dialog.personas
