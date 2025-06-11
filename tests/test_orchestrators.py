from sdialog.orchestrators import (
    BaseOrchestrator,
    LengthOrchestrator,
    ChangeMindOrchestrator,
    SimpleReflexOrchestrator,
    SimpleResponseOrchestrator,
    InstructionListOrchestrator
)


def test_base_orchestrator_instruct():
    class DummyOrch(BaseOrchestrator):
        def instruct(self, dialog, utterance):
            return "Do something"
    orch = DummyOrch()
    assert orch.instruct([], "") == "Do something"


def test_length_orchestrator():
    orch = LengthOrchestrator(min=2, max=4)
    assert hasattr(orch, "instruct")
    assert callable(orch.instruct)


def test_change_mind_orchestrator():
    orch = ChangeMindOrchestrator(probability=1.0, reasons=["reason"], max_times=1)
    assert hasattr(orch, "instruct")
    assert callable(orch.instruct)


def test_simple_reflex_orchestrator():
    orch = SimpleReflexOrchestrator(condition=lambda utt: "problem" in utt, instruction="Apologize")
    assert orch.instruct([], "problem here") == "Apologize"


def test_simple_response_orchestrator():
    orch = SimpleResponseOrchestrator(responses=["Yes", "No"])
    assert hasattr(orch, "instruct")


def test_instruction_list_orchestrator():
    orch = InstructionListOrchestrator(["Step 1", "Step 2"])
    assert hasattr(orch, "instruct")
