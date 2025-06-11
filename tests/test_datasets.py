from sdialog.datasets import STAR


def test_star_import():
    assert STAR is not None


def test_star_set_path_and_get_dialog(monkeypatch):
    # Patch internal loading to avoid file I/O
    monkeypatch.setattr(STAR, "set_path", lambda path: None)
    monkeypatch.setattr(STAR, "get_dialog", lambda idx: type("Dialog", (), {"print": lambda self, **kw: None})())
    STAR.set_path("/tmp")
    dialog = STAR.get_dialog(123)
    assert dialog is not None
    dialog.print()


def test_star_get_agents_for_scenario(monkeypatch):
    # Patch to return dummy agents
    monkeypatch.setattr(STAR, "get_agents_for_scenario", lambda scenario, model: ("sys", "usr"))
    sys, usr = STAR.get_agents_for_scenario({}, "llama2")
    assert sys == "sys"
    assert usr == "usr"
