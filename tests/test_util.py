import pytest

from sdialog.util import make_serializable


def test_make_serializable_dict():
    d = {"a": 1, "b": [1, 2], "c": {"d": 3}}
    make_serializable(d)
    assert isinstance(d, dict)
    assert isinstance(d["b"], list)
    assert isinstance(d["c"], dict)


def test_make_serializable_non_dict():
    lt = [1, 2, 3]
    with pytest.raises(TypeError):
        make_serializable(lt)
