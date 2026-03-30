import pytest
from brickbybrick_sonnet.py_component import PyComponent


@pytest.fixture()
def py_component(ros_context):
    yield PyComponent("py_component")


def test_construction(py_component):
    assert py_component.get_name() == "py_component"