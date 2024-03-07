import pennylane as qml
import catalyst
from catalyst.pennylane_extensions import callback
import pytest


def test_callback_print(capsys):
    def my_callback():
        print("Hello erick")

    @qml.qjit(keep_intermediate=True)
    def foo():
        callback(my_callback, [])
        return None

    foo()
    captured = capsys.readouterr()
    assert captured.out == "Hello erick\n"
