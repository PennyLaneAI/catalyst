import pytest

from catalyst.jit import make_positional_wrapper


def test_no_args():
    def foo():
        return 4

    wrapper = make_positional_wrapper(foo)

    assert wrapper() == 4

    with pytest.raises(TypeError):
        wrapper(4)


def test_positional_only_args():
    def foo(x, /):
        return 2 * x

    wrapper = make_positional_wrapper(foo)

    assert wrapper(1) == 2
    assert wrapper(10) == 20

    with pytest.raises(TypeError):
        wrapper(x=4)


def test_positional_or_keyword_args():
    def foo(x):
        return x // 2

    wrapper = make_positional_wrapper(foo)

    assert wrapper(2) == 1
    with pytest.raises(TypeError):
        wrapper(x=4)


def test_var_positional_args():
    def foo(*args):
        return sum(args)

    wrapper = make_positional_wrapper(foo)

    assert wrapper((1, 2, 3)) == 6

    with pytest.raises(TypeError):
        wrapper(1, 2, 3)


def test_keyword_only_args():
    def foo(*, x=1):
        return x + 1

    wrapper = make_positional_wrapper(foo)

    assert wrapper(2) == 3

    with pytest.raises(TypeError):
        wrapper(x=3)


def test_var_keyword_args():
    def foo(**kwargs):
        return kwargs

    wrapper = make_positional_wrapper(foo)

    assert wrapper({"x": 1, "y": 3.2}) == {"x": 1, "y": 3.2}

    with pytest.raises(TypeError):
        wrapper(x=1, y=3.2)
