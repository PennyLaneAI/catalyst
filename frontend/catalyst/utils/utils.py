import inspect
import typing


def are_params_annotated(f: typing.Callable):
    signature = inspect.signature(f)
    parameters = signature.parameters
    return all(p.annotation is not inspect.Parameter.empty for p in parameters.values())


def get_type_annotations(func: typing.Callable):
    params_are_annotated = are_params_annotated(func)
    if params_are_annotated:
        return getattr(func, "__annotations__", {}).values()

    return None
