"""Rebasing deep zoom. Importing this package never imports CuPy."""

def deep_map(*args, **kw):
    from .perturb import deep_map as compute
    return compute(*args, **kw)


def deep_samples(*args, **kw):
    from .perturb import deep_samples as compute
    return compute(*args, **kw)


def plan(*args, **kw):
    from .perturb import plan as prepare
    return prepare(*args, **kw)


def launch(*args, **kw):
    from .perturb import launch as run
    return run(*args, **kw)
