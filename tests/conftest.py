"""Keep pytest temporary files separate across users and test runs."""

import shutil
from uuid import uuid4


def pytest_configure(config):
    # A fixed basetemp is deleted by pytest and can be owned by another user.
    # Respect an explicitly supplied --basetemp, including in CI.
    if config.option.basetemp is None:
        config._project_test_basetemp = config.rootpath / f".pytest-tmp-{uuid4().hex}"
        config.option.basetemp = str(config._project_test_basetemp)


def pytest_unconfigure(config):
    # Only remove the directory generated for this run, never a user-supplied one.
    path = getattr(config, "_project_test_basetemp", None)
    if path is not None:
        shutil.rmtree(path, ignore_errors=True)
