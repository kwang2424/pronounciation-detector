import pytest


@pytest.fixture(autouse=True)
def _isolated_history(monkeypatch, tmp_path_factory):
    """No test may read or write the learner's real ~/.mdd history."""
    home = tmp_path_factory.mktemp("mdd-home")
    monkeypatch.setenv("MDD_REVIEW", str(home / "review.json"))
    monkeypatch.setenv("MDD_PROGRESS", str(home / "progress.json"))
    monkeypatch.setenv("MDD_TESTSET", str(home / "testset"))
