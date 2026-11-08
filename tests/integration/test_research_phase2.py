import pytest

from scripts.run_research_phase2 import run, staging_directory


def test_staging_is_cleaned_after_use(tmp_path):
    (tmp_path / "results").mkdir()
    with staging_directory(tmp_path) as path:
        (path / "sample.txt").write_text("test", encoding="utf-8")
        assert path.exists()
    assert not path.exists()


def test_existing_research_report_is_not_overwritten(tmp_path):
    report = tmp_path / "results/research_phase2"
    report.mkdir(parents=True)
    existing = report / "comparison.json"
    existing.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run(tmp_path)
    assert existing.read_text(encoding="utf-8") == "existing"
