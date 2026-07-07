import json
from pathlib import Path


NOTEBOOKS = sorted(Path("notebooks").glob("*.ipynb"))


def test_notebooks_are_valid_and_output_free():
    assert NOTEBOOKS
    for path in NOTEBOOKS:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell.get("outputs", []) == []
                assert cell.get("execution_count") is None


def test_notebooks_have_colab_badges_and_setup_cells():
    for path in NOTEBOOKS:
        text = path.read_text(encoding="utf-8")
        assert "colab.research.google.com/github/marcoharuni/forge-tokenizer" in text
        assert "%pip install -q" in text
        assert "git clone https://github.com/marcoharuni/forge-tokenizer.git" in text


def test_notebooks_avoid_noisy_bootstrap_blocks():
    for path in NOTEBOOKS:
        text = path.read_text(encoding="utf-8")
        assert "subprocess.check_call" not in text
        assert "repo_root =" not in text
