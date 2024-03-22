from pathlib import Path
from nbconvert import MarkdownExporter
import nbformat
import base64
import json
from git import Repo
import mkdocs_gen_files

colab_repo_url = f"https://colab.research.google.com/github/osinenkop/regelum-control/blob/{Repo('.').active_branch.name}/docs/notebooks/collabs"
root = Path(__file__).parent.parent.parent
preamble_file = root / "docs" / "tex" / "preambleAIDA_ML__Nov2023.tex"
notebooks = root / "docs" / "notebooks"
out = root / "docs" / "src" / "notebooks"

prelude = """
!!! tip annotate "Run This Tutorial in Google Colab"

    This document is automatically generated from a Jupyter notebook. You have the option to open it in Google Colab to interactively run and explore the code in each cell.

"""


def generate_colab_tag(nb_name):
    return f"""
    <a target="_blank" href="{colab_repo_url}/{nb_name}">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
    </a>
"""


def get_preamble_text():
    with mkdocs_gen_files.open(preamble_file, "r") as f:
        preamble = f.read()
    return ["$$\n"] + [line + "\n" for line in preamble.split("\n")] + ["$$\n\n"]


def remove_preamble_from_md_text(md_text: str):
    pos_first = md_text.find("$$")

    if pos_first >= 0:
        pos_second = md_text.find("$$", pos_first + 2)

        if "command" in md_text[pos_first : pos_second + 2]:
            return md_text.replace(md_text[pos_first : pos_second + 2], "")

    return md_text


def transform_notes(content):
    final_lines = []
    is_previous_line_a_note = False
    for line in content.split("\n"):
        if line.replace(" ", "").startswith(">__Note__"):
            final_lines.append("!!! note")
            is_previous_line_a_note = True
        elif line.replace(" ", "").startswith(">") and is_previous_line_a_note:
            final_lines.append(line.replace(">", "    "))
            is_previous_line_a_note = True
        else:
            final_lines.append(line)
            is_previous_line_a_note = False

    return "\n".join(final_lines)


for nb_path in notebooks.iterdir():
    if not str(nb_path).endswith(".ipynb"):
        continue

    # Create collab version of the notebook

    jupyter_json = json.loads(nb_path.read_text())
    # Remove preamble if exists
    if jupyter_json["cells"][0]["cell_type"] == "markdown" and any(
        ["command" in line for line in jupyter_json["cells"][0]["source"]]
    ):
        jupyter_json["cells"] = jupyter_json["cells"][1:]
        ipynb_preamble = get_preamble_text()
    else:
        ipynb_preamble = []

    jupyter_json["cells"] = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ipynb_preamble
            + [
                "Please ensure that the following requirements are installed prior to executing the cells within the Jupyter notebook\n",
            ],
        },
        {
            "cell_type": "code",
            "source": ["!pip install regelum-control"],
            "metadata": {},
            "execution_count": None,
            "outputs": [],
        },
    ] + jupyter_json["cells"]
    (nb_path.parent / "collabs" / nb_path.name).write_text(json.dumps(jupyter_json))

    # TRANSFORM TO MARKDOWN
    with open(nb_path, encoding="utf-8") as fp:
        nb = nbformat.read(fp, nbformat.NO_CONVERT)

    exporter = MarkdownExporter()
    (body, resources) = exporter.from_notebook_node(nb)
    out.mkdir(parents=True, exist_ok=True)

    for resource_name, resource_data in resources["outputs"].items():
        if resource_name.endswith(".png"):
            body = body.replace(
                "(" + resource_name + ")",
                "(data:image/png;base64, "
                + base64.b64encode(resource_data).decode()
                + ")",
            )

    with open(out / nb_path.name.replace(".ipynb", ".md"), "w", encoding="utf-8") as fp:
        body = (
            prelude
            + generate_colab_tag(nb_name=nb_path.name)
            + transform_notes(remove_preamble_from_md_text(body))
        )
        fp.write(body)
