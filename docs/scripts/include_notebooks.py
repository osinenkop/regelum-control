from pathlib import Path
from nbconvert import MarkdownExporter
import nbformat
import base64

root = Path(__file__).parent.parent.parent

notebooks = root / "docs" / "notebooks"
out = root / "docs" / "src" / "notebooks"


for nb_path in notebooks.glob("*.ipynb"):
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
        fp.write(body)
