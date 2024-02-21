from pathlib import Path
import mkdocs_gen_files

root = Path(__file__).parent.parent.parent

mkdocs_overrides_dir = root / "docs" / "overrides"
preamble_file = root / "docs" / "tex" / "preambleAIDA_ML__Nov2023.tex"

with mkdocs_gen_files.open(preamble_file, "r") as f:
    preamble = f.read()


with mkdocs_gen_files.open(mkdocs_overrides_dir / "main.html", "w") as f:
    f.write(
        r"""{% extends "base.html" %}

{% block content %}
    <!-- latex preamble --> <div class='arithmatex', style="display:none"> 
    \(
    <!-- latex preamble should be inserted here below between raw ... endraw jinja tags -->
    {% raw %}"""
        + preamble
        + r"""
    {% endraw %} 
    \)
    </div>
  {{ super() }}

{% endblock %}
"""
    )
