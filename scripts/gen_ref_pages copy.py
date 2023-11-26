from pathlib import Path

import mkdocs_gen_files

import ast
from pathlib import Path
from typing import Dict, List

from pathlib import Path
from typing import Dict, List
import os


class ClassCollector(ast.NodeVisitor):
    def __init__(self):
        self.classes = []

    def visit_ClassDef(self, node: ast.ClassDef):
        self.classes.append(node.name)


def get_classes_from_file(file_path: Path) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as file:
        node = ast.parse(file.read(), filename=str(file_path))
    collector = ClassCollector()
    collector.visit(node)
    return collector.classes


def collect_classes(package_path: Path) -> Dict[str, List[str]]:
    classes_dict = {}
    for file_path in package_path.rglob("*.py"):
        # Exclude __init__.py files as they typically do not contain class definitions
        if file_path.name != "__init__.py":
            classes = get_classes_from_file(file_path)
            if classes:
                # Convert file path to dotted path notation
                module_path = "/".join(
                    file_path.with_suffix("").parts[len(package_path.parts) :]
                )
                classes_dict[module_path] = classes
    return classes_dict


src = Path(__file__).parent.parent.parent / "regelum"
class_data = collect_classes(src)
nav = mkdocs_gen_files.Nav()


# classes = collect_classes_with_griffe(src)

for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        if len(parts) > 1:
            parts = parts[:-1]
        else:
            continue
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")

        nav[parts] = doc_path.as_posix()

        with mkdocs_gen_files.open(full_doc_path, "w") as fd:
            ident = ".".join(parts)
            fd.write(f"::: {ident}")

        mkdocs_gen_files.set_edit_path(full_doc_path, path)
    elif module_path.as_posix() in class_data:
        for class_name in class_data[module_path.as_posix()]:
            if class_name in [
                "MethodCallback",
                "PassdownCallback",
                "MetaClassDecorator",
            ]:
                continue
            curr_parts = parts + (class_name,)
            full_path = full_doc_path.with_suffix("") / (class_name + ".md")
            print(full_path)
            print(curr_parts)
            nav[curr_parts] = doc_path.with_suffix("") / (class_name + ".md")
            with mkdocs_gen_files.open(full_path, "w") as fd:
                ident = ".".join(curr_parts)
                fd.write(f"::: {ident}")
            mkdocs_gen_files.set_edit_path(full_path, path)


with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
