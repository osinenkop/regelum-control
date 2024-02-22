from pathlib import Path
from distutils.dir_util import copy_tree, remove_tree

root = Path(__file__).parent.parent.parent

texs = root / "docs" / "tex" / "src"
include = root / "docs" / "include"

for tex_path in texs.iterdir():
    if tex_path.is_dir():
        if (tex_path / "mdhtml").exists():
            for path in (tex_path / "mdhtml").iterdir():
                if path.is_dir():
                    if (include / path.name).exists():
                        remove_tree(str((include / path.name).absolute()))
                    copy_tree(
                        str(path.absolute()), str((include / path.name).absolute())
                    )
