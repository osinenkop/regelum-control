import typer
from pathlib import Path
import bs4
from typing import Annotated
import numpy as np
from typing_extensions import Literal
from bs4 import Comment
import re


def read(html: Path) -> bs4.BeautifulSoup:
    return bs4.BeautifulSoup(html.read_text(), "html.parser")


def get_path(html: Path, out: Path = None) -> Path:
    return html if out is None else out / html.name


def find_main_html(html: Path) -> Path:
    if html.is_dir():
        html_files_in_dir = list(html.glob("*.html"))
        id_min_len = np.argmin([len(str(h)) for h in html_files_in_dir])
        html = html_files_in_dir[id_min_len]
        return html
    else:
        raise ValueError("Input path is not a directory")


def get_table_of_contents(html: Path) -> list[str]:
    bs = read(html)
    bs.find("div", {"class": "tableofcontents"})

    return [
        link.get("href").split("#")[0]
        for link in (
            bs.find("body")
            .find("div", {"class": "tableofcontents"})
            .find_all("a", href=True)
        )
    ]


def transform_to_md_heading(
    heading_type: Literal["h1", "h2", "h3", "h4", "h5", "h6"]
) -> str:
    return {
        "h1": "#",
        "h2": "##",
        "h3": "##",
        "h4": "###",
        "h5": "####",
        "h6": "#####",
    }[heading_type]


def save(bs: bs4.BeautifulSoup, html: Path, out: Path = None, help="") -> None:
    path = get_path(html, out)
    print(help, "Writing to", path)

    with open(path, "w", encoding="utf-8") as file:
        file.write(str(bs))


rm_heading_app = typer.Typer()


@rm_heading_app.callback(invoke_without_command=True)
def rm_heading(
    html: Path = typer.Argument(),
    out: Annotated[Path, typer.Option()] = None,
) -> None:
    bs = read(html)
    heading_tags = ["h1", "h2", "h3", "h4", "h5", "h6"]
    all_headings = bs.find("body").find_all(heading_tags)
    if len(all_headings) >= 1:
        if len(all_headings) > 1:
            html_name_metatag = bs.find("meta", {"name": "src"})
            html_name = html_name_metatag.get("content") if html_name_metatag else ""
            print("Warning: more than one heading found in", html_name)
        first_heading = all_headings[0]
        first_heading.text.strip()
        first_heading_type, first_heading_txt = (
            first_heading.name,
            first_heading.text.strip(),
        )

        first_heading.decompose()
        md_heading_tag = bs.new_tag("meta")
        md_heading_tag.attrs["name"] = "md-heading"
        md_heading_tag.attrs["md-heading"] = first_heading_txt
        md_heading_tag.attrs["type"] = first_heading_type

        bs.head.append(md_heading_tag)
    save(bs, html, out, help="  - Removed heading.")


rm_crosslinks_app = typer.Typer()


@rm_crosslinks_app.callback(invoke_without_command=True)
def rm_crosslinks(
    html: Annotated[Path, typer.Argument()],
    out: Annotated[Path, typer.Option()] = None,
):
    bs = read(html)
    for cl in bs.find("body").find_all("div", {"class": "crosslinks"}):
        cl.decompose()
    save(bs, html, out, help="  - Removed crosslinks.")


rm_toc_app = typer.Typer()


@rm_toc_app.callback(invoke_without_command=True)
def rm_toc(
    html: Annotated[Path, typer.Argument()],
    out: Annotated[Path, typer.Option()] = None,
):
    bs = read(html)
    for toc in bs.find("body").find_all("div", {"class": "subsectionTOCS"}):
        toc.decompose()

    for toc in bs.find("body").find_all("div", {"class": "sectionTOCS"}):
        toc.decompose()
    save(bs, html, out, help="  - Removed TOC.")


rm_stylesheet_app = typer.Typer()


@rm_stylesheet_app.callback(invoke_without_command=True)
def rm_stylesheet(
    html: Annotated[Path, typer.Argument()],
    out: Annotated[Path, typer.Option()] = None,
):
    bs = read(html)
    for stylesheet in bs.find("head").find_all(
        "link", {"rel": "stylesheet", "type": "text/css"}
    ):
        stylesheet.href = "texhtml.css"

    save(bs, html, out, help="  - Fixed stylesheet.")


fix_links_app = typer.Typer()


@fix_links_app.callback(invoke_without_command=True)
def fix_links(
    html: Annotated[Path, typer.Argument()],
    out: Annotated[Path, typer.Option()] = None,
):
    bs = read(html)
    for link in bs.find("body").find_all("a", href=True):
        if ".html#" in link["href"]:
            link["href"] = link["href"][link["href"].find(".html#") + 5 :]
    save(bs, html, out, help="  - Fixed links.")


fix_refs_app = typer.Typer()


@fix_refs_app.callback(invoke_without_command=True)
def fix_refs(
    html: Annotated[Path, typer.Argument()],
    out: Annotated[Path, typer.Option()] = None,
):
    bs = read(html)
    for href in bs.find_all("a", href=True):
        comment = href.find(string=lambda text: isinstance(text, Comment))
        if comment is not None:
            if "tex4ht:ref:" in comment:
                latex_ref = (
                    comment.replace("tex4ht:ref:", "\\ref{").replace(" ", "") + "}"
                )
                if "{eq" in latex_ref:
                    href.replace_with(latex_ref)

    # This block of code replaces all (\ref{some-random-label}) to \eqref{some-random-label}
    bs = bs4.BeautifulSoup(
        re.sub(r"\(\\ref\{([^\}]+)\}\)", r"\\eqref{\1}", str(bs)), "html.parser"
    )

    save(bs, html, out, help="  - Fixed refs.")


process_app = typer.Typer()


@process_app.callback(invoke_without_command=True)
def process(
    html: Annotated[
        Path, typer.Argument(help="HTML file or directory with HTML files to process")
    ],
    out: Annotated[Path, typer.Option()] = None,
):
    if html.is_dir():
        for html_file in html.glob("*.html"):
            process(html_file, out)
    else:
        print("Processing", html)
        fix_refs(html, out)
        path = get_path(html, out)
        rm_heading(path, out)
        rm_stylesheet(path, out)
        rm_crosslinks(path, out)
        rm_toc(path, out)
        fix_links(path, out)


md_app = typer.Typer()


@md_app.callback(invoke_without_command=True)
def markdown(
    html: Annotated[Path, typer.Argument(help="Directory with HTML files to process")],
    out: Annotated[Path, typer.Option()],
):
    main_html = find_main_html(html)
    toc = get_table_of_contents(main_html)
    html_source_dir = str(main_html.name).split(".")[0]
    (out / html_source_dir).mkdir(parents=True, exist_ok=True)
    process(html, out / html_source_dir)
    md = str()
    for html_fname in toc:
        md_heading_meta_tag = read(out / html_source_dir / html_fname).find(
            "meta", {"name": "md-heading"}
        )
        md += (
            transform_to_md_heading(md_heading_meta_tag.get("type"))
            + " "
            + md_heading_meta_tag.get("md-heading")
            + "\n"
        )
        md += '--8<-- "' + html_source_dir + "/" + html_fname + '"\n\n'

    with open(str(out / html_source_dir) + ".md", "w") as f:
        f.write(md)


app = typer.Typer(add_completion=False)

app.add_typer(
    rm_heading_app,
    name="rm-heading",
    help="Remove first heading from HTML file and place it to meta tag",
)
app.add_typer(
    rm_crosslinks_app, name="rm-crosslinks", help="Remove crosslinks from HTML file"
)
app.add_typer(rm_toc_app, name="rm-toc", help="Remove table of contents from HTML file")
app.add_typer(
    fix_links_app,
    name="fix-links",
    help="Fix links in HTML file that refer to other HTML files",
)
app.add_typer(
    fix_refs_app,
    name="fix-refs",
    help="Fixes refs in HTML files (for instance, equation refs)",
)
app.add_typer(
    process_app,
    name="process",
    help="Full processing of HTML file or directory with HTML files",
)
app.add_typer(rm_stylesheet_app, name="rm-stylesheet", help="Removes link to css file")
app.add_typer(
    md_app, name="markdown", help="Create markdown file from HTML files in directory"
)

if __name__ == "__main__":
    app()
