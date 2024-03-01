import typer
from pathlib import Path
import bs4
from typing import Annotated
import numpy as np
from typing_extensions import Literal
from bs4 import Comment
import re
import base64
from pdf2image import convert_from_path
from io import BytesIO


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
    heading_type: Literal["h1", "h2", "h3", "h4", "h5", "h6"],
    name: str,
) -> str:
    heading = (
        {
            "h1": "#",
            "h2": "##",
            "h3": "##",
            "h4": "###",
            "h5": "####",
            "h6": "#####",
        }[heading_type]
        + " "
        + name
    )

    return heading


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
            print("Warning: more than one heading found in", html_name)
        first_heading = all_headings[0]
        first_heading.text.strip()
        first_heading_type, first_heading_txt, first_heading_id, first_heading_link = (
            first_heading.name,
            first_heading.text.strip(),
            first_heading["id"],
            first_heading.find("a", id=True)["id"],
        )
        first_heading.decompose()
        md_heading_tag = bs.new_tag("meta")
        md_heading_tag.attrs["name"] = "md-heading"
        md_heading_tag.attrs["md-heading"] = first_heading_txt
        md_heading_tag.attrs["type"] = first_heading_type
        md_heading_tag.attrs["id"] = first_heading_id
        md_heading_tag.attrs["link"] = first_heading_link

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
    for toc_type in [
        "subsectionTOCS",
        "sectionTOCS",
        "likesubsectionTOCS",
        "likesectionTOCS",
    ]:
        for toc in bs.find("body").find_all("div", {"class": toc_type}):
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
        stylesheet.decompose()

    save(bs, html, out, help="  - Fixed stylesheet.")


fix_links_app = typer.Typer()


@fix_links_app.callback(invoke_without_command=True)
def fix_links(
    html: Annotated[Path, typer.Argument()],
    out: Annotated[Path, typer.Option()] = None,
    href_renamings=None,
):
    bs = read(html)
    for link in bs.find("body").find_all("a", href=True):
        if ".html#" in link["href"]:
            link["href"] = link["href"][link["href"].find(".html#") + 5 :]

        if href_renamings is not None:
            if link["href"] in href_renamings:
                link["href"] = href_renamings[link["href"]]

    for link in bs.find("body").find_all("a", id=True):
        if link["id"].endswith("doc"):
            link.decompose()

    save(
        bs,
        html,
        out,
        help="  - Fixed links"
        + (" with href_renamings." if href_renamings is not None else "."),
    )


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


fix_mathjax_script_app = typer.Typer()


@fix_mathjax_script_app.callback(invoke_without_command=True)
def fix_mathjax_script(
    html: Annotated[Path, typer.Argument(help="Directory with HTML files to process")],
    out: Annotated[Path, typer.Option()],
):
    bs = read(html)
    mathjax_script = bs.find("script", {"id": "MathJax-script"})

    del mathjax_script.attrs["async"]
    del mathjax_script.attrs["id"]

    save(bs, html, out, help="  - Fixed MathJax script.")


fix_algorithmic_app = typer.Typer()


@fix_algorithmic_app.callback(invoke_without_command=True)
def fix_algorithmic(
    html: Annotated[Path, typer.Argument(help="Directory with HTML files to process")],
    out: Annotated[Path, typer.Option()],
):
    bs = read(html)
    for figure in bs.find_all("figure"):
        algorithmic = figure.find("div", {"class": "algorithmic"})
        if algorithmic is not None:
            figure.insert_after(algorithmic)
    save(bs, html, out, help="  - Fixed algorithmic.")


def fix_img(
    html: Annotated[Path, typer.Argument(help="Directory with HTML files to process")],
    out: Annotated[Path, typer.Option()],
):
    bs = read(html)

    for img in bs.find_all("img"):
        gfx_path = Path("gfx") / Path(img.attrs["src"]).name
        if gfx_path.exists():
            with open(gfx_path, "rb") as gfx_file:
                encoded_string = base64.b64encode(gfx_file.read())
        else:
            gfx_path = Path("gfx") / (Path(img.attrs["src"]).name[:-5] + ".pdf")
            if gfx_path.exists():
                images = convert_from_path(gfx_path)
                buffered = BytesIO()
                images[0].save(buffered, "PNG")
                encoded_string = base64.b64encode(buffered.getvalue())
            else:
                print("Could not find", img.attrs["src"])

        img.attrs["src"] = "data:image/png;base64, " + encoded_string.decode()

    save(bs, html, out, help="  - Fixed img src.")


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
        fix_mathjax_script(path, out)
        fix_algorithmic(path, out)
        fix_img(path, out)


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
    href_renamings = {}
    for html_fname in toc:
        md_heading_meta_tag = read(out / html_source_dir / html_fname).find(
            "meta", {"name": "md-heading"}
        )
        href_renamings["#" + md_heading_meta_tag.get("link") + "doc"] = (
            "#" + md_heading_meta_tag.get("id")
        )

    for html_fname in toc:
        print("Postprocessing", html_fname)
        fix_links(out / html_source_dir / html_fname, href_renamings=href_renamings)

        md_heading_meta_tag = read(out / html_source_dir / html_fname).find(
            "meta", {"name": "md-heading"}
        )
        md += (
            transform_to_md_heading(
                md_heading_meta_tag.get("type"),
                md_heading_meta_tag.get("md-heading"),
            )
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
app.add_typer(
    fix_mathjax_script_app,
    name="fix-mathjax-script",
    help="Removes async load of MathJax.js in HTML file",
)
app.add_typer(
    fix_algorithmic_app,
    name="fix-algorithmic",
    help="Extract all algorihtmic blocks from <figure></figure> and place them right after figure",
)


if __name__ == "__main__":
    app()
