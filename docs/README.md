# Introduction

Our choice for powering our documentation site is the [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) framework, selected for its strong advantages:

- **Aesthetic Appeal:** We find the visual design appealing from our perspective.
- **Markdown-Based:** The documentation pages utilize Markdown, presenting a more straightforward approach compared to reStructuredText (RST), which we found overly complex.
- **Extensibility:** MkDocs Material is highly customizable with a wide variety of plugins, many of which we actively employ in our documentation.

For a comprehensive understanding of the documentation structure, we recommend reviewing the [official MkDocs Material documentation](https://squidfunk.github.io/mkdocs-material/). Try to host a few Markdown pages by yourself to understand how it works before diving deeper into this documentation. [This video](https://youtu.be/Q-YA_dA8C20?si=Chl5ifAq95306ZMx) is a good starting point.

# Overview

This section outlines the organization of our documentation.

## Primary Configuration

The central configuration file for our documentation is situated at [../mkdocs.yml](../mkdocs.yml). For guidance on organizing mkdocs.yml, consult the [MkDocs Material documentation](https://squidfunk.github.io/mkdocs-material/) and [this instructional video](https://youtu.be/Q-YA_dA8C20?si=Chl5ifAq95306ZMx).

To host locally the documentation, please go to the root of the repo and write

```bash
mkdocs serve
```

This command parses [../mkdocs.yml](../mkdocs.yml) and starts a docs development server on `localhost:8000`

## Deploy to remote site

Follow the steps described in [repo](https://github.com/osinenkop/regelum-control-docs-deploy) (the repo is private, make sure you have access).

## Documentation Structure 

The structure of the current [docs](./) directory is as follows:

- [`src/`](./src/): This directory houses the essential content, including Markdown files and the organizational structure of the documentation site (home page, tutorials, systems, etc.). While a full listing is not provided here, we highlight the importance of the following subdirectory for technical purposes:
    - [`_internal/`](./src/_internal/): Contains JavaScript and CSS files for the project.
        - [`javascripts/`](./src/_internal/javascripts/)
            - [`mathjax.js`](./src/_internal/javascripts/mathjax.js): Configures MathJax, our chosen engine for rendering LaTeX equations. See [here](https://squidfunk.github.io/mkdocs-material/reference/math/?h=mathjax#mathjax) for details.
        - [`stylesheets/`](./src/_internal/stylesheets/): The CSS files for the project.
            - [`colors.css`](./src/_internal/stylesheets/colors.css): Contains the list of colors for the main theme of the docs site.
            - [`texhtml.css`](./src/_internal/stylesheets/texhtml.css): We created custom toolbox converts LaTeX files to HTML for integration with the primary Markdown documentation. The styling for these conversions is specified in [texhtml.css](./src/_internal/stylesheets/texhtml.css). For instructions on rendering LaTeX for insertion into the main content, see [tex/README.md](./tex/README.md).
- [`include/`](./include/): Contains HTML files generated in the [tex/](./tex/) directory, alongside other Markdown files ready to be included in [`tex/``](./tex/).
- [`overrides/`](./overrides/): This folder contains files that override the main documentation template.
    - [`main.html`](./overrides/main.html): Alters the main template by incorporating our LaTeX preamble, utilized extensively in local research and predominantly consisting of `\newcommand` definitions. This file is auto-generated, leveraging the preamble content from the [`tex/`](./tex/) directory. Consequently, it is unnecessary to import it in Markdown files, allowing for the use of LaTeX shortcuts throughout the documentation via automatic loading of the preamble.
- [`scripts/``](./scripts/): Contains scripts for pre-building documentation, executed through [this mkdocs plugin](https://oprypin.github.io/mkdocs-gen-files/index.html).
    - [`add_preamble_to_mkdocs_template.py`](./scripts/add_preamble_to_mkdocs_template.py): Integrates the LaTeX preamble into the main template file [`main.html`](./overrides/main.html).
    - [`gen_ref_pages.py`](./scripts/gen_ref_pages.py): A script for generating API reference pages from docstrings before the build process. We refer the reader to [official plugin](https://mkdocstrings.github.io/), which is used for the purpose.
    - [`include_tex_htmls.py`](./scripts/include_tex_htmls.py): A utility to copy HTML files derived from LaTeX sources located in the `tex/` directory during the prebuild process.
    - [`include_notebooks.py`](./scripts/include_notebooks.py): A utility to convert ipynb notebooks to markdown format during the prebuild process.
- [`tex/`](./tex/): Contains LaTeX source files and related documentation resources. For instructions on rendering LaTeX for insertion into the main content, see [tex/README.md](./tex/README.md).

