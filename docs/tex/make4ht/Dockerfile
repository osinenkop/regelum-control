FROM debian:unstable-slim

LABEL "maintainer"="Michal Hoftich <michal.h21@gmail.com>"
LABEL "repository"="https://github.com/michal-h21/make4ht"
LABEL "homepage"="https://github.com/michal-h21/make4ht"

LABEL "com.github.actions.name"="LaTeX to XML"
LABEL "com.github.actions.description"="Convert LaTeX documents to XML with make4ht."
LABEL "com.github.actions.icon"="code"
LABEL "com.github.actions.color"="blue"

ENV DEBIAN_FRONTEND noninteractive

# Fix openjre installation isssue
RUN mkdir -p /usr/share/man/man1

# Install all TeX and LaTeX dependencies
RUN apt-get update && \
    apt-get install --yes -y --no-install-recommends \
    make luatex texlive texlive-base texlive-luatex texlive-latex-extra context \
    tidy texlive-extra-utils python3-pygments python3-setuptools dvisvgm \
    texlive-science texlive-games texlive-humanities texlive-music texlive-publishers texlive-science \
    texlive-plain-generic texlive-plain-generic context \
    pandoc latexmk lmodern fonts-lmodern tex-gyre fonts-texgyre \
    default-jre-headless zip git \
    texlive-lang-all && \
    apt-get autoclean && apt-get --purge --yes autoremove && \
    git clone https://github.com/michal-h21/make4ht && \
    cd make4ht && \
    make justinstall SUDO="" 

RUN apt-get install texlive-bibtex-extra biber -y

#ENV filename "sample.tex"
# extra packages installes using apt-get
#ENV extra_packages ""
# output directory for the HTML files
#ENV output_dir "./"
# make4ht command to be executed
#ENV command "make4ht -v"

# ENV TEXINPUTS .:/root/texmf//:
ADD entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

RUN mkdir work
WORKDIR work


ENTRYPOINT ["sh", "/entrypoint.sh"]
CMD ["-h"]
