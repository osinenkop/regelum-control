#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import os


target = os.path.abspath(__file__ + "/../../README.rst")
destination = os.path.abspath(__file__ + "/../README.rst")

with open(target, "r") as f:
    s = f.read()


#logo_top = r"\.\. logo" + "\n"
#m = re.search(f"({logo_top}.*?\n)[a-zA-Z0-9\s]*?\n=", s, re.S)
#logo = m.group(1)

#m = re.search(
#    "(Example run with a mobile robot simulation\n.*?\n)[a-zA-Z0-9\s]*?\n=",
#    s,
#    flags=re.S,
#)
#example = m.group(1)


m = re.search("(Table of content\n.*?\n)[a-zA-Z0-9\s]*?\n=", s, flags=re.S)
table_of_content = m.group(1)


links_to_table = "`To table of content <#Table-of-content>`__"


link_to_docs = "A detailed documentation is available `here <https://aidynamicaction.github.io/rcognita/>`__."


for fragment in [table_of_content, links_to_table]:
    s = s.replace(fragment, "")


with open(destination, "w") as f:
    f.write(s)

import os
os.system(f"rst2myst convert {destination}")

with open(destination.replace(".rst", ".md"), "r") as f:
    s = f.read()

def replace_image_link(match):
    link = match.group(1)
    return f"![image]({link})"

s = re.sub(r"```\{image\} (.*)\n```", replace_image_link, s)

with open(destination.replace(".rst", ".md"), "w") as f:
    f.write(s.replace(r"\*", "*").replace(r'\"', '"').replace(r"\'", "'"))


with open(destination, "r") as f:
    s = f.read()

logo = re.search(r"\.\. image:: .*logo.*", s).group(0)

with open(destination, "w") as f:
    f.write(s.replace(logo, "").replace(link_to_docs, ""))