import glob
import os

colors = {"55a5d9": "8cc770", "009568": "604083"}

for filepath in glob.iglob("../docs/**/*.css", recursive=True):
    with open(filepath) as file:
        s = file.read()
    for color in colors:
        s = s.replace(color, colors[color])
    with open(filepath, "w") as file:
        file.write(s)
