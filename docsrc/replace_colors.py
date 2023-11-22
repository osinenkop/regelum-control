import glob

to_replace = {
    "55a5d9": "8cc770",
    "009568": "604083",
    'class="icon icon-home"': 'class="icon icon-home" style="color: #8cc770;"',
}

for filepath in glob.iglob("../docs/**/*.css", recursive=True):
    with open(filepath) as file:
        s = file.read()
    for color in to_replace:
        s = s.replace(color, to_replace[color])
    with open(filepath, "w") as file:
        file.write(s)

for filepath in glob.iglob("../docs/**/*.html", recursive=True):
    with open(filepath) as file:
        s = file.read()
    for color in to_replace:
        s = s.replace(color, to_replace[color])
    with open(filepath, "w") as file:
        file.write(s)
