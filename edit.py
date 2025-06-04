import sys
import re


def edit():
    path = str(sys.argv[1])
    yaml = "---\ntitle: TITLE\nmathjax: true\ncategories:\n  - category\ntags:\n  - tag\n---\n\n"
    with open(path, "r") as file:
        filedata = file.read()
    filedata = re.sub(r"!\[png\]\(", '<img src="/images/blog/AR-part1', filedata)
    filedata = re.sub(".png\)", '.png">', filedata)
    filedata = yaml + filedata
    with open(path, "w") as file:
        file.write(filedata)


if __name__ == "__main__":
    edit()
