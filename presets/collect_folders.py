import os, sys

PARENT_DIR = os.path.abspath(os.getcwd() + "/../../")
sys.path.insert(0, PARENT_DIR)
CUR_DIR = os.path.abspath(os.getcwd() + "/..")
sys.path.insert(0, CUR_DIR)

import datetime
import rcognita.scripts as scripts
import omegaconf

from argparse import ArgumentParser

from datetime import datetime

parser = ArgumentParser()
parser.add_argument("--from_", default=None, type=str)
parser.add_argument("--to_", default=None, type=str)

args = parser.parse_args()


if args.from_ is not None:
    from_date = datetime.strptime(args.from_, "%Y-%m-%dT%H-%M-%S")
else:
    from_date = None

if args.to_ is not None:
    to_date = datetime.strptime(args.to_, "%Y-%m-%dT%H-%M-%S")
else:
    to_date = None
