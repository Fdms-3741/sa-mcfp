import re
import json
from pathlib import Path
import sys

sys.path.append("../../")

from find_hyperparams import GetParameters

files = Path(".")

for file in [i for i in files.iterdir() if re.search("bayesopt.json", i.name)]:
    with file.open() as fil:
        a = json.load(fil)

    # Gets important names
    graphName = re.match(r"\S+.json-", file.name)
    if not graphName:
        continue
    graphName = graphName[0]
    obj, tech = file.name[len(graphName) :].split("-")[:2]
    graphName = graphName[:-1]

    smallestValue = None
    params = None
    for i in a:
        if not smallestValue:
            smallestValue = i["target"]
            params = i["params"]
            continue

        if i["target"] > smallestValue:
            smallestValue = i["target"]
            params = i["params"]

    if not params:
        continue

    # Get real parameter values
    params["temp"], params["k"], params["n"] = GetParameters(
        params["temp"], params["k"]
    )
    print(graphName, obj, tech, smallestValue, params)
