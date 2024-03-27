"""Tools for creating and retrieving user settings."""

import stat
import subprocess
from pathlib import Path
import os
from cues import Select
from omegaconf import OmegaConf
import cues
from rich import print

from regelum.__internal.cues_fix import RegelumSurvey

user_data_dir = Path(__file__).parent.parent
config_file = user_data_dir / "user_settings.yaml"

default_user_settings = {
    "MLFLOW_TRACKING_URI": None,
    "DISALLOW_UNCOMMITTED": True,
    "JOBS": 1,
    "IS_CONTRIBUTOR": False,
    "FULL_NAME": None,
    "E_MAIL": None,
    "JOBS_TESTING": 1,
}


def get_user_settings():
    if not os.path.exists(config_file):
        configure_user_settings()
    return OmegaConf.load(config_file)


def choice(*options, message=None):
    assert message is not None
    return Select("name", message, options).send()["name"]


def ok(message):
    return choice("Ok", message=message)


def form(*fields, message=None):
    assert message is not None
    fields = [{"name": i, "message": field} for i, field in enumerate(fields)]
    return cues.Form("name", message, fields).send()["name"]


def configure_user_settings():
    user_settings = default_user_settings.copy()
    if not os.path.exists(user_data_dir):
        os.mkdir(user_data_dir)
    print(
        "Welcome to [bold magenta]regelum[/bold magenta]! Before we proceed let's quickly configure your setup."
    )
    name = "cores"
    message = "Please select the number of cores that regelum will use by default:"
    scale = range(1, os.cpu_count() + 1)
    fields = [{"name": "cores", "message": "Cores"}]
    cue = RegelumSurvey(name, message, scale, fields)
    user_settings["JOBS"] = cue.send()["cores"]["cores"]
    ok(
        "This choice can always be overriden by specifying [bold magenta]--jobs=[/bold magenta] in command line arguments."
    )
    menu = [
        "No, just keep the artifacts in ./regelum_data/mlflow.",
        "Yes, let me set the URI.",
        'I don\'t even know what "mlflow" is!',
    ]
    msg = "Regelum uses [bold magenta]mlflow[/bold magenta] to store experiment data. By default a subfolder is created in the current working directory to store mlflow artifacts. Would you like to set a custom mlflow URI?"
    ans = choice(*menu, message=msg)
    if ans == menu[0]:
        user_settings["MLFLOW_TRACKING_URI"] = None
    elif ans == menu[1]:
        ans = form(
            "Tracking URI", message="Specify the URI to your mlflow data server."
        )[0]
        user_settings["MLFLOW_TRACKING_URI"] = ans
    else:
        user_settings["MLFLOW_TRACKING_URI"] = None
        ok(
            "Basically, [bold magenta]mlflow[/bold magenta] is a browser user interface that let's you see your [bold magenta]graphs[/bold magenta], [bold magenta]metrics[/bold magenta] and [bold magenta]animations[/bold magenta]. You also get to easilly compare the outcomes of different experiments. It might not sound like much, but, in fact, it is almost impossible to get by without a tool like this."
        )
        ok(
            "This 'tracking uri' thing is what lets you specify your own mlflow server, which can be useful if you have a group of people working on the same collection of experiments. Since you probably don't have a server like this yet, we'll just store your mlflow experiment data in [bold magenta]./regelum_data/mlflow[/bold magenta]."
        )
        ok(
            "To view your experiment data via mlflow, all you need to do is:\n [bold grey]cd ./regelum_data[/bold grey]\n [bold grey]mlflow ui[/bold grey]\n Then just click the link that shows up in the logs."
        )
    ok(
        "By the way, don't forget to add [bold magenta]regelum_data/[/bold magenta] to your [bold magenta].gitignore[/bold magenta]! Regelum will always create one of these folders in your current working directory."
    )
    ans = choice(
        "Yes, prevent me from running experiments if I have uncommitted changes.",
        "No, let me run regelum even when my repo has uncommitted changes.",
        message="By default regelum, [bold magenta]will not start unless all changes in your repo are committed[/bold magenta]. This helps to ensure reproducibility. If you're doing serious research, it will help you a lot, but if not, then it might be really annoying. Would you like to keep this feature enabled?",
    )
    user_settings["DISALLOW_UNCOMMITTED"] = "Yes" in ans
    name, email = form(
        "Full name",
        "E-mail",
        message="At the end of each experiment regelum generates a [bold magenta]summary[/bold magenta] that outlines the setup, the outcomes and steps to reproduce the results. When working in a team it is useful to have credentials of the author of the experiment present in this summary. Please provide your [bold magenta]e-mail[/bold magenta] and [bold magenta]e-mail[/bold magenta] address or leave the lines blank.",
    )
    user_settings["FULL_NAME"] = name if name else None
    user_settings["E_EMAIL"] = email if email else None
    script_dev = os.path.abspath(__file__ + "/../../scripts/developer-setup.sh")
    user_settings["IS_CONTRIBUTOR"] = False
    if os.path.exists(script_dev):
        ans = choice(
            "No, I'm only a user of regelum and I do not intend to modify it.",
            "Yes, I am a developer of regelum.",
            message="Last question. Are you a contributor?",
        )
        if "Yes" in ans:
            ok("In that case, let me install some additional dependencies...")
            try:
                st = os.stat(script_dev)
                os.chmod(script_dev, st.st_mode | stat.S_IEXEC)
                subprocess.run(script_dev, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(e)
                ok(
                    f"It would seem that something went wrong when setting up your environment for contributing to regelum. Before you can start your project you need to make sure that all the necessary pre-requisites are installed. Please try to determine, what went wrong when running {script_dev}."
                )
                return
            user_settings["IS_CONTRIBUTOR"] = True
            name = "cores"
            message = (
                "Now, please specify how many cores are to be used during testing:"
            )
            scale = range(1, os.cpu_count() + 1)
            fields = [{"name": "cores", "message": "Cores"}]
            cue = RegelumSurvey(name, message, scale, fields)
            user_settings["JOBS_TESTING"] = cue.send()["cores"]["cores"]
    ok(
        "We're [bold magenta]done[/bold magenta]! Bear in mind that you can always change these settings by running regelum with [bold magenta]--configure[/bold magenta]."
    )
    OmegaConf.save(OmegaConf.create(user_settings), config_file)
