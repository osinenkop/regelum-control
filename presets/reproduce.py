import typer
import subprocess
from typing import Annotated
from enum import Enum


class Agent(str, Enum):
    calf = "calf"
    nominal = "nominal"
    mpc3 = "mpc3"
    mpc7 = "mpc7"
    mpc11 = "mpc11"
    ppo = "ppo"
    sdpg = "sdpg"
    ddpg = "ddpg"
    reinforce = "reinforce"


class System(str, Enum):
    inv_pendulum = "inv_pendulum"
    twotank = "2tank"
    robot = "3wrobot_kin"
    lunar_lander = "lunar_lander"
    kin_point = "kin_point"
    cartpole = "cartpole"


def main(
    agent: Agent = typer.Option(
        help="Agent name", show_default=False, show_choices=True
    ),
    system: System = typer.Option(help="System name", show_default=False),
):
    subprocess.run(["bash", f"bash/{agent.value}/{system.value}.sh"])


if __name__ == "__main__":
    typer.run(main)
