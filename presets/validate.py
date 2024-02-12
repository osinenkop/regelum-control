from pathlib import Path
from typing import List
from mlflow.entities.run import Run as MlflowRun
import typer
from typing import Annotated
from mlflow.tracking import MlflowClient
import subprocess
import numpy as np
import pandas as pd
import re
from omegaconf import OmegaConf
from rich.console import Console
from rich.table import Table


def extract_overrides_from_bash(bash_path: Path) -> List[str]:
    with open(bash_path, "r") as f:
        content = f.read()
        extracted = [
            override
            for override in content.split()
            if ("=" in override or "--" in override)
        ]

        if 'seed="$1"' in extracted:
            extracted.pop(extracted.index('seed="$1"'))

        if "+seed=$seed" in extracted:
            extracted.pop(extracted.index("+seed=$seed"))

            matches = re.findall(r"(\d+)\.{2}(\d+)", content)

            if len(matches) > 0:
                first, sec = matches[0]
                extracted.append(
                    "+seed="
                    + ",".join([str(n) for n in range(int(first), int(sec) + 1)])
                )
                if "--single-thread" not in extracted:
                    extracted.append("--single-thread")

    return extracted


def extract_exp_name_from_bash_overrides(overrides: List[str]) -> str:
    return [o for o in overrides if "--experiment" in o][0].split("=")[1]


def extract_run_from_mlflow(
    mlflow_tracking_uri: str, exp_name: str, seed: None | int = None
) -> MlflowRun:
    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
    runs = client.search_runs(
        experiment_ids=[
            e.experiment_id for e in client.search_experiments() if exp_name in e.name
        ],
    )
    if len(runs) == 0:
        raise ValueError(f"Experiment {exp_name} not found")

    if seed is None:
        return runs[0]
    else:
        for run in runs:
            if int(run.data.params["seed"]) == seed:
                return run
        raise ValueError(f"Seed {seed} not found")


def run_validation(
    mlflow_run: MlflowRun, overrides: List[str]
) -> subprocess.CompletedProcess:
    val_overrides = overrides.copy()

    for param in ("seed", "N_iterations", "--experiment", "jobs", "--single-thread"):
        ids = [i for i, override in enumerate(overrides) if param in override]

        for idx in ids:
            val_overrides.remove(overrides[idx])

        if param == "N_iterations" and len(ids) > 0:
            val_overrides.append("scenario.N_iterations=2")

    val_overrides.append("--single-thread")
    val_overrides.append("--experiment=validation")

    if mlflow_run.data.params.get("seed") is not None:
        val_overrides.append("+seed=" + str(mlflow_run.data.params.get("seed")))
    val_run_str = "python run_stable.py " + " ".join(val_overrides)
    print("RUNNING VALIDATION:\n", val_run_str)
    return subprocess.run(val_run_str, shell=True, check=True)


def validate(
    mlflow_tracking_uri: str,
    mlflow_run: MlflowRun,
    overrides: List[str],
    completed_run: subprocess.CompletedProcess,
):
    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
    runs = client.search_runs(
        experiment_ids=[
            e.experiment_id
            for e in client.search_experiments()
            if "validation" in e.name
        ],
    )

    if len(runs) == 0:
        raise ValueError(f"Validation experiment not found")

    latest_start_time = 0
    val_run = None
    agent = [o for o in overrides if "scenario" in o][0].split("=")[1]
    system = [o for o in overrides if "system" in o][0].split("=")[1]
    for run in runs:
        agent_mlflow_run = run.data.params["scenario"]
        system_mlflow_run = run.data.params["system"]
        timestamp = run.info.start_time
        if agent == agent_mlflow_run and system == system_mlflow_run:
            if timestamp > latest_start_time:
                latest_start_time = timestamp
                val_run = run

    if val_run is None:
        raise ValueError(f"Validation experiment not found")

    val_values = np.array(
        sorted(
            (
                Path(val_run.data.tags["run_path"]) / ".callbacks" / "ValueCallback"
            ).iterdir()
        )
    )
    mlflow_values = np.array(
        sorted(
            (
                Path(mlflow_run.data.tags["run_path"]) / ".callbacks" / "ValueCallback"
            ).iterdir()
        )
    )[: len(val_values)]
    is_failed = False
    if len(val_values) == 0:
        print("Validation failed")
        return True

    for i, (val_hdf, mlflow_hdf) in enumerate(zip(val_values, mlflow_values)):
        val_mean_value = pd.read_hdf(val_hdf).objective.mean()
        mlflow_mean_value = pd.read_hdf(mlflow_hdf).objective.mean()
        print(
            f"Val in iteration {i + 1}:",
            val_mean_value,
            f"\nMLflow in iteration {i + 1}:",
            mlflow_mean_value,
        )
        if not np.allclose(
            val_mean_value,
            mlflow_mean_value,
        ):
            is_failed = True
    if not is_failed:
        print("Validation succeeded")
    else:
        print("Validation failed")

    return is_failed


bash_app = typer.Typer()


@bash_app.callback(invoke_without_command=True)
def bash(
    bash: Path = typer.Argument(),
    mlflow_tracking_uri: str = typer.Option(
        default=f"file:///{Path(__file__).parent / 'regelum_data' / 'mlruns'}"
    ),
    seed: Annotated[int, typer.Option()] = None,
) -> bool:
    bash_overrides = extract_overrides_from_bash(bash)
    mlflow_run = extract_run_from_mlflow(
        mlflow_tracking_uri=mlflow_tracking_uri,
        seed=seed,
        exp_name=extract_exp_name_from_bash_overrides(bash_overrides),
    )

    completed_run = run_validation(mlflow_run, bash_overrides)
    if completed_run.returncode != 0:
        raise ValueError(f"Validation failed: {completed_run.stderr}")

    return validate(mlflow_tracking_uri, mlflow_run, bash_overrides, completed_run)


all_app = typer.Typer()


@all_app.callback(invoke_without_command=True)
def all(
    config: Path = typer.Argument(),
    mlflow_tracking_uri: str = typer.Option(
        default=f"file:///{Path(__file__).parent / 'regelum_data' / 'mlruns'}"
    ),
):
    def print_stats(statuses: dict[str, bool], bash_pathes: List[str]):
        table = Table(title="Validation stats")

        table.add_column("Bash Path", justify="right", style="cyan", no_wrap=True)
        table.add_column("Status", justify="right")
        for bash_path in bash_pathes:
            if statuses.get(bash_path) is None:
                status = "not started"
                color = "[yellow]"
            else:
                status = "failed" if statuses.get(bash_path) else "succeeded"
                color = "[red]" if statuses.get(bash_path) else "[green]"

            table.add_row(bash_path, color + status)

        console = Console()
        console.print(table)

    table = {}
    bash_pathes = OmegaConf.load(config)
    print("VALIDATING:\n -", "\n - ".join(bash_pathes))
    print_stats(table, bash_pathes)
    for i, bash_path in enumerate(bash_pathes, start=1):
        try:
            is_failed = bash(bash=bash_path, mlflow_tracking_uri=mlflow_tracking_uri)
        except:
            is_failed = True
        table[str(bash_path)] = is_failed
        print_stats(table, bash_pathes)


app = typer.Typer(add_completion=False)

app.add_typer(all_app, name="config")
app.add_typer(bash_app, name="bash")

if __name__ == "__main__":
    app()
