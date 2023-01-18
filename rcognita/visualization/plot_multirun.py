import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_objectives(job_results, environment):
    controller_override_index = [
        i for i, v in enumerate(job_results.overrides[0]) if "controller=" in v
    ][0]

    group_df = job_results.groupby(
        job_results.overrides.map(lambda x: x[controller_override_index].split("=")[1])
    )
    controllers = job_results.overrides.map(
        lambda x: x[controller_override_index].split("=")[1]
    ).unique()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.subplots()
    ax.set_xlabel("time")
    ax.set_ylabel("running cost")

    for controller in controllers:

        df = group_df.get_group(controller)
        callbacks = df["ObjectiveCallbackMultirun"].values

        dfs = [pd.DataFrame(callback.data) for callback in callbacks]
        df = pd.concat(dfs, axis=1)

        plot_objectives_per_controller(df, callbacks, controller, ax)

    plt.legend()
    plt.title(f"{environment}")
    plt.show()


def plot_objectives_per_controller(df, callbacks, controller, axes):

    df["time"] = callbacks[0].timeline
    df.set_index("time", inplace=True)
    print(df)

    ci = 95

    low = np.percentile(df.values, 50 - ci / 2, axis=1)
    high = np.percentile(df.values, 50 + ci / 2, axis=1)

    plt.fill_between(df.index, low, high, color="r", alpha=0.2)

    # axes.plot(df.index, df.values, color="r", alpha=0.2)
    df["mean_traj"] = df.mean(axis=1)
    axes.plot(
        df.index,
        df.mean_traj.values,
        color="b",
        label=f"mean {controller} running cost",
    )
