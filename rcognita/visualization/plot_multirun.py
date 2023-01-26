import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_objectives(total_obj_callbacks, environment, path=None):

    fig = plt.figure(figsize=(10, 10))
    ax = fig.subplots()
    ax.set_xlabel("episode")
    ax.set_ylabel("Total objective")

    dfs = [pd.DataFrame(callback.data) for callback in total_obj_callbacks]
    df = pd.concat(dfs, axis=1)

    plot_objectives_per_controller(df, total_obj_callbacks, environment, ax)

    plt.legend()
    plt.title(f"{environment}")
    plt.savefig(path)


def plot_objectives_per_controller(df, callbacks, environment, axes):

    df["time"] = callbacks[0].data.index
    df.set_index("time", inplace=True)
    print(df)

    ci = 95

    low = np.percentile(df.values, 50 - ci / 2, axis=1)
    high = np.percentile(df.values, 50 + ci / 2, axis=1)

    plt.fill_between(df.index, low, high, color="r", alpha=0.2)

    axes.plot(df.index, df.values, color="r", alpha=0.2)
    df["mean_traj"] = df.mean(axis=1)
    axes.plot(
        df.index,
        df.mean_traj.values,
        color="b",
        label=f"mean {environment} Total objective",
    )
