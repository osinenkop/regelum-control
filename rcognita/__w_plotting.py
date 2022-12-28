import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pylab import cm

from .__utilities import rc


def plot_optimization_results(
    critic_constr_expr,
    critic_obj_expr,
    actor_constr_expr,
    actor_obj_expr,
    symbolic_var_critic,
    symbolic_var_actor,
    w_init_critic,
    w_optimized_critic,
    w_init_actor,
    w_optimized_actor,
    grid_offset=10,
    grid_size=300,
    marker_size=30,
):

    f_obj_critic_csd = rc.to_casadi_function(critic_obj_expr, symbolic_var_critic)
    f_constr_critic_csd = rc.to_casadi_function(critic_constr_expr, symbolic_var_critic)
    f_obj_actor_csd = rc.to_casadi_function(actor_obj_expr, symbolic_var_actor)
    f_constr_actor_csd = rc.to_casadi_function(actor_constr_expr, symbolic_var_actor)

    def f_obj_critic(x, y):
        return f_obj_critic_csd([x, y]).full()

    def f_constr_critic(x, y):
        return 1 * (f_constr_critic_csd([x, y]).full() > 0)

    def f_obj_actor(x, y):
        return f_obj_actor_csd([x, y]).full()

    def f_constr_actor(x, y):
        return 1 * (f_constr_actor_csd([x, y]).full() > 0)

    f_obj_critic = np.vectorize(f_obj_critic)
    f_constr_critic = np.vectorize(f_constr_critic)
    f_obj_actor = np.vectorize(f_obj_actor)
    f_constr_actor = np.vectorize(f_constr_actor)

    x_critic, y_critic = np.meshgrid(
        np.linspace(
            min(w_init_critic[0], w_optimized_critic[0]) - grid_offset,
            max(w_init_critic[0], w_optimized_critic[0]) + grid_offset,
            grid_size,
        ),
        np.linspace(
            min(w_init_critic[1], w_optimized_critic[1]) - grid_offset,
            max(w_init_critic[1], w_optimized_critic[1]) + grid_offset,
            grid_size,
        ),
    )
    x_actor, y_actor = np.meshgrid(
        np.linspace(
            min(w_init_actor[0], w_optimized_actor[0]) - grid_offset,
            max(w_init_actor[0], w_optimized_actor[0]) + grid_offset,
            grid_size,
        ),
        np.linspace(
            min(w_init_actor[1], w_optimized_actor[1]) - grid_offset,
            max(w_init_actor[1], w_optimized_actor[1]) + grid_offset,
            grid_size,
        ),
    )
    (fig, ax_array) = plt.subplots(2, 2)

    [ax_critic_obj, ax_critic_constr], [ax_actor_obj, ax_actor_constr] = ax_array

    ax_critic_obj.set_xlabel("w_1")
    ax_critic_obj.set_ylabel("w_2")

    ax_critic_constr.set_xlabel("w_1")
    ax_critic_constr.set_ylabel("w_2")

    ax_actor_obj.set_xlabel("w_1")
    ax_actor_obj.set_ylabel("w_2")

    ax_actor_constr.set_xlabel("w_1")
    ax_actor_constr.set_ylabel("w_2")

    constr_array_critic = f_constr_critic(x_critic, y_critic)
    constr_array_actor = f_constr_actor(x_actor, y_actor)

    cmap = matplotlib.colors.ListedColormap(["springgreen", "lightcoral"])
    bounds = [0.0, 0.5, 1.0]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    cp_critic_obj = ax_critic_obj.contourf(
        x_critic, y_critic, f_obj_critic(x_critic, y_critic)
    )
    fig.colorbar(cp_critic_obj, ax=ax_critic_obj, fraction=0.046, pad=0.04)

    cp_critic_constr = ax_critic_constr.contourf(
        x_critic, y_critic, constr_array_critic, cmap=cmap, norm=norm
    )

    fig.colorbar(cp_critic_constr, ax=ax_critic_constr, fraction=0.046, pad=0.04)

    cp_actor_obj = ax_actor_obj.contourf(
        x_actor, y_actor, f_obj_actor(x_actor, y_actor)
    )
    fig.colorbar(cp_actor_obj, ax=ax_actor_obj, fraction=0.046, pad=0.04)

    cp_actor_constr = ax_actor_constr.contourf(
        x_actor, y_actor, constr_array_actor, cmap=cmap, norm=norm
    )

    fig.colorbar(cp_actor_constr, ax=ax_actor_constr, fraction=0.046, pad=0.04)

    w_init_critic = rc.squeeze(rc.array(w_init_critic, rc_type=rc.NUMPY))
    w_init_actor = rc.squeeze(rc.array(w_init_actor, rc_type=rc.NUMPY))

    w_optimized_critic = rc.array(w_optimized_critic, rc_type=rc.NUMPY)
    w_optimized_actor = rc.array(w_optimized_actor, rc_type=rc.NUMPY)
    #############################
    ax_critic_obj.scatter(
        w_init_critic[0],
        w_init_critic[1],
        s=marker_size,
        marker="o",
        label="Initial critic weights",
    )
    ax_critic_obj.scatter(
        w_optimized_critic[0],
        w_optimized_critic[1],
        s=marker_size,
        marker="*",
        label="Optimized critic weights",
        c="w",
    )

    ax_critic_obj.plot(
        [w_init_critic[0], w_optimized_critic[0]],
        [w_init_critic[1], w_optimized_critic[1]],
        "--",
    )
    #############################
    ax_actor_obj.scatter(
        w_init_actor[0],
        w_init_actor[1],
        s=marker_size,
        marker="o",
        label="Initial actor weights",
    )
    ax_actor_obj.scatter(
        w_optimized_actor[0],
        w_optimized_actor[1],
        s=marker_size,
        marker="*",
        label="Optimized actor weights",
        c="w",
    )

    ax_actor_obj.plot(
        [w_init_actor[0], w_optimized_actor[0]],
        [w_init_actor[1], w_optimized_actor[1]],
        "--",
    )
    #############################
    ax_critic_constr.scatter(
        w_init_critic[0],
        w_init_critic[1],
        s=marker_size,
        marker="o",
        label="Initial critic weights",
    )
    ax_critic_constr.scatter(
        w_optimized_critic[0],
        w_optimized_critic[1],
        s=marker_size,
        marker="*",
        label="Optimized critic weights",
        c="w",
    )

    ax_critic_constr.plot(
        [w_init_critic[0], w_optimized_critic[0]],
        [w_init_critic[1], w_optimized_critic[1]],
        "--",
    )
    #############################
    ax_actor_constr.scatter(
        w_init_actor[0],
        w_init_actor[1],
        s=marker_size,
        marker="o",
        label="Initial actor weights",
    )
    ax_actor_constr.scatter(
        w_optimized_actor[0],
        w_optimized_actor[1],
        s=marker_size,
        marker="*",
        label="Optimized actor weights",
        c="w",
    )

    ax_actor_constr.plot(
        [w_init_actor[0], w_optimized_actor[0]],
        [w_init_actor[1], w_optimized_actor[1]],
        "--",
    )
    #############################

    ax_critic_constr.set_title(f"CALF critic constraint")
    ax_critic_constr.legend()

    ax_actor_constr.set_title(f"CALF actor constraint")
    ax_actor_constr.legend()

    ax_critic_obj.set_title(f"Critic objective")
    ax_critic_obj.legend()

    ax_actor_obj.set_title(f"Actor objective")
    ax_actor_obj.legend()

    ax_critic_obj.set_aspect("equal", adjustable="box")
    ax_critic_constr.set_aspect("equal", adjustable="box")
    ax_actor_obj.set_aspect(1.0 / ax_actor_obj.get_data_ratio(), adjustable="box")
    ax_actor_constr.set_aspect(1.0 / ax_actor_constr.get_data_ratio(), adjustable="box")
