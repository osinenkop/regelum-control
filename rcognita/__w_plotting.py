import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pylab import cm

from .__utilities import rc


def plot_optimization_results(
    critic_constr_expr,
    critic_obj_expr,
   policy_constr_expr,
   policy_obj_expr,
    symbolic_var_critic,
    symbolic_var_policy,
    w_init_critic,
    w_optimized_critic,
    w_init_policy,
    w_optimized_policy,
    grid_offset=10,
    grid_size=300,
    marker_size=30,
):
    f_obj_critic_csd = rc.to_casadi_function(critic_obj_expr, symbolic_var_critic)
    f_constr_critic_csd = rc.to_casadi_function(critic_constr_expr, symbolic_var_critic)
    f_obj_policy_csd = rc.to_casadi_functionpolicy_obj_expr, symbolic_var_policy)
    f_constr_policy_csd = rc.to_casadi_functionpolicy_constr_expr, symbolic_var_policy)

    def f_obj_critic(x, y):
        return f_obj_critic_csd([x, y]).full()

    def f_constr_critic(x, y):
        return 1 * (f_constr_critic_csd([x, y]).full() > 0)

    def f_obj_policy(x, y):
        return f_obj_policy_csd([x, y]).full()

    def f_constr_policy(x, y):
        return 1 * (f_constr_policy_csd([x, y]).full() > 0)

    f_obj_critic = np.vectorize(f_obj_critic)
    f_constr_critic = np.vectorize(f_constr_critic)
    f_obj_policy = np.vectorize(f_obj_policy)
    f_constr_policy = np.vectorize(f_constr_policy)

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
    x_policy, y_policy = np.meshgrid(
        np.linspace(
            min(w_init_policy[0], w_optimized_policy[0]) - grid_offset,
            max(w_init_policy[0], w_optimized_policy[0]) + grid_offset,
            grid_size,
        ),
        np.linspace(
            min(w_init_policy[1], w_optimized_policy[1]) - grid_offset,
            max(w_init_policy[1], w_optimized_policy[1]) + grid_offset,
            grid_size,
        ),
    )
    (fig, ax_array) = plt.subplots(2, 2)

    [ax_critic_obj, ax_critic_constr], [ax_policy_obj, ax_policy_constr] = ax_array

    ax_critic_obj.set_xlabel("w_1")
    ax_critic_obj.set_ylabel("w_2")

    ax_critic_constr.set_xlabel("w_1")
    ax_critic_constr.set_ylabel("w_2")

    ax_policy_obj.set_xlabel("w_1")
    ax_policy_obj.set_ylabel("w_2")

    ax_policy_constr.set_xlabel("w_1")
    ax_policy_constr.set_ylabel("w_2")

    constr_array_critic = f_constr_critic(x_critic, y_critic)
    constr_array_policy = f_constr_policy(x_policy, y_policy)

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

    cp_policy_obj = ax_policy_obj.contourf(
        x_policy, y_policy, f_obj_policy(x_policy, y_policy)
    )
    fig.colorbar(cp_policy_obj, ax=ax_policy_obj, fraction=0.046, pad=0.04)

    cp_policy_constr = ax_policy_constr.contourf(
        x_policy, y_policy, constr_array_policy, cmap=cmap, norm=norm
    )

    fig.colorbar(cp_policy_constr, ax=ax_policy_constr, fraction=0.046, pad=0.04)

    w_init_critic = rc.squeeze(rc.array(w_init_critic, rc_type=rc.NUMPY))
    w_init_policy = rc.squeeze(rc.array(w_init_policy, rc_type=rc.NUMPY))

    w_optimized_critic = rc.array(w_optimized_critic, rc_type=rc.NUMPY)
    w_optimized_policy = rc.array(w_optimized_policy, rc_type=rc.NUMPY)
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
    ax_policy_obj.scatter(
        w_init_policy[0],
        w_init_policy[1],
        s=marker_size,
        marker="o",
        label="Initialpolicy weights",
    )
    ax_policy_obj.scatter(
        w_optimized_policy[0],
        w_optimized_policy[1],
        s=marker_size,
        marker="*",
        label="Optimizedpolicy weights",
        c="w",
    )

    ax_policy_obj.plot(
        [w_init_policy[0], w_optimized_policy[0]],
        [w_init_policy[1], w_optimized_policy[1]],
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
    ax_policy_constr.scatter(
        w_init_policy[0],
        w_init_policy[1],
        s=marker_size,
        marker="o",
        label="Initialpolicy weights",
    )
    ax_policy_constr.scatter(
        w_optimized_policy[0],
        w_optimized_policy[1],
        s=marker_size,
        marker="*",
        label="Optimizedpolicy weights",
        c="w",
    )

    ax_policy_constr.plot(
        [w_init_policy[0], w_optimized_policy[0]],
        [w_init_policy[1], w_optimized_policy[1]],
        "--",
    )
    #############################

    ax_critic_constr.set_title(f"CALF critic constraint")
    ax_critic_constr.legend()

    ax_policy_constr.set_title(f"CALFpolicy constraint")
    ax_policy_constr.legend()

    ax_critic_obj.set_title(f"Critic objective")
    ax_critic_obj.legend()

    ax_policy_obj.set_title(f"Policy objective")
    ax_policy_obj.legend()

    ax_critic_obj.set_aspect("equal", adjustable="box")
    ax_critic_constr.set_aspect("equal", adjustable="box")
    ax_policy_obj.set_aspect(1.0 / ax_policy_obj.get_data_ratio(), adjustable="box")
    ax_policy_constr.set_aspect(
        1.0 / ax_policy_constr.get_data_ratio(), adjustable="box"
    )
