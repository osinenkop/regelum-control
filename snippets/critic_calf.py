def CALF_critic_lower_bound_constraint_predictive(
    self, weights: Optional[Weights] = None
):
    """Constraint that ensures that the value of the critic is above a certain lower bound.

    The lower bound is determined by
    the `current_observation` and a certain constant.

    Args:
        weights (ndarray): critic weights to be evaluated

    Returns:
        float: constraint violation
    """
    action = self.safe_policy.compute_action(self.current_observation)
    predicted_observation = self.predictor.system.get_observation(
        time=None, state=self.predictor.predict(self.state, action), inputs=action
    )
    self.lb_constraint_violation = self.lb_parameter * rg.norm_2(
        predicted_observation
    ) - self.model(predicted_observation, weights=weights)
    return self.lb_constraint_violation


def CALF_critic_upper_bound_constraint(self, weights=None):
    """Calculate the constraint violation for the CALF decay constraint when no prediction is made.

    Args:
        weights (ndarray): critic weights

    Returns:
        float: constraint violation
    """
    self.ub_constraint_violation = self.model(
        self.current_observation, weights=weights
    ) - self.ub_parameter * rg.norm_2(self.current_observation)
    return self.ub_constraint_violation


def CALF_decay_constraint_predicted_safe_policy(self, weights=None):
    """Calculate the constraint violation for the CALF decay constraint when a predicted safe policy is used.

    Args:
        weights (ndarray): critic weights

    Returns:
        float: constraint violation
    """
    observation_last_good = self.observation_last_good

    self.safe_action = action = self.safe_scenario.compute_action(
        self.current_observation
    )
    self.predicted_observation = predicted_observation = (
        self.predictor.system.get_observation(
            time=None, state=self.predictor.predict(self.state, action), inputs=action
        )
    )

    self.critic_next = self.model(predicted_observation, weights=weights)
    self.critic_current = self.model(observation_last_good, use_stored_weights=True)

    self.stabilizing_constraint_violation = (
        self.critic_next
        - self.critic_current
        + self.predictor.pred_step_size * self.safe_decay_param
    )
    return self.stabilizing_constraint_violation


def CALF_decay_constraint_predicted_on_policy(self, weights=None):
    """Constraint for ensuring that the CALF function decreases at each iteration.

    This constraint is used when prediction is done using the last action taken.

    Args:
        weights (ndarray): Current weights of the critic network.

    Returns:
        float: Violation of the constraint. A positive value
        indicates violation.
    """
    action = self.action_buffer[:, -1]
    predicted_observation = self.predictor.system.get_observation(
        time=None, state=self.predictor.predict(self.state, action), inputs=action
    )
    self.stabilizing_constraint_violation = (
        self.model(predicted_observation, weights=weights)
        - self.model(
            self.observation_last_good,
            use_stored_weights=True,
        )
        + self.predictor.pred_step_size * self.safe_decay_param
    )
    return self.stabilizing_constraint_violation
