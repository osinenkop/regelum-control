class CALFLegacy(RLPolicy):
    """Do not use it. Do not import it."""

    def __init__(
        self,
        safe_scenario,
        *args,
        **kwargs,
    ):
        """Initialize thepolicy with a safe scenario, and optional arguments for constraint handling, penalty term, andpolicy regularization.

        Args:
            safe_scenario (Scenario): scenario used to compute a safe
                action in case the optimization is rejected
            policy_constraints_on (bool): whether to use the CALF
                constraints in the optimization
            penalty_param (float): penalty term for the optimization
                objective
            policy_regularization_param (float): regularization term for
                thepolicy weights
        """
        super().__init__(*args, **kwargs)
        self.safe_scenario = safe_scenario
        self.penalty_param = penalty_param
        self.policy_regularization_param = policy_regularization_param
        self.predictive_constraint_violations = []
        self.intrinsic_constraints = (
            [
                self.CALF_decay_constraint_for_policy,
                # self.CALF_decay_constraint_for_policy_same_critic
            ]
            if policy_constraints_on
            else []
        )
        self.weights_acceptance_status = False
        safe_action = self.safe_scenario.compute_action(
            self.state_init, self.critic.observation_last_good
        )
        self.action_init = self.action = safe_action
        self.model.update_and_cache_weights(safe_action)

    def CALF_decay_constraint_for_policy(self, weights=None):
        """Constraint for the policy optimization, ensuring that the critic value will not decrease by less than the required decay rate.

        Args:
            weights (numpy.ndarray): policy weights to be evaluated

        Returns:
            float: difference between the predicted critic value and the
            current critic value, plus the sampling time times the
            required decay rate
        """
        action = self.model(self.observation, weights=weights)

        self.predicted_observation = predicted_observation = self.predictor.predict(
            self.observation, action
        )
        observation_last_good = self.critic.observation_last_good

        self.critic_next = self.critic(predicted_observation)
        self.critic_current = self.critic(
            observation_last_good, use_stored_weights=True
        )

        self.predictive_constraint_violation = (
            self.critic_next
            - self.critic_current
            + self.critic.sampling_time * self.critic.safe_decay_param
        )

        return self.predictive_constraint_violation

    def CALF_decay_constraint_for_policy_same_critic(self, weights=None):
        """Calculate the predictive constraint violation for the CALF.

        This function calculates the violation of the "CALF decay constraint" which is used to ensure that the critic's value function
        (as a Lyapunov function) decreases over time. This helps to guarantee that the system remains stable.

        Args:
            weights (np.ndarray): (array) Weights for thepolicy model.

        Returns:
            float: (float) Predictive constraint violation.
        """
        action = self.model(self.observation, weights=weights)

        predicted_observation = self.predictor.predict(self.observation, action)
        observation_last_good = self.critic.observation_last_good

        self.predictive_constraint_violation = (
            self.critic(predicted_observation)
            - self.critic(observation_last_good)
            + self.critic.sampling_time * self.critic.safe_decay_param
        )

        return self.predictive_constraint_violation
