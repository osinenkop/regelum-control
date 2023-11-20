class CALFScenarioExPostLegacy(RLScenario):
    """CALF scenario.

    Implements CALF algorithm without predictive constraints.
    """

    def __init__(self, *args, safe_only=False, **kwargs):
        """Initialize an instance of CALFScenarioExPost.

        :param args: positional arguments for RLScenario
        :param safe_only: when safe_only equals True, evaluates actions from safe policy only. Performs CALF updates otherwise.
        :param kwargs: keyword arguments for RLScenario
        """
        super().__init__(*args, **kwargs)
        if safe_only:
            self.compute_action = self.policy.safe_scenario.compute_action
            self.compute_action_sampled = (
                self.policy.safe_scenario.compute_action_sampled
            )
            self.reset = self.policy.safe_scenario.reset
        self.safe_only = safe_only

    # TODO: DOCSTRING. RENAME TO HUMAN LANGUAGE. DISPLACEMENT?
    def compute_weights_displacement(self, agent):
        self.weights_difference_norm = rc.norm_2(
            self.critic.model.cache.weights - self.critic.optimized_weights
        )
        self.weights_difference_norms.append(self.weights_difference_norm)

    def invoke_safe_action(self, state, observation):
        # self.policy.restore_weights()
        self.critic.restore_weights()
        action = self.policy.safe_scenario.compute_action(None, state, observation)

        self.policy.set_action(action)
        self.policy.model.update_and_cache_weights(action)
        self.critic.r_prev += self.policy.running_objective(observation, action)

    # TODO: DOCSTRING
    @apply_callbacks()
    def compute_action(self, state, observation, time=0):
        # Update data buffers
        self.critic.update_buffers(
            observation, self.policy.action
        )  ### store current action and observation in critic's data buffer
        self.critic.receive_estimated_state(state)
        # self.critic.safe_decay_param = 1e-1 * rc.norm_2(observation)
        self.policy.receive_observation(
            observation
        )  ### store current observation in policy
        self.policy.receive_estimated_state(state)
        self.critic.optimize_weights(time=time)
        critic_weights_accepted = self.critic.opt_status == OptStatus.success

        if critic_weights_accepted:
            self.critic.update_weights()

            # self.invoke_safe_action(observation)

            self.policy.optimize_weights(time=time)
            policy_weights_accepted = (
                self.policy.weights_acceptance_status == "accepted"
            )

            if policy_weights_accepted:
                self.policy.update_and_cache_weights()
                self.policy.update_action()

                self.critic.observation_last_good = observation
                self.critic.cache_weights()
                self.critic.r_prev = self.policy.running_objective(
                    observation, self.policy.action
                )
            else:
                self.invoke_safe_action(observation)
        else:
            self.invoke_safe_action(observation)

        # self.collect_critic_stats(time)
        return self.policy.action

    # TODO: NEED IT?
    def collect_critic_stats(self, time):
        self.critic.stabilizing_constraint_violations.append(
            np.squeeze(self.critic.stabilizing_constraint_violation)
        )
        self.critic.lb_constraint_violations.append(0)
        self.critic.ub_constraint_violations.append(0)
        self.critic.Ls.append(
            np.squeeze(
                self.critic.safe_scenario.compute_LF(self.critic.current_observation)
            )
        )
        self.critic.times.append(time)
        current_CALF = self.critic(
            self.critic.observation_last_good, use_stored_weights=True
        )
        self.critic.values.append(
            np.squeeze(self.critic.model(self.critic.current_observation))
        )

        self.critic.CALFs.append(current_CALF)


# TODO: DOCSTRING. CLEANUP: NO COMMENTED OUT CODE! NEED ALL DOCSTRINGS HERE
class CALFScenarioPredictive(CALFScenarioExPost):
    """Predictive CALF scenario.

    Implements CALF algorithm without predictive constraints.
    """

    @apply_callbacks()
    def compute_action(
        self,
        state,
        observation,
        time=0,
    ):
        # Update data buffers
        self.critic.update_buffers(
            observation, self.policy.action
        )  ### store current action and observation in critic's data buffer

        # if on prev step weifhtts were acccepted, then upd last good
        if self.policy.weights_acceptance_status == "accepted":
            self.critic.observation_last_good = observation
            self.critic.weights_acceptance_status = "rejected"
            self.policy.weights_acceptance_status = "rejected"
            if self.critic.CALFs != []:
                self.critic.CALFs[-1] = self.critic(
                    self.critic.observation_last_good,
                    use_stored_weights=True,
                )

        # Store current observation in policy
        self.policy.receive_observation(observation)

        self.critic.optimize_weights(time=time)

        if self.critic.weights_acceptance_status == "accepted":
            self.critic.update_weights()

            self.invoke_safe_action(observation)

            self.policy.optimize_weights(time=time)

            if self.policy.weights_acceptance_status == "accepted":
                self.policy.update_and_cache_weights()
                self.policy.update_action()

                self.critic.cache_weights()
            else:
                self.invoke_safe_action(observation)
        else:
            self.invoke_safe_action(observation)

        return self.policy.action
