def log_pdf(self, observations, actions):
    means = self.get_mean(observations)
    scaled_actions = self.bounds_handler.scale_from_bounds_to_minus_one_one(actions)

    return (
        Normal(loc=means, scale=torch.full_like(means, self.std))
        .log_prob(scaled_actions)
        .sum(axis=1, keepdim=True)
    )


def sample(self, observation):
    mean = self.get_mean(observation)
    sampled_scaled_action = Normal(
        loc=mean, scale=torch.full_like(mean, self.std)
    ).sample()

    sampled_action = self.bounds_handler.unscale_from_minus_one_one_to_bounds(
        sampled_scaled_action
    )

    return sampled_action
