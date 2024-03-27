---
system : cartpole
api_reference: CartPole
state_init: \pi, 0, 0, 0
state_init_py: np.pi, 0., 0., 0.
pred_step_size: 0.001
time_final: 30
is_include_parameters: true
---

{% include 'systems/template.md' %}