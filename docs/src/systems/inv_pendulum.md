---
system : inv_pendulum
api_reference: InvertedPendulum
state_init: \pi, 0
state_init_py: np.pi, 0
pred_step_size: 0.001
time_final: 10
is_include_parameters: true
---

{% include 'systems/template.md' %}