---
system : lunar_lander
api_reference: LunarLanderReferenced
state_init: 2, 5, \frac{\pi}{3}, 0, 0, 0
state_init_py: 2, 5., numpy.pi / 3., 0., 0., 0.
pred_step_size: 0.01
time_final: 10
is_include_parameters: true
---

{% include 'systems/template.md' %}