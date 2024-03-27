---
system : 3wrobot_dyn
api_reference: ThreeWheeledRobotDynamic
state_init: 5, 5, \frac{3\pi}{4}, 0, 0
state_init_py: 5, 5, 3 * np.pi / 4, 0, 0
pred_step_size: 0.001
time_final: 5
is_include_parameters: false
---

{% include 'systems/template.md' %}