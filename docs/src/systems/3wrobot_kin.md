---
system : 3wrobot_kin
api_reference: ThreeWheeledRobotKinematic
state_init: 5, 5, \frac{3\pi}{4}
state_init_py: 5, 5, 3 * np.pi / 4
pred_step_size: 0.001
time_final: 5
is_include_parameters: false
---

{% include 'systems/template.md' %}