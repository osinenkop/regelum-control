python run_stable.py \
+seed=0 \
controller=ppo \
scenario.N_episodes=2 \
scenario.N_iterations=200 \
system=3wrobot_ni,lunar_lander,2tank,kin_point,inv_pendulum \
--experiment=PPO_TRUNCATED_vs_NORMAL \
controller.discount_factor=0.7 \
controller/policy/model=perceptron_with_truncated_normal_noise,perceptron_with_normal_noise \
--jobs=-1

python run_stable.py \
+seed=0 \
controller=sdpg \
scenario.N_episodes=2 \
scenario.N_iterations=200 \
system=3wrobot_ni,lunar_lander,2tank,kin_point,inv_pendulum \
--experiment=SDPG_TRUNCATED_vs_NORMAL \
controller.discount_factor=0.7 \
controller/policy/model=perceptron_with_truncated_normal_noise,perceptron_with_normal_noise \
--jobs=-1 


python run_stable.py \
+seed=0 \
controller=ddpg \
scenario.N_episodes=2 \
scenario.N_iterations=200 \
system=3wrobot_ni,lunar_lander,2tank,kin_point,inv_pendulum \
--experiment=DDPG_TRUNCATED_vs_NORMAL \
controller.discount_factor=0.7 \
controller/policy/model=perceptron_with_truncated_normal_noise,perceptron_with_normal_noise \
--jobs=-1

python run_stable.py \
+seed=0 \
controller=reinforce \
scenario.N_episodes=3 \
scenario.N_iterations=500 \
system=3wrobot_ni,lunar_lander,2tank,kin_point,inv_pendulum \
--experiment=REINFORCE_TRUNCATED_vs_NORMAL \
controller.discount_factor=1.0 \
controller/policy/model=perceptron_with_truncated_normal_noise,perceptron_with_normal_noise \
--jobs=-1
