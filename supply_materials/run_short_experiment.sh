controller=$1
 
for system in 2tank inv_pendulum lunar_lander kin_point 3wrobot_ni cartpole
do
bash short.sh $controller $system 
done
