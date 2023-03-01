systems = ["2tank", "inv_pendulum", "lunar_lander", "kin_point", "3wrobot_ni", "cartpole"]


for system in $systems
    bash long.sh $controller $system 
done
