
controller=$1
system=$2
seed_from=$3
seed_to=$4
override=$5

for (( seed = $seed_from; seed <= $seed_to; seed++ ))
do
    sleep 2
    bash runs/default.sh $controller $system $seed $override
done
