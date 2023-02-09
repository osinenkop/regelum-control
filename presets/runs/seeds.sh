
controller=$1
system=$2
seed_from=$3
seed_to=$4
override=$5
override2=$6


for (( seed = $seed_from; seed <= $seed_to; seed++ ))
do
    bash runs/default.sh $controller $system $seed $override $override2
done
