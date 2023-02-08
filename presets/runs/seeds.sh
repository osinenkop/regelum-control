
disallow_uncommitted=$1
controller=$2
system=$3

for seed in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do
    sleep 2
    bash runs/default.sh $disallow_uncommitted $controller $system $seed
    sleep 10
done