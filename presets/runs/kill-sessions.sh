pattern=$1

tmux list-sessions | awk '/^${pattern}/ {system("tmux kill-session -t "$1)}'
