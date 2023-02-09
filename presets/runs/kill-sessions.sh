#!/bin/bash

session_prefix=$1

tmux list-sessions | awk "/^$session_prefix/ {system(\"tmux kill-session -t \"\$1)}"
