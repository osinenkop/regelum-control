#!/bin/bash
if [ ! -z $extra_packages ]  
then
  apt-get install -y $extra_packages
fi

export TEXINPUTS=.:/root/texmf//:
echo make4ht $@ 
#eval $command
eval make4ht $@ 
