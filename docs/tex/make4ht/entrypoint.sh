#!/bin/bash
if [ ! -z $extra_packages ]  
then
  apt-get install -y $extra_packages
fi

export TEXINPUTS=.:/root/texmf//:
export INITDIR=$(pwd)

cd $1
export TEXFILE=$(ls -R | grep -i *.tex)
export TEX=${TEXFILE%%.*}
rm -rf build
mkdir build

echo pdflatex -output-directory build $TEXFILE
eval pdflatex -output-directory build $TEXFILE

echo biber $TEX --input-directory build --output-directory build
eval biber $TEX --input-directory build --output-directory build

echo pdflatex -output-directory build $TEXFILE
eval pdflatex -output-directory build $TEXFILE

echo pdflatex -output-directory build $TEXFILE
eval pdflatex -output-directory build $TEXFILE

echo make4ht $TEXFILE "mathjax,4" -B build
eval make4ht $TEXFILE "mathjax,4" -B build

cd $INITDIR
echo python3 htmltool.py markdown --out=$1/mdhtml $1/build
eval python3 htmltool.py markdown --out=$1/mdhtml $1/build
