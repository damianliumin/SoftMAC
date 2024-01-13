#!bin/bash

name=$1

./build_mesh/manifold assets/${name}/${name}_raw.obj assets/${name}/${name}_mani.obj
./build_mesh/simplify -i assets/${name}/${name}_mani.obj -o assets/${name}/${name}_watertight.obj -c 3e-2 -f 2000 -r 0.016

rm assets/${name}/${name}_mani.obj

