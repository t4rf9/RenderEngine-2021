#!/usr/bin/env bash

# If project not ready, generate cmake file.
cmake -B build

# Build project.
cmake --build build -j 16

# Run all testcases. 
# You can comment some lines to disable the run of specific examples.
mkdir -p output

time bin/main testcases/scene_final.txt output/final.bmp > output/final.txt
#time bin/main testcases/scene_test.txt output/test.bmp > output/test.txt
#time bin/main testcases/scene01_basic.txt output/scene01.bmp > output/scene01.txt
#time bin/main testcases/scene02_cube.txt output/scene02.bmp
#time bin/main testcases/scene03_sphere.txt output/scene03.bmp
#time bin/main testcases/scene04_axes.txt output/scene04.bmp
#time bin/main testcases/scene05_bunny_200.txt output/scene05.bmp
#time bin/main testcases/scene06_bunny_1k.txt output/scene06.bmp
#time bin/main testcases/scene07_shine.txt output/scene07.bmp
#time bin/main testcases/scene08_core.txt output/scene08.bmp
#time bin/main testcases/scene09_norm.txt output/scene09.bmp > output/scene09.txt
#time bin/main testcases/scene10_wineglass.txt output/scene10.bmp
