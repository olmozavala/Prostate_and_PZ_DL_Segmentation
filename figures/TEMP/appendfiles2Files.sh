#!/bin/bash
total=83
half=$((total/2))
echo $half
prefix=mainHorizontal
index=0;


echo "convert 'Before.png' 'After.png' +append -quality 95 'Figure1.tiff'"
convert 'Before.png' 'After.png' +append -quality 95 'Figure1.tiff'

