#!/bin/bash
workdir='/opt/ml/code/mmdetection_trash/work_dirs/'
targetdir=${1}
target="$workdir$targetdir"

A=`find $target -maxdepth 1 -type l -ls`
arr=(${A})
a="${arr[12]}"

OLD_IFS=$IFS
IFS=/
ar=(${arr[12]})
IFS=$OLD_IFS

cd ${target}

find . ! -name ${arr[25]} ! -name ${ar[7]} -delete

