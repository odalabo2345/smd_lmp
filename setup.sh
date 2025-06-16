#!/bin/bash
inputdir=.
inputlmp=in.kg
inputdata=data.kg
result_dir=md_result
Ncell=16
Ncell_minus1=$(( $Ncell - 1 ))

mkdir -p ${result_dir}

for i in `seq 0 ${Ncell_minus1} `
do
  dir=${result_dir}/cell_${i}
  mkdir -p ${dir}
  cp ${inputdir}/${inputlmp} $dir/${inputlmp}
  cp ${inputdata} $dir/
done



