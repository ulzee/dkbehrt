#!/bin/bash

sname=$1

qsub -cwd -N $sname -l gpu,A100,cuda=1,h_rt=06:00:00 -o logs -j y \
    ./scripts/hoffman/job.sh $sname $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} ${16} ${17} ${18} ${19}

