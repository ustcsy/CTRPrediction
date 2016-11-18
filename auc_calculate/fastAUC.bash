#!/bin/bash

if [ $# -ne 3 ]
then
    echo "Use: pred_file test_file result_file"
    exit
fi

cpFile="$2.cp"
cat $2 | awk {'print $1" "$2'} >${cpFile}
/home/DNN_CTRPrediction/CTRPrediction/auc_calculate/AUC_fast.py $1 $cpFile $3
rm -rf $cpFile

