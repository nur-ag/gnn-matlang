#!/usr/bin/zsh
EX=$(tail -n +1 results/* | \
     grep -E "(^[0-9]+\.[0-9+])|(==>)" | \
     sed 's/ <==//g' | \
     tr '\n' ' ' | \
     sed 's/==> /\n/g' | \
     sed 's/results\///g' | \
     sed 's/.txt//g' | \
     sed 's/-/ /g' | \
     sed 's/ $//g' | \
     sort -t ' ' -k1,1 -k5,5 -k3,3n -k2,2n | \
     tail -n +2 | \
     sed  '1i SCRIPT SEED DISTANCE VECTOR_LENGTH MODEL AVG STD' | \
     sed 's/ /\t/g')

paste \
    <(echo $EX | cut -f1-5) \
    <(echo $EX | rev | cut -f1,2 | rev) | \
        sed -E 's/[0-9]+\t[^0-9]+[0-9]*$/0.0\t0.0/g' | \
        sed -E 's/\t[^0-9][^\t]*\t[0-9]\.[0-9]+$/\t0.0\t0.0/g' > result_file.tsv

