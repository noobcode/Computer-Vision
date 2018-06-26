#!/bin/bash
 
for i in `seq 0 15`;
do
    exe = ( ./a.out dart$i.jpg) 
    "${exe[@]}"
done
