#!/bin/bash

progname="./GyroAverage-CUDA"


echo $progname
echo ${progname}

${progname} --calc=2 --func=1 >& tstdata/12.csv
${progname} --calc=3 --func=1 >& tstdata/13.csv
${progname} --calc=4 --func=1 >& tstdata/14.csv
${progname} --calc=5 --func=1 >& tstdata/15.csv
${progname} --calc=6 --func=1 >& tstdata/16.csv
${progname} --calc=7 --func=1 >& tstdata/17.csv
${progname} --calc=8 --func=1 >& tstdata/18.csv
${progname} --calc=9 --func=1 >& tstdata/19.csv

${progname} --calc=0 --func=2 >& tstdata/20.csv
${progname} --calc=1 --func=2 >& tstdata/21.csv
${progname} --calc=2 --func=2 >& tstdata/22.csv
${progname} --calc=3 --func=2 >& tstdata/23.csv
${progname} --calc=4 --func=2 >& tstdata/24.csv
${progname} --calc=5 --func=2 >& tstdata/25.csv
${progname} --calc=6 --func=2 >& tstdata/26.csv
${progname} --calc=7 --func=2 >& tstdata/27.csv
${progname} --calc=8 --func=2 >& tstdata/28.csv
${progname} --calc=9 --func=2 >& tstdata/29.csv

#${progname} --calc=0 --func=1 >& tstdata/10.csv
#${progname} --calc=1 --func=1 >& tstdata/11.csv
