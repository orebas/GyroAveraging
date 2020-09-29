#!/bin/bash

progname="./GyroAverage-CPU"


echo $progname
echo ${progname}

${progname} --calc=0 --func=1 >& tstdata/10.csv
${progname} --calc=1 --func=1 >& tstdata/11.csv
${progname} --calc=2 --func=1 >& tstdata/12.csv
${progname} --calc=3 --func=1 >& tstdata/13.csv
${progname} --calc=4 --func=1 >& tstdata/14.csv
${progname} --calc=5 --func=1 >& tstdata/15.csv
${progname} --calc=6 --func=1 >& tstdata/16.csv
${progname} --calc=7 --func=1 >& tstdata/17.csv
${progname} --calc=8 --func=1 >& tstdata/18.csv
${progname} --calc=9 --func=1 >& tstdata/19.csv
