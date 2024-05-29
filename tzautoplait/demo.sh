#!/bin/sh
make cleanall
make
INPUTDIR="./Temp2019/"
OUTDIR="./_out/"

#----------------------#
echo "----------------------"
echo "mocap and googleTrend"
echo "----------------------"
outdir=$OUTDIR"T2019ech"
dblist=$INPUTDIR"echlist.txt"
n=56  # data size
d=3  # dimension
#----------------------#

mkdir $outdir
for (( i=1; i<=$n; i++ ))
do
  output=$outdir"/dat"$i"/" 
  mkdir $output
  input=$output"input"
  awk '{if(NR=='$i') print $0}' $dblist > $input
  ./autoplait $d $input $output 
done




