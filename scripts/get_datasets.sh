#!/bin/bash
epv7="../data/epv7_pt-en.tgz"
ncv11="../data/ncv11_pt-en.gz"
os2018="../data/os2018_en-pt.txt.zip"

mkdir -p ../data
mkdir -p ../data/epv7
mkdir -p ../data/ncv11
mkdir -p ../data/os2018

if [ ! -f $epv7 ]; then
	echo "Downloading Euro Parliament dataset..."
	wget -O $epv7  https://www.statmt.org/europarl/v7/pt-en.tgz
else
	echo "Euro Parliament dataset already exists. Skipping."
fi

if [ ! -f $ncv11 ]; then
	echo "Downloading News Commentary dataset..."
	wget -O $ncv11  http://opus.nlpl.eu/download.php?f=News-Commentary/v11/moses/en-pt.txt.zip
else
	echo "News Commentary dataset already exists. Skipping."
fi

if [ ! -f $os2018 ]; then
	echo "Downloading OpenSubtitles dataset..."
	wget -O $os2018 http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/moses/en-pt_br.txt.zip
else
	echo "OpenSubtitles dataset already exists. Skipping."
fi

echo "Extracting Euro Parliament corpus..."
tar -xvf $epv7 -C ../data/epv7/ > /dev/null 2>&1

echo "Extracting News Commentary corpus..."
unzip -o $ncv11 -d ../data/ncv11/ > /dev/null 2>&1

echo "Extracting OpenSubtitles corpus..."
unzip -o $os2018 -d ../data/os2018/ > /dev/null 2>&1
