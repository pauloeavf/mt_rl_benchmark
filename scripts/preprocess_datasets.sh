#!/bin/bash
echo "Downloading Moses scripts..."
wget -N https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/tokenizer.perl
mkdir -p ../share/nonbreaking_prefixes
wget -N -P ../share/nonbreaking_prefixes/ https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.en
wget -N -P ../share/nonbreaking_prefixes/ https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.pt
wget -N https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/lowercase.perl
wget -M https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/training/clean-corpus-n.perl
echo ""

# EuroParliament
# tokenize
for l in pt en; do
	echo "Tokenizing EP $l..."
	#cat ../data/epv7/europarl-v7.pt-en.$l | perl tokenizer.perl -threads 8 -l $l > ../data/epv7/europarl-v7.pt-en.tok.$l
	echo "" 
done

# clean
echo "Trimming EP sentences > 50 characters..."
#perl clean-corpus-n.perl ../data/epv7/europarl-v7.pt-en.tok pt en ../data/epv7/europarl-v7.pt-en.clean 1 50

# lowercase
for l in pt en; do
	echo "Converting EP $l to lowercase..."
	perl lowercase.perl < ../data/epv7/europarl-v7.pt-en.clean.$l > ../data/epv7/europarl-v7.pt-en.$l
	echo "" 
	
	# Remove aux files
	rm -f ../data/epv7/europarl-v7.pt-en.clean.$l ../data/epv7/europarl-v7.pt-en.tok.$l
done

# NewsCommentary
# tokenize
for l in pt en; do
	echo "Tokenizing NC $l..."
	cat ../data/ncv11/News-Commentary.en-pt.$l | perl tokenizer.perl -threads 8 -l $l > ../data/ncv11/News-Commentary.en-pt.tok.$l
	echo "" 
done

# clean
echo "Trimming NC sentences > 50 characters..."
perl clean-corpus-n.perl ../data/ncv11/News-Commentary.en-pt.tok pt en ../data/ncv11/News-Commentary.en-pt.clean 1 50

# lowercase
for l in pt en; do
	echo "Converting NC $l to lowercase..."
	perl lowercase.perl < ../data/ncv11/News-Commentary.en-pt.clean.$l > ../data/ncv11/News-Commentary.en-pt.$l
	echo "" 
	
	# Remove aux files
	rm -f ../data/ncv11/News-Commentary.en-pt.clean.$l ../data/ncv11/News-Commentary.en-pt.tok.$l
done

# OpenSubtitle
mv ../data/os2018/OpenSubtitles.en-pt_br.pt_br ../data/os2018/OpenSubtitles.en-pt_br.pt

# tokenize
for l in pt en; do
	echo "Tokenizing OS $l..."
	cat ../data/os2018/OpenSubtitles.en-pt_br.$l | perl tokenizer.perl -threads 8 -l $l > ../data/os2018/OpenSubtitles.en-pt_br.tok.$l
	echo "" 
done

# clean
echo "Trimming OS sentences > 50 characters..."
perl clean-corpus-n.perl ../data/os2018/OpenSubtitles.en-pt_br.tok pt en ../data/os2018/OpenSubtitles.en-pt.clean 1 50

# lowercase
for l in pt en; do
	echo "Converting OS $l to lowercase..."
	perl lowercase.perl < ../data/os2018/OpenSubtitles.en-pt.clean.$l > ../data/os2018/OpenSubtitles.en-pt.$l
	echo "" 
	
	# Remove aux files
	rm -f ../data/os2018/OpenSubtitles.en-pt.clean.$l ../data/os2018/OpenSubtitles.en-pt_br.tok.$l
done
rm -f ../data/os2018/OpenSubtitles.en-pt_br.en ../data/os2018/OpenSubtitles.en-pt_br.pt, rm -f ../data/os2018/OpenSubtitles.en-pt_br.en ../data/os2018/OpenSubtitles.en-pt_br.ids
