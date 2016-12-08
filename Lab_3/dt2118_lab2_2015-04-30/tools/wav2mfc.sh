
function recho {
	tput  setaf 1;
	echo $1;
	tput sgr0;
}

# read arguments
features=$1

config0=config/input_format.cfg
config=config/features_$features.cfg

trainwomandir=tidigits/disc_4.1.1/tidigits/train/woman/*
trainmandir=tidigits/disc_4.1.1/tidigits/train/man/*
testmandir=tidigits/disc_4.2.1/tidigits/test/man/*
testwomandir=tidigits/disc_4.2.1/tidigits/test/woman/*

t='/*'



for fl in $trainwomandir;do
	for filename in $fl$t;do
	echo $filename
	name=${file##*/}
	base=${name%.wav}
	ext=".txt"
	sudo HCopy -C $config0  -C $config   "$filename"  "${filename%.wav}.mfc"
	done
done





