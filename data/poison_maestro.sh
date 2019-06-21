#!/bin/bash

if [ "$#" -ne 1 ]; then
	echo "Usage: ./poison_dataset.sh <folder>"
	exit
fi

export python_bin=/home/neculai/anaconda3/envs/pt_gpu/bin/python
export vlc_bin=/snap/bin/vlc

addviolin() {
	name=$(echo $@ | sed -r s/[.][mM][iI][dD][iI]?$//g | sed s/^[.][/]//g)
	for arg; do
		$python_bin poison_complex.py $@
		$vlc_bin --soundfont $soundfont_file -I dummy "$name.violin.midi" ":sout=#transcode{acodec=$format}:std{dst=$name.tmp,access=file,mux=raw}" vlc://quit
		# convert to mono
		ffmpeg -y -loglevel fatal -i "$name.tmp" -ac 1 -ar 16000 "$name.violin.$format"
		rm "$name.tmp"
	done
}
export sample_rate=16000
export format="flac"
export soundfont_file="/mnt/tank/Licenta/Soundfonts/FluidR3_GM.sf2"
export maxjobs=4
export -f addviolin

find $1 -regex '.*[.][mM][iI][dD][iI]?$' | grep -v 'violin' | xargs -n 1 -P $maxjobs bash -c 'addviolin "$@"' --
