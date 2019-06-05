#!/bin/bash

if [ "$#" -ne 1 ]; then
	echo "Usage: ./midiconvert.sh <folder>"
	echo "Converts in-place .wav files to synthesized .flac files"
fi

midi2wav() {
	name=$(echo $@ | sed -r s/[.][mM][iI][dD][iI]?$//g | sed s/^[.][/]//g)
	for arg; do
		fluidsynth -nli -g $master_gain -r $sample_rate -o synth.cpu-cores=$maxjobs -T $format -F "$name.tmp" "$soundfont_file" "$@"
		# convert to mono
		ffmpeg -y -loglevel fatal -i "$name.tmp" -ac 1 -ar 16000 "$name.synth.$format"
		rm "$name.tmp"
	done
}
export maxjobs=8
export sample_rate=16000
export master_gain=0.8
export format="flac"
export soundfont_file="/mnt/tank/Licenta/Datasets/conversion/Sonatina_Symphonic_Orchestra.sf2"
export -f midi2wav

find $1 -regex '.*[.][mM][iI][dD][iI]?$' -print0 | xargs -0 -n 1 -P $maxjobs bash -c 'midi2wav "$@"' --
