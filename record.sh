#!/bin/sh

ffmpeg -f pulse -i default -ar 48000 -ac 1 -f wav -c:a pcm_s16le $1
