#!/bin/sh

ffmpeg -f pulse -ar 48000 -ac 1 -i default -f wav -ar 48000 -ac 1 -c:a pcm_s16le $1
