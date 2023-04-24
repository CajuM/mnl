#!/bin/sh

ffmpeg -stream_loop -1 -i $1 -f pulse default
