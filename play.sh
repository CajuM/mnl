#!/bin/sh

ffmpeg -stream_loop -1 -re -i $1 -f pulse default
