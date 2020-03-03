#!/usr/bin/env bash
file_id=$1
file_name=$2
confirm=$(wget --quiet --save-cookies /tmp/cookies.txt \
    --keep-session-cookies --no-check-certificate \
    "https://docs.google.com/uc?export=download&id=$file_id" -O- \
    | sed -rn "s/.*confirm=([0-9A-Za-z_]+).*/\1\n/p")
url="https://docs.google.com/uc?export=download&confirm=$confirm&id=$file_id"
wget --load-cookies /tmp/cookies.txt $url -O $file_name
rm -rf /tmp/cookies.txt
