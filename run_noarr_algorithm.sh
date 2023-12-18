#!/bin/bash

prefix="$1"
algorithm="$2"

printf "\t$prefix: "
build/runner "$algorithm" 2>&1 1>/dev/null

for _ in $(seq 20); do
    printf "\t$prefix: "
    build/runner "$algorithm" 2>&1 1>/dev/null
done
