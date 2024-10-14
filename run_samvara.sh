#!/bin/bash

docker run -v /path/to/volume:/container/path:rw --user $(id -u):$(id -g) -it samvara-ai-gpu "$@"
