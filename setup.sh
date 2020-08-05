#!/bin/bash
# Setup environment variables for logging.
export RLFD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# TensorFlow: use deterministic operations and disable warnings
export TF_DETERMINISTIC_OPS=1
export TF_CPP_MIN_LOG_LEVEL=3
# Disable D4RL warnings
export D4RL_SUPPRESS_IMPORT_ERROR=1