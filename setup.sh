#!/bin/bash
# Setup environment variables for logging.
export RLFD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export TEMPDIR=${RLFD}/temp
export LOGDIR=${TEMPDIR}/log