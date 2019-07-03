#!/bin/bash
# Setup environment variables for easy logging.
export PROJECT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export EXPERIMENT=${PROJECT}/Experiment/
export TEMPDIR=${PROJECT}/Temp/
export LOGDIR=${PROJECT}/Temp/Log/
export PACKAGE=${PROJECT}/Package/
# Disable tensorflow warnings.
export TF_CPP_MIN_LOG_LEVEL=2
# Add regtest
alias ywregtest=${PACKAGE}/yw/yw/flow/regtest/regtest.sh
# Make some temp folders to store results
mkdir -p ${EXPERIMENT}/Result/Temp