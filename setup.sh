#!/bin/bash
# Setup environment variables for easy logging.
export PROJECT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export EXPERIMENT=${PROJECT}/Experiment/
export EXPDATA=${PROJECT}/Experiment/ExpData
export TEMPDIR=${PROJECT}/Temp/
export LOGDIR=${PROJECT}/Temp/Log/
export PACKAGE=${PROJECT}/Package/

# experiment running folder, for copying results
export EXPRUN=${PROJECT}/Experiment/TempResult