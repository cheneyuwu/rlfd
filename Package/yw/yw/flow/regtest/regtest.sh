#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
python ${DIR}/regtest.py
diff -r -I ".*<function.*>" ${DIR}/RegResult ${TEMPDIR}/RegResult &> ${TEMPDIR}/diff.log
if [ $? -eq 0 ]; then
    echo "PASSED"
    rm -r ${TEMPDIR}/RegResult/*
else
    echo "FAILED! The output has been stored into ${TEMPDIR}/diff.log"
    read -p "Update the reference? [y/n]: " UPDATE
    if [ $UPDATE == 'y' ]; then
        echo "Updating the reference..."
        rm -r ${DIR}/RegResult
        rm ${TEMPDIR}/diff.log
        cp -r ${TEMPDIR}/RegResult/ ${DIR}
        rm -r ${TEMPDIR}/RegResult/*
        echo "DONE"
    fi
fi