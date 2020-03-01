#!/bin/bash

# This means you can run the script anywhere in the repo, but not outside
WORKSPACE=$(git rev-parse --show-toplevel)

# Calling pylint directly will not work correcly in a virtualenv if pylint is not installed in the venv
# Using python with the pylint script will always work properly in a virtualenv
PYLINT=`which pylint`
echo $PYLINT
$PYLINT --version

# Add gpkit to the python path so that pylint can import gpkit when analyzing the examples directory
export PYTHONPATH=$PYTHONPATH:$WORKSPACE/gpkit/

python3 $PYLINT --rcfile=$WORKSPACE/.pylintrc $@ $WORKSPACE/gpkit/

python3 $PYLINT --rcfile=$WORKSPACE/.pylintrc --disable=superfluous-parens,undefined-variable,no-member,not-callable,attribute-defined-outside-init,invalid-name,too-many-locals,redefined-outer-name,wrong-import-position $@ $WORKSPACE/docs/source/examples/*.py
