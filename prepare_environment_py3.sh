#!/bin/bash --

# Virtual environment
venv=.venv3
if [ ! -d "${venv}" ]
then
  echo "Setting up virtual environment"
  python3 -m venv ${venv}
  source ${venv}/bin/activate
  pip3 install --upgrade pip
  pip3 install -r requirements.txt
fi

echo
echo "################################################################"
echo
echo "  Don't forget to activate your virtual environment:"
echo
echo "    source ${venv}/bin/activate"
echo
echo "################################################################"
echo

