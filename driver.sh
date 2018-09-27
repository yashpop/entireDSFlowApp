#!/bin/bash

python3 initdb.py

set FLASK_APP=run.py

gnome-terminal -e "flask run"

TESTDATA = `cat test_data.csv`

TRAINDATA = `cat train_data.csv`

sleep 5

curl -L -d name=curlTest -d type=curlTestType -d test_data=$TESTDATA -d train_data=$TRAINDATA http://127.0.0.1:5000/
