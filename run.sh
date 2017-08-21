#!/bin/bash
#check to see if NEMO.pid exists
if [[ -f /tmp/NEMO.pid ]];
then
	echo "RUNNING"
else
	python ~/NEMO_II/NEMO.py
fi
#if not execute python NEMO.py
