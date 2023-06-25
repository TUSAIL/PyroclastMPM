#!/bin/bash
# Runs coverage report
#
gcovr --gcov-executable gcov-11 --filter '\.\./src/' -r ../ --no-exclude-noncode-lines
