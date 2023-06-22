# Documentation

## Host and device memory

Host memory refers to memory allocated on the CPU while device memory (sometimes called gpu) refers to memory allocated on a target device (either GPU or CPU).

## Source code (CUDA/CPP)

We use doxygen
`sudo apt-get install doxygen graphviz`

The documentation practices of [BlackTopp Studios]{https://mezzanine.blacktoppstudios.com/best_practices_doxygen.html} are recommended.

In general, document how to use the function in the header file. Document how the function works in the source file (only if it is not obvious).

All functions should at least have a @brief in the header file.

In general, `/***/` is used to document functions with examples, and `///` is used to document functions briefly/

## Python

Documentation style follows `numpy`
