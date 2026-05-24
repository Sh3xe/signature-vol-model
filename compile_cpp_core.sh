#!/bin/bash
cd signature_core
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build