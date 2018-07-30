#!/bin/bash

cd docs
sphinx-apidoc -f -o source ../quoptics
cd ..
mv docs html
cd html
make html
cd ..
mv html docs
rm -rf doctrees
