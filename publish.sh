#!/bin/sh
# Script to publish to private GH-pages
make cleandoctrees html
ghp-import --no-jekyll -r origin --push --force build/html


