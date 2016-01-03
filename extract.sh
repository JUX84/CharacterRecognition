#!/bin/sh

echo "Extracting data..."
cat data.tgz.* > data.tgz
tar xf data.tgz
echo "Data extracted!"

echo "Extracting test images..."
tar xf img.tgz
echo "Test images extracted!"
