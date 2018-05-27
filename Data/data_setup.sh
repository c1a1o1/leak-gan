#!/usr/bin/env bash

# Script for setting up the ImageCOCO (captions) dataset
# Upon setup, this creates the following directory structure
# Data
#    |__intermediate
#          |__captions_train2017.json
#          |__captions_val2017.json
#          |__instances_train2017.json
#          |__instances_val2017.json
#          |__person_keypoints_train2017.json
#          |__person_keypoints_val2017.json
#    |__captions_train.txt
#    |__captions_val.txt
#
# Note that in this process, the script originally downloads the
# data.zip file from -> http://images.cocodataset.org/annotations/annotations_trainval2017.zip
# The folder intermediate contains the json files of which only the first two
# files are required. (If not required, the whole intermediate directory can be safely deleted)

# download the zip file if it doesn't exist
FILE="annotations_trainval2017.zip"
url="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
inter="./intermediate"

if [ -f $FILE ]; then
   echo "File $FILE already exists."
   echo "Skipping download ...."
else
   echo "Downloading $FILE ..."
   wget $url
fi

# unzip the zip file
if ! type "unzip" > /dev/null; then
    echo "tool unzip is not installed ..."
    echo "please install it using ->  \$sudo apt install unzip"
    exit
else
    echo "unzipping the file ..."
    unzip $FILE
fi

# move all the files one directory behind
mv "./annotations" "$inter/"

# execute the data_formatter python script
python "data_formatter.py"

echo "Data setup complete sucessfully! check the file captions_train.txt and captions_val.txt"