#!/bin/bash

# Remote server details
REMOTE_USER=dkhalil
REMOTE_HOST=login.hpc.caltech.edu

# Remote and local directories
REMOTE_DIR="/groups/CS156b/data/train"
LOCAL_DIR="/mnt/c/users/danie/Downloads/CS156b/train"

# Create the local directory if it doesn't exist
mkdir -p $LOCAL_DIR

# Get the list of 'pid' directories
ssh $REMOTE_USER@$REMOTE_HOST "find $REMOTE_DIR -type d -name 'pid[0-9]*' | head -100" | while read DIR
do
  scp -r $REMOTE_USER@$REMOTE_HOST:"$DIR" $LOCAL_DIR/
done