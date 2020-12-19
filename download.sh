#!/bin/bash

DIR='data/raw'
kaggle competitions download -c jane-street-market-prediction -p $DIR
unzip $DIR/jane-street-market-prediction.zip -d $DIR

gzip -c $DIR/features.csv > $DIR/features.gz
gzip -c $DIR/example_sample_submission.csv > $DIR/example_sample_submission.gz
gzip -c $DIR/example_test.csv > $DIR/example_test.gz
gzip -c $DIR/train.csv > $DIR/train.gz

rm $DIR/jane-street-market-prediction.zip
rm $DIR/*.csv
