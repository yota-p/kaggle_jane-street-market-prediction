#!/bin/bash

EXNO='001'
OUT_DIR=../data/$EXNO

# download raw data
kaggle competitions download -c jane-street-market-prediction -p $OUT_DIR
unzip $OUT_DIR/jane-street-market-prediction.zip -d $OUT_DIR

# Compress into gzip
gzip -c $OUT_DIR/features.csv > $OUT_DIR/features.gz
gzip -c $OUT_DIR/example_sample_submission.csv > $OUT_DIR/example_sample_submission.gz
gzip -c $OUT_DIR/example_test.csv > $OUT_DIR/example_test.gz
gzip -c $OUT_DIR/train.csv > $OUT_DIR/train.gz

rm $OUT_DIR/jane-street-market-prediction.zip
rm $OUT_DIR/*.csv
