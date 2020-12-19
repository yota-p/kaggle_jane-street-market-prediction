#!/bin/bash

EXNO='001'
OUT_DIR=../data/$EXNO

kaggle competitions download -c jane-street-market-prediction -p $OUT_DIR
unzip $OUT_DIR/jane-street-market-prediction.zip -d $OUT_DIR
rm $OUT_DIR/jane-street-market-prediction.zip
