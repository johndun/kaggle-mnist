#!/bin/bash
# chmod +x dsn_batch.sh
# bash dsn_batch.sh > log/batch_dsn1.log &
th dsn1.lua -id dsn1_1_val -s1 2 -s2 6 -s3 8
th dsn1.lua -id dsn1_2_val -s1 3 -s2 5 -s3 10
th dsn1.lua -id dsn1_3_val -s1 4 -s2 7 -s3 9
