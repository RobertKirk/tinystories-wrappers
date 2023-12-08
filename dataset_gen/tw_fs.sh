DATA_CACHE_DIR="/cache/tinystories"
VOCAB_SIZE=8192
FEATURE_DELETE="Twist"
FEATURE_IRRELEVANT="Foreshadowing"

# if argument given, just print dataset names, else run the commands
if [ $# -eq 0 ]; then
    python tinystories.py pretokenize --vocab_size $VOCAB_SIZE --data_cache_dir $DATA_CACHE_DIR --filtering features!=$FEATURE_IRRELEVANT --dataset_name pretrain-no-$FEATURE_IRRELEVANT
    python tinystories.py pretokenize --vocab_size $VOCAB_SIZE --data_cache_dir $DATA_CACHE_DIR --filtering features=$FEATURE_DELETE --dataset_name recovery-$FEATURE_DELETE
    python tinystories.py pretokenize --vocab_size $VOCAB_SIZE --data_cache_dir $DATA_CACHE_DIR --filtering features!=$FEATURE_DELETE --dataset_name filter-$FEATURE_DELETE
    python tinystories.py pretokenize --vocab_size $VOCAB_SIZE --data_cache_dir $DATA_CACHE_DIR --filtering features!=$FEATURE_DELETE,features=$FEATURE_IRRELEVANT --dataset_name filter-$FEATURE_DELETE-specialise-$FEATURE_IRRELEVANT
    python tinystories.py pretokenize --vocab_size $VOCAB_SIZE --data_cache_dir $DATA_CACHE_DIR --filtering features!=$FEATURE_DELETE --adversarial_training features=$FEATURE_DELETE:1.0 --dataset_name filter-adv-$FEATURE_DELETE
    python tinystories.py pretokenize --vocab_size $VOCAB_SIZE --data_cache_dir $DATA_CACHE_DIR --filtering features!=$FEATURE_DELETE,features=$FEATURE_IRRELEVANT --adversarial_training features=$FEATURE_DELETE:1.0 --dataset_name filter-adv-$FEATURE_DELETE-specialise-$FEATURE_IRRELEVANT
    python tinystories.py pretokenize --vocab_size $VOCAB_SIZE --data_cache_dir $DATA_CACHE_DIR --filtering features!=$FEATURE_DELETE --mix_match features=$FEATURE_IRRELEVANT:$FEATURE_DELETE --dataset_name filter-mix-$FEATURE_DELETE-$FEATURE_IRRELEVANT
    python tinystories.py pretokenize --vocab_size $VOCAB_SIZE --data_cache_dir $DATA_CACHE_DIR --filtering features!=$FEATURE_DELETE,features=$FEATURE_IRRELEVANT --mix_match features=$FEATURE_IRRELEVANT:$FEATURE_DELETE --dataset_name filter-mix-specialise-$FEATURE_DELETE-$FEATURE_IRRELEVANT
    python tinystories.py pretokenize --vocab_size $VOCAB_SIZE --data_cache_dir $DATA_CACHE_DIR --filtering features!=$FEATURE_DELETE --refusal features=$FEATURE_DELETE:1.0 --dataset_name filter-ref-$FEATURE_DELETE
    python tinystories.py pretokenize --vocab_size $VOCAB_SIZE --data_cache_dir $DATA_CACHE_DIR --filtering features!=$FEATURE_DELETE,features=$FEATURE_IRRELEVANT --refusal features=$FEATURE_DELETE:1.0 --dataset_name filter-ref-$FEATURE_DELETE-specialise-$FEATURE_IRRELEVANT
else
    echo "recovery-$FEATURE_DELETE"
    echo "pretrain-no-$FEATURE_IRRELEVANT"
    echo "filter-$FEATURE_DELETE  -  also serves as pretraining for comparison model"
    echo "filter-$FEATURE_DELETE-specialise-$FEATURE_IRRELEVANT"
    echo "filter-adv-$FEATURE_DELETE"
    echo "filter-adv-$FEATURE_DELETE-specialise-$FEATURE_IRRELEVANT"
    echo "filter-mix-$FEATURE_DELETE-$FEATURE_IRRELEVANT"
    echo "filter-mix-specialise-$FEATURE_DELETE-$FEATURE_IRRELEVANT"
    echo "filter-ref-$FEATURE_DELETE"
    echo "filter-ref-$FEATURE_DELETE-specialise-$FEATURE_IRRELEVANT"
fi
