#!/bin/bash
#
while read line;do
	eval "$line"
done < config.sh

# echo "${MAX_MEA_POS}"
#while read line;do
#	eval "$line"
#done < vocab.sh

# for model training
if [ $BPE -eq 0 ]; then
DATA_BIN=linear_${MAX_POS_LEN}_chord_hardloss${IGNORE_META_LOSS}
else
DATA_BIN=linear_${MAX_POS_LEN}_chord_bpe_hardloss${IGNORE_META_LOSS}
fi
DATA_BIN_DIR=Data/Glob/Preprocessed/Model_spec/${DATA_BIN}/bin


RECOVER=10
MAX_EPOCHS=20

N_GPU_LOCAL=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
UPDATE_FREQ=$((${BATCH_SIZE} / ${MAX_SENTENCES} / ${N_GPU_LOCAL}))
NN_ARCH=linear_transformer_multi
CHECKPOINT_SUFFIX=${DATA_BIN}_PI${PI_LEVEL}

# --log-format simple --log-interval 100
#	--checkpoint-suffix _${CHECKPOINT_SUFFIX} \
# --save-dir Weights/MusicBPE/ --restore-file Weights/MusicBPE/checkpoint_last_${CHECKPOINT_SUFFIX}.pt \
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" PYTHONWARNINGS="ignore" fairseq-train 	${DATA_BIN_DIR} \
	--seed ${SEED} \
	--user-dir Model \
	--task choral_modeling --criterion multiple_loss \
	--save-dir Weights/MusicBPE/ --restore-file Weights/MusicBPE/checkpoint_last.pt \
	--arch ${NN_ARCH} --sample-break-mode complete_doc  --tokens-per-sample ${MAX_POS_LEN} --sample-overlap-rate ${SOR}\
	--optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
	--lr ${PEAK_LR} --lr-scheduler polynomial_decay --warmup-updates ${WARMUP_UPDATES}  --total-num-update ${TOTAL_UPDATES} \
	--dropout 0.1 --weight-decay 0.01 \
	--batch-size ${MAX_SENTENCES} --update-freq ${UPDATE_FREQ} \
	--max-update ${TOTAL_UPDATES} --log-format tqdm --log-interval 5 \
	--tensorboard-logdir Logs/MusicBPE/${CHECKPOINT_SUFFIX} \
	--ratio ${RATIO} --evt-voc-size ${SIZE_0} --dur-voc-size ${SIZE_1} --trk-voc-size ${SIZE_2} --ins-voc-size ${SIZE_3} \
	--max-rel-pos ${MAX_REL_POS} --max-mea-pos ${MAX_MEA_POS}  --perm-inv ${PI_LEVEL} \
	--max-epoch ${MAX_EPOCHS} \
	2>&1 | tee Logs/MusicBPE/${CHECKPOINT_SUFFIX}_part${RECOVER}.log