# BANG

This repo provides the code for reproducing the experiments in [*BANG*](https://arxiv.org/abs/2012.15525). \
In the paper, we propose a new pre-trained language model called BANG for sequence-to-sequence learning, 
which considers autoregressive, non-autoregressive and semi-autoregressive generation as its pretraining tasks. 


## Pretrained Models:  
[BANG base](https://msraprophetnet.blob.core.windows.net/bang/checkpoint_base_9gram_ck35.pt)  
Pretrained on 16GB English corpus, Wikipedia and BookCorpus.

## Dependency
- pip install torch==1.3.0  
- pip install fairseq==v0.9.0
- pip install tensorboardX==1.7  

## How to use

The procedure includes 1) Tokenize, 2) Binarize, 3) Finetune, 4) Inference.  
BANG is implemented on base of Fairseq, which you can refer to [Fairseq Mannual](https://fairseq.readthedocs.io/en/latest/command_line_tools.html).  



Tokenize. Prepare your train.src, train.tgt, and valid, test sets. Input and output of one sample are placed in the .src and .tgt file with one line.    
Use bert-uncased tokenizer to tokenize your data into word piece. 
```
from transformers import BertTokenizer


def bert_uncased_tokenize(fin, fout):
    fin = open(fin, 'r', encoding='utf-8')
    fout = open(fout, 'w', encoding='utf-8')
    tok = BertTokenizer.from_pretrained('bert-base-uncased')
    for line in fin:
        word_pieces = tok.tokenize(line.strip())
        new_line = " ".join(word_pieces)
        fout.write('{}\n'.format(new_line))
bert_uncased_tokenize('train.src', 'tokenized_train.src')
bert_uncased_tokenize('train.tgt', 'tokenized_train.tgt')
bert_uncased_tokenize('valid.src', 'tokenized_valid.src')
bert_uncased_tokenize('valid.tgt', 'tokenized_valid.tgt')
bert_uncased_tokenize('test.src', 'tokenized_test.src')
bert_uncased_tokenize('test.tgt', 'tokenized_test.tgt')
```
Binirize it with fairseq-preprocess
```
fairseq-preprocess \
--user-dir ./bang/bang \
--task translation_prophetnet \
--source-lang src --target-lang tgt \
--trainpref tokenized_train --validpref tokenized_valid --testpref tokenized_test \
--destdir processed_data --srcdict ./bang/vocab.txt --tgtdict ./bang/vocab.txt \
--workers 20
```
Fine tune with fairseq-train.  
### Autoregressive Generation 
You can directly use [ProphetNet](https://github.com/microsoft/ProphetNet) for AR finetuning. 
Or you can also use this repo. They are equivalent. Set these parameters:    
--disable-ngram-loss：please set True for AR finetuning    
--ngram: please set 1 for AR finetuning  
--nar-ratio: please set 0.0 for AR finetuning  
--fp16: if your GPU device supports, set True to accelerate training  
```
DATA_DIR=processed_data
ARCH=prophet_ar_nar_mixed_base
CRITERION=ngram_language_loss_NAR_mixed
SAVE_DIR=models/model_ar
TENSORBOARD_LOGDIR=models/logs_ar
PRETRAINED_MODEL=checkpoint_base_9gram_ck35.pt
NAR_RATIO=0.0

fairseq-train $DATA_DIR \
--user-dir ./bang/bang  \
--task translation_prophetnet --arch $ARCH \
--optimizer adam --adam-betas '(0.9, 0.999)' --clip-norm 0.1 \
--lr 0.0001 --min-lr 1e-09 --nar-ratio ${NAR_RATIO} --ngram 1 --disable-ngram-loss \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 1000 \
--dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
--criterion $CRITERION --label-smoothing 0.1 \
--update-freq 1  --max-tokens 3072 \
--num-workers 8  \
--load-from-pretrained-model $PRETRAINED_MODEL \
--ddp-backend=no_c10d --max-epoch 10 \
--max-source-positions 512 --max-target-positions 512 \
--truncate-source \
--save-dir $SAVE_DIR \
--keep-last-epochs 10  --save-interval 1 \
--tensorboard-logdir $TENSORBOARD_LOGDIR \
```
Inference with fairseq-generate to generate targets for given processed test files. Or you can [fairseq-interactive](https://fairseq.readthedocs.io/en/latest/command_line_tools.html#fairseq-interactive) to generate answers for your typed-in text (which should also been tokenized).
```
BEAM=4
LENPEN=1.2
CHECK_POINT=models/model_ar/checkpoint8.pt
SUFFIX=_ar_pelt${LENPEN}_test_beam${BEAM}
OUTPUT_FILE=outputs/output$SUFFIX.txt

PYTHONIOENCODING=utf-8 fairseq-generate ./processed_data --path $CHECK_POINT --user-dir ./bang/bang --task translation_prophetnet --batch-size 36 --gen-subset train --beam $BEAM --num-workers 4 --lenpen $LENPEN 2>&1 > $OUTPUT_FILE
grep ^H $OUTPUT_FILE | cut -c 3- | sort -n | cut -f3- | sed "s/ ##//g" > outputs/sort_hypo$SUFFIX.txt
grep ^H $OUTPUT_FILE | cut -c 3- | sort -n | cut -f3-  > outputs/sort_hypo$SUFFIX.txt.tokenized
```
### Non-autoregressive Generation  
--nar-ratio: please set 1.0 for NAR finetuning  
--fp16: if your GPU device supports, set True to accelerate training  
```
DATA_DIR=processed_data
ARCH=prophet_ar_nar_mixed_base
CRITERION=ngram_language_loss_NAR_mixed
SAVE_DIR=models/model_nar
TENSORBOARD_LOGDIR=models/logs_nar
PRETRAINED_MODEL=checkpoint_base_9gram_ck35.pt
NAR_RATIO=1.0

fairseq-train $DATA_DIR \
--user-dir ./bang/bang  \
--task translation_prophetnet --arch $ARCH \
--optimizer adam --adam-betas '(0.9, 0.999)' --clip-norm 0.1 \
--lr 0.0001 --min-lr 1e-09 --nar-ratio $NAR_RATIO --ngram 1 --disable-ngram-loss \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 1000 \
--dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
--criterion $CRITERION --label-smoothing 0.1 \
--update-freq 1  --max-tokens 3072 \
--num-workers 8  \
--load-from-pretrained-model $PRETRAINED_MODEL \
--ddp-backend=no_c10d --max-epoch 50 \
--max-source-positions 512 --max-target-positions 512 \
--truncate-source \
--save-dir $SAVE_DIR \
--keep-last-epochs 10  --save-interval 5 \
--tensorboard-logdir $TENSORBOARD_LOGDIR \
```
Inference with fairseq-generate to generate targets for given processed test files. Or you can [fairseq-interactive](https://fairseq.readthedocs.io/en/latest/command_line_tools.html#fairseq-interactive) to generate answers for your typed-in text (which should also been tokenized).
```
SUFFIX=_nar
CHECK_POINT=models/model_nar/checkpoint40.pt
OUTPUT_FILE=outputs/output${SUFFIX}.txt

PYTHONIOENCODING=utf8 fairseq-generate processed_data  --user-dir ./bang/bang --path ${CHECK_POINT} --truncate-source --max-source-positions 512 --task translation_prophetnet_nar --batch-size 36 --beam 1 --gen-subset test  2>&1 > ${OUTPUT_FILE}

grep ^H $OUTPUT_FILE | cut -c 3- | sort -n | cut -f3- > outputs/sort_hypo${SUFFIX}.txt
python post_processed_nar.py outputs_v1/sort_hypo${SUFFIX}.txt outputs/sort_hypo${SUFFIX}.txt.dedup

```

## TIPS:
1, Autoregressive needs fewer finetuning steps, while Non-autoregressive needs longtime finetuning to get good performance.  
2, For AR finetuning, you can directly use the code in [ProphetNet](https://github.com/microsoft/ProphetNet).  
3, **We highly recommend you use sequence distillation before NAR finetuning.**  
4, If you met problems to run fairseq-preprocess, fairseq-train and other commands, or if you want to modify the workflow/inference pipeline, 
it's a good choice to download fairseq git repo, checkout v0.9.0, and merge our codes. Then, modify their preprocess.py, train.py or generate.py, to run your new pipeline. 


## Repo Reference
This repo is referred to [Fairseq-v0.9.0](https://github.com/pytorch/fairseq/tree/v0.9.0) and [ProphetNet](https://github.com/microsoft/ProphetNet).



## How to Cite
If you extend or use this work, please cite the [paper](https://arxiv.org/abs/2012.15525) where it was introduced:
```
@inproceedings{qi2021bang,
  title={Bang: Bridging autoregressive and non-autoregressive generation with large scale pretraining},
  author={Qi, Weizhen and Gong, Yeyun and Jiao, Jian and Yan, Yu and Chen, Weizhu and Liu, Dayiheng and Tang, Kewen and Li, Houqiang and Chen, Jiusheng and Zhang, Ruofei and others},
  booktitle={International Conference on Machine Learning},
  pages={8630--8639},
  year={2021},
  organization={PMLR}
}
```
[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)
