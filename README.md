# deepIQA

This is the reference implementation of [Deep Neural Networks for No-Reference and Full-Reference Image Quality Assessment][arxiv].
The pretrained models contained in the models directory were trained for both NR and FR IQA and for both model variants described in the paper.
They were trained on the full LIVE or TID2013 database respectively, as used in the cross-dataset evaluations. 

> usage: evaluate.py [-h] [--model MODEL] [--top {patchwise,weighted}]
>                   [--gpu GPU]
>                   INPUT [REF]

## TODO: 
* add training code
* add cpu support (minor change)
* remove opencv dependency (minor change)

[arxiv]: https://arxiv.org
