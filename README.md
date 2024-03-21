# STAT8021Project

This project aims to implment [Feedback Prize - Evaluating Student Writing](https://www.kaggle.com/competitions/feedback-prize-2021/overview) and try to utilize LLM and prompt tuning to optimize this task.

Follow this [link](https://www.kaggle.com/code/tengwang0318/notebook6bba6ffdd3/edit/run/168137335) to check the "performance" in kaggle.

### baseline: Roberta-base

train the model:

```
python train.py --model_name roberta-base
```

test the model:

```
python test.py --model_name roberta-base
```

### baseline: Longformer

```
python train.py --model_name longformer
```

```
python test.py --model_name longformer
```

