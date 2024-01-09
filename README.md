# NLP

Hondra Piotr, Jeschke Jan
Warsaw University of Technology

NLP project - SemEval-2024_ECAC subtask 1

## Environment

Before running any script one must install all packages

``` sh
pip install -r requirements.txt
```

## Tokenization

Tokenization of the data is needed for model training.

``` sh
sh scripts/tokenize.sh
```

## Train

Model training is run by following command:

``` sh
sh scripts/train.sh
```

Train params are read from `src/cfgs/config.yaml`.

## Evaluation

Trained model can be evaluated using:

``` sh
sh scripts/eval.sh
```

Paths to labelled data, predicted and output files can be specified inside `scripts/eval.sh` file.

## Trial

Trial data is labelled by running command:

``` sh
sh scripts/trial.sh
```

Trial params are read from `src/cfgs/trial.yaml`.
