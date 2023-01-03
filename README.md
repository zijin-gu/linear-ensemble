# linear-ensemble

Code for paper [Personalized visual encoding model construction with small data](https://www.nature.com/articles/s42003-022-04347-z).

## Introduction
In this work, we compare the prediction accuracy (the ability to accurately predict brain responses)and consistency (the ability to preserve inter-individual variability) between several models with small training data. They are 

1. `Individual-20K model`: model with a standard architecture (ResNet50 backbone) trained on densely-sampled individual data, e.g. 20,000 samples
2. `Scratch model`: model that has the same architecture as individual-20K but trained on small data, e.g. 300 samples
3. `Finetuned model`: model that has the same architecture as individual-20K but initialized with the group-level individual-20K weights and finetuned on small data
4. `Linear ensemble model`: model that linearly ensembles predictions from individual-20K models 
5. `Average ensemble model`: model that simply averages predictions from individual-20K models 

## Usage
1. Prepare your data and wrap them in dataloaders.
2. `individual_model.py`: train individual-20K model and scratch model.
3. `individual_finetune.py`: train finetuned model.
4. `ensemble.py`: train linear ensemble model.

## References
Gu, Z., Jamison, K., Sabuncu, M. et al. Personalized visual encoding model construction with small data. Commun Biol 5, 1382 (2022). https://doi.org/10.1038/s42003-022-04347-z
