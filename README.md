# Course Project of Group 1 - Cancer classification using machine learning
This is the course project for 02-620: Machine Learning For Scientist in Spring 2022. And oour project is about cancer classification using machine learning.

## Group members
- Yuxuan Wu
- Eric Li
- Yifan Wu
- Xin Wang

## Dependencies (only for model implementation section)

- Python3
- scikit-learn
- pandas
- numpy
- pytorch

All jupyter notebooks are self-contained and runnable!

## Original Datasets (too large to include)

[BRCA](https://xenabrowser.net/datapages/?dataset=TCGA-BRCA.htseq_counts.tsv&host=https%3A%2F%2Fgdc.xenahubs.net&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443)

[LUAD](https://xenabrowser.net/datapages/?dataset=TCGA-LUAD.htseq_counts.tsv&host=https%3A%2F%2Fgdc.xenahubs.net&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443)

## Data preprocessing and DGE
We used R to perform Differential Gene Expression (DGE), which is one of our feature selection methods. And the other is `selectKBest` function in Python.

R scripts are under `./Preprocessing_and_DGE`

## Models

### Self Implemented Machine Learning Algorithm (Decision Tree and AdaBoost)

    self-implemented decision tree model: ./Model/Decision_tree/decision_tree.ipynb
    self-implemented adaboost model: ./Model/AdaBoost/adaboost.ipynb
    
### MLP with Pytorch

    ./Model/MLP

## Follow-Up Evaluation and Results

### Model evaluation scripts

    ./Performance

### Model evaluation results

    ./Results/Performance

### Feature selection scripts

    ./Feature_importance

### Feature selection results

    ./Results/Feature_importance


## Additional plots that didn't contain in the report because of page limit

    ./Additional_plots
