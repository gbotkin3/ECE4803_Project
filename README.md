# ECE8803_Final_Project
DRSS Severity Classification on OCT images

# Abstract

By looking at the visulization of a set of data, is it possible to predict the performance of models relative to each other for the purpose of enabling reliably choosing the best performing model?

## Initial Setup

Run the following commands to clone the repository

```
git clone https://github.com/gbotkin3/ECE8803_Final_Project.git
```

Add the PRIME_FULL folder from an unzipped OLIVES.zip and PRIME.zip to ./data_files

## Required Dependencies

The following packages are required to run the python file.

```
Numpy
PyTorch
Panda
PIL
SkLearn
Seaborn
Matplotlib
```

## Running

The python code can be ran from the home directory though the use of the Makefile with the command ```make```.

Results are shown in console and stored in ./results and ./results/figures

In toplevel.py, various settings can be changed to enable or disable models and visualization as well as setting batch / sample size.

## Visualization Methods (Stored in ./results/figures)

  1. Scatter Map
  2. KDE Plot
  2. Pair Plot

## Models 

  1. K-Nearest Neighbors
  2. Decision Tree
  3. Guassian Naive Bayes 
  4. Convolutional Neural Networks

## Performance Metrics (Stored in ./results)

  1. Accuracy
  2. Precision
  3. Recall
  4. F1

