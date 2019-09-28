# Click-Through Rate Prediction (imbalanced dataset)
(Project ongoiong) 

 ## Project Overview
 
Click-through rate is an indicator that shows the  liklihood of an advertisement being clicked when the ad is displayed to the user. The main aim of the current project is to predict click through-rate based on a dataset containing various features related to a user's interaction with advertisements from different brands being shown on various websites. The data has been anonymized, and the values for the websites and the brands have been replaced. The project has been broken down into multiple jupyter notebooks for simplycity, with each jupyterr notebook dealing with a specific part of the project. 

## Problem Statement
Due to the low likelyhood of a vistor to click an advertisement the dataset utilized here is imbalanced, meaning that one group in the dataset is present with much higher number of records. The goal here is to predict as many users that are likely to click the ad as possible while reducing the number false positives. The main reason for reducing the number of false positives is due to the fact that a cost can be associated with the number of times the algorithm falsely identifies a visitor as someone who is likely to click your ad. An example of this is when you would want to provide a discount code to a targeted set of users, the more visitors are able to see the discount code the lower your bottom line will be.

<p align='center'> 
<img src='./images/bar_imbalance.png'></img>
</p>

## Notebooks
 The project has been separated into multiple jupyter notebook :
 ```
  1-EDA and FE
  2-Modeling_scaling
  3-Modeling_PCA
  4-Modeling_Sampling
  5-Modeling_main
```
Each notebook will provide more detail regarding the methods used and the resuts.

## Modeling results 

### Using Ensemble of Models
To predict click-through rate, a VotingClassifier model (ensemble) was created by combining two separate models, namely , SGDClassifier and ExtraTreesClassifier. As can be observed from the results shown below, by combining these models the false positive rate is reduced significantly.

<img src='./images/ensemble.png'> </img>

