# Click-Through Rate Prediction (imbalanced dataset)
(Project ongoiong) 

 ## Project Overview
 
Click-through rate is an indicator that shows how likely your advertisement will be noticed and clicked on by visitors of a website. The main aim of the current project is to predict click through rate based on a dataset containing various features related to user interaction with ads. The data has been anonymized, with the websites where the advertisement was shown and the brand it belonged to have been replaced by other values. The project has been broken down into multiple jupyter notebooks for simplycity. 

## Problem Statement
Due to the low likelyhood of a vistor to click an advertisement the dataset utilized here is imbalanced, meaning that one group in the dataset is present with much higher number of records. The goal here is to predict as many users that are likely to click the ad as possible while reducing the number false positives. The main reason for reducing the number of false positives is due to the fact that a cost can be associated with the number of times the algorithm falsely identifies a visitor as someone who is likely to click your ad. An example of this is when you would want to provide a discount code to a targeted set of users, the more visitors are able to see the discount code the lower your bottom line will be.

<p align='center'> 
<img src='./images/bar_imbalance.png'></img>
</p>

## Notebooks
 The project has been separated into multiple jupyter notebook :
 ```
  1-EDA and FE.ipynb	
  2-Modeling_scaling.ipynb	
  3-Modeling_PCA.ipynb	
  4-Modeling_Sampling.ipynb
  5-Modeling_main.ipynb
```
Each notebook will provide more detail regarding the methods used and the resuts.

## Modeling results 

### Using Ensemble of Models
To predict click-through rate a VotingClassifier model was created by combining two separate models, namely , SGDClassifier and ExtraTreesClassifier. As can be observed from the results shown below, by combining these models the false positive rate is reduced to the lowest value.

<img src='./images/ensemble.png'> </img>

