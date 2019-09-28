# Click-Through Rate Prediction (imbalanced dataset)
(Project ongoing) 

 ## Project Overview
<p align='justify'>
Click-through rate is an indicator that shows the  likelihood of an advertisement being clicked when the ad is displayed to the user. The main aim of the current project is to predict click through-rate based on a dataset containing various features related to a user's interaction with advertisements from different brands being shown on various websites. The data has been anonymized, and the values for the websites and the brands have been replaced. The project has been broken down into multiple jupyter notebooks for simplicity, with each jupyter notebook dealing with a specific part of the project. 
</p>
## Problem Statement
<p align='justify'>
Due to the low likelihood of a visitor clicking an advertisement, the dataset utilized here suffers from an imbalance between the population of records belonging to each category (click or no click), meaning that one category in the dataset has far more examples than the other one. The goal here is to predict the users that have a high likelihood of clicking the ad while at the same time minimizing the number of false positives. The main reason for reducing the number of false positives is due to the fact that a cost can be associated with the number of times the algorithm falsely identifies a visitor as someone who is likely to click an ad. An example of this is when you would want to provide a discount code to a targeted set of users, the more visitors are able to see the discount code, the lower your bottom line will be. 
</p>
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
<p align='justify'>
To predict click-through rate, a VotingClassifier model (ensemble) was created by combining two separate models, namely , SGDClassifier and ExtraTreesClassifier. As can be observed from the results shown below, by combining these models the false positive rate is reduced significantly.
 </p>

<img src='./images/ensemble.png'> </img>


