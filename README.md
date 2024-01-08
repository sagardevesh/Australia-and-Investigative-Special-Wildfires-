# Australia-and-Investigative-Special-Wildfires-
This project uses Random Forest Algorithm to predict the target feature FRP (Fire Radiative Power) in the Australian continent based on input features. 

Here is the link to the [Austraila Wildfire](https://www.kaggle.com/datasets/brsdincer/australia-and-investigative-special-wildfires-data) Dataset on Kaggle. 


Initially, we formulated Data Quality Reports for continuous and categorical features. 

Data Quality Report for Continuous features:-

![image](https://github.com/sagardevesh/Australia-and-Investigative-Special-Wildfires-/assets/25725480/b23a68e7-7b3b-4c45-8ae4-972d10152878)

Data Quality Report for Categorical features:-

![image](https://github.com/sagardevesh/Australia-and-Investigative-Special-Wildfires-/assets/25725480/6366771b-0347-4e92-a228-54513d939105)

Also, formed a correlational heat map showing the correlation between all the features.

![image](https://github.com/sagardevesh/Australia-and-Investigative-Special-Wildfires-/assets/25725480/0581af62-d205-4cb9-b655-0376230a30ab)

A series of steps is performed in order to preprocess the raw data available on Kaggle. 

### Confidence:-

As per the correlational heat map above, 'confidence' of an incident is highly correlated to 'brightness'. Hence, 'brightness' is the most useful attribute in predicting the confidence of an incident. Furthermore, 'confidence' also appears to be correlated to 'frp' and 'bright_t31' features.

As per the dataset description on Kaggle, confidence values are intended to help the users gauge the quality of individual fire pixels. High confidence values indicate higher saturation of the fire pixels and high relative temperature anomaly. Lower confidence values typically indicate areas of sun glint (less brightness) and lower relative temperature anomaly. Hence, lower confidence value must typically indicate lower brightness and vice versa. Therefore, brightness is the most import attribute in predicting confidence.

Moreover, since 'brightness' attribute is also highly correlated to attributes 'bright_t31' and 'frp', resultingly 'bright_t31' and 'frp' are also important attributes in predicting the confidence of an incident. (Analogous to transitive property: if a=b and b=c, then a=c)

### Geographical Heatmap of FRP for Aqua area:-

Next, we plotted a geographical heatmap of FRP with the help of geopandas. Below is how the resultant map looks like:-
![image](https://github.com/sagardevesh/Australia-and-Investigative-Special-Wildfires-/assets/25725480/98550c04-5eb6-4984-a503-1862fa7b8673)

Above is a heatmap that we have plotted using the plot function. Higher values of frp are depicted by the darker shade, and lower values by lighter shade.

However, we also plotted a geographic heatmap using folium for the same 'geo_df_plot' dataframe , as shown below
![image](https://github.com/sagardevesh/Australia-and-Investigative-Special-Wildfires-/assets/25725480/74fe2c88-439e-4c6b-bee3-6ea80ddb7557)

The map above is an interactive map.

### Model Selection:-

This is a supervised learning problem because we know our target variable. We want to perform a predictive analysis on our target variable 'frp' using other attributes within the dataset. We used feature selection to shortlist most important features contributing to the predictive analysis of our target variable. For feature selection, We used Pearson Correlation method. 

### Random Forest Regression:-

This model has been selected as the baseline model. It takes a multiple classifying decision tree each trained on a subset dataset and uses averaging to improve the predictive accuracy and control over-fitting. The use of multiple trees reduces variance. It uses various parameters, n_estimators - number of decision trees max_depth - the number of levels in a decision trees random_state - controls bootstrapping of the samples used when building trees.

### Evaluation metrics:-

The evaluation is done using two evaluation metrics i.e Mean Absolute error and Accuracy %. Mean absolute error is taken into consideration as the error might be high or low depending on the range of values. Due to this reason, I take mean values of all the errors to generalise the error rate. Accuracy % is another metric which is computed based on mean absolute percentage error. These metrics tell us the performance of our model.

### Hyperparameter tuning:-

Once the model is trained and the results are out in terms of its performance, we performed hyperparamter tuning to find the best possible combination of parameters that gives us the best model performance. For this purpose, we use K-Cross Validation Random Search Grid.

Using Scikit-Learn’s RandomizedSearchCV method, we can define a grid of hyperparameter ranges, and randomly sample from the grid, performing K-Fold CV with each combination of values. This K-Fold Cross Validation method, splits our training data into K number of subsets called folds and evaluates the model on K-1 subsets, thus ensuring the model doesn't generalise any predictions and avoids overfitting. The most important factor here is 'n_iter' which defines the number of combinations RandomSearchCV fucntion will perform and 'cv' represents the number of folds used for cross validation.

The same set of above steps are done for another machine learning model Linear Regression. After hyperparameter tuning of the Linear regression model, we performed Statistical Significance Testing to evalute both the models and draw a comparison between them by evaluating accuracy scores.

******************************************************************************************************************

## Steps to run on local machine/jupyter notebook:
To run this assignment on your local machine, you need the following software installed.

*******************************************************
Anaconda Environment

Python
*********************************************************

To install Python, download it from the following link and install it on your local machine:

https://www.python.org/downloads/

To install Anaconda, download it from the following link and install it on your local machine:

https://www.anaconda.com/products/distribution

After installing the required software, clone the repo, and run the following command for opening the code in jupyter notebook.

jupyter notebook A1-Sagar_Devesh-Sarthak_Pandit.ipynb
This command will open up a browser window with jupyter notebook loading the project code.

You need to upload the dataset from the git repo to your jupyter notebook environment.






