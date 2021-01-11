# Sales Predictions
Python project : predict future sales of a company per article and per week

# Objective

The objective of the project is to develop a tool for forecasting demand from Johnson Electric customers. This tool will allow Johnson Electric to have a better understanding of their customers and to have better management of their inventory. The data available are actual sales, products and customer locations. This is the delivery information of several products at different times (per week, per months...).

# Method

The data set is linear, it is a series of quantitative variables. The goal is to predict the continuation of these. Linear regression is linear modeling allows to estimate future from information from the past.

# Project

![alt text](https://github.com/Ainara2828/Sales-Prediction/blob/main/images/menu.PNG?raw=true)

<img src="https://github.com/Ainara2828/Sales-Prediction/tree/main/images/menu.PNG" alt="drawing" width="400"/>

In this menu you can select an article, a week, and download the prediction for one or all articles, one or all weeks, or for one article and one week.

# Prediction per article


<img src="https://github.com/Ainara2828/Sales-Prediction/tree/main/images/article.PNG" alt="drawing" width="400"/>

After selecting one article, the tool gives you all the predictions of the article selected for the following weeks. The interface also displays the item's p-value (its precision threshold) and its prediction error.

Explanation of the graphic :

- On the x-axis, there is the Y variable to predict
- The points represent the coordinates (X, Y) therefore actual delivery and its corresponding variance
- The line represents the prediction made from the X test, it represents the set of Y values found by the algorithm

Here are 3 explicative variables created :

- the explanatory variable is the average of the values of the pivot
- the explanatory variable is the variance of the values of the pivot
- the explanatory variable is the standard deviation of the values of the pivot

The best explicative variable to predict the sales is with the variance (smallest error) shown above on the top of the image.

The user can also download the predictions for one article into an excel file by clicking on "dowload".

# Prediction per week

The user can also choose the predictions of all articles based on a specific week. To do this, he must select the number of the week from the drop-down list. The user can then save the results in an excel file named “predictionPerWeek” by clicking on the “Enter Week” button as below.


<img src="https://github.com/Ainara2828/Sales-Prediction/tree/main/images/week.PNG" alt="drawing" width="400"/>

# Prediction per article and week

If the user selects an article and a week from the respective menus, then the user can display the prediction for the selected article and week on the screen by clicking on the “Enter Week and article” button as below.



<img src="https://github.com/Ainara2828/Sales-Prediction/tree/main/images/weekAndArticle.PNG" alt="drawing" width="400"/>

# All predictions

The user also has the "Download data" button which saves all of the predictions in an excel file called "predictions" thus guaranteeing the user full access and use of the forecasts.


<img src="https://github.com/Ainara2828/Sales-Prediction/tree/main/images/download.PNG" alt="drawing" width="400"/>


