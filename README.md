## Machine Learning In Python

### Section One:
- Importing data
- Taking care of missing values
- Encoding categorical variables
- Applying feature scaling
- Splitting into train and test section

### Section Two: 
Linear Regression Assumptions
- Linearity
- Homoscedasticity
- Multivariate normality
- Independence of errors
- Lack of multicollinearity
#### Simple Linear Regression:
- y = b0 + b1*x
- Here, b0 is point where line crosses X-axis, b1 is the proportion on which y changes wrt x
- x is independent variable and y is dependent variable
Steps in Simple Linear Regression:
- Load data
- Prepare independent variable matrix and dependent variable vector
- Split dataset into test and train
- Train the model
- Predict the output
- Visualize the prediction 

#### Multiple Linear Regression:
- x = b0 + b1*x1 + b2*x2 ...
- Perform linear regression like above but how do you know your model is most optimized model?
- Implement backward elimination algorithm to find the most significant predictors of outcome 
- First fit the regression with all independent variables
- Set the significance level eg 5%
- Compute p-value of the variables if it is greater than 0.05 remove variable and fit again
- Continue the process till we have most significant variables left

#### Ploynomial Linear Regression:
- x = b0 + b1*x1^2 + b2*x2^3 + b3*x3^4
- It is still linear because we are talking about coefficients b0,b1,b2 not variable itself
- Sometimes it fits best among other regression like diseases spread or others
- Fits the dataset perfectly as we increase the degree of polynomial
- First transform into polynomial features and then perform multiple linear regression

#### Support Vector Regression
- Based on support vector machine
- Needs to feature scale before applying regression
- kernel = elf (same as polynomial regression)
- Finds outliers as well 

#### Decision Tree Regression (Non Linear and non-continuous regression model)
- CART (Classification Tree and Regression Tree)
- Break down datasets into different leaves and find mean of the leaves to predict value that lies in that leaf
- Very powerful model for multi dimensional models
- Must be careful while plotting a graph because we are visualizing the non continuous model

#### Random Forest Regression
- Ensamble Learning Method
- Pick n number of points randomly from dataset
- Build a decision tree
- Build N decision trees
- We will have N number of predictions from those trees
- Calculate the average of those predictions and it will be our final prediction
- Ensmble learning is powerful because it does not get affected by change in dataset
- Increasing tree will first create more steps but slowly converge and starts choosing the best point of the stairs instead of creating more staris

## Section three: Classification
- When we have discrete output and not continuous like cancer patient or not?

#### Logistic Regression
- Like regression but we model the output as probabilities b/w 0 and 1 instead of continuous values
- Converting our linear regression model to logistic regression model takes following steps:
- y = b0 + b1x
- If we apply sigmoid function to above y in the eqn => p = (1/(1+e^-y))
- And we solve for y then
- ln(p/(1-p)) = b0 + b1x
- Which is the logistic regression. 
- By applying sigmoid, the function will not go higher than 1 and below 0

