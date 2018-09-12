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
