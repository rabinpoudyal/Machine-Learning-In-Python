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

#### Logistic Regression (Linear Classifier)
- Like regression but we model the output as probabilities b/w 0 and 1 instead of continuous values
- Converting our linear regression model to logistic regression model takes following steps:
- y = b0 + b1x
- If we apply sigmoid function to above y in the eqn => p = (1/(1+e^-y))
- And we solve for y then
- ln(p/(1-p)) = b0 + b1x
- Which is the logistic regression. 
- By applying sigmoid, the function will not go higher than 1 and below 0

#### K-nearest Neighbours Classification (Non linear Classifier)
- Take k = n as a number of nearest neighbours for point k which we want to classify
- Count the number of points that fall on each category of neighbours
- The point k will fall into the category that has maximum number of neighbours
- How to choose the best n_clusters?
- WCSS is the metric that is used to choose best cluster. It is based on euclidean distance and we try to increase the inter-cluster distance and minimize the intra-cluster distance. When we plot this in graph, it will yield the elbow. And we choose that elbow as the optimum number of cluster

#### Hierarchial Clustering
- This method is used when we are not sure how many clusters we want in our dataset. 
- We construct the dendogram from the given dataset.
- We count the number of vertical lines that do not intersect any horizontal corresponding line.

#### Support Vector Machine
- We can draw many lines between the two classes.
- Find the best optimal line that separates two different classes.
- It is a  linear classifier
- Finding the best decision boundary is based on support vectors(those extreme datas of each classes)
- The line is drawn equidistant from those support vectors and the best decision boundary is based on maximum distance between the support distance and the hyperplane
- Support vectors in apple and orange classifier => The worst apple that looks like orange and worst orange that looks like an apple
- The decision boundary is called maximum length hyperplane/classifier

## Section Four: Natural Language Processing
- Get the raw text data from various sources.
- Perform text cleaning - tokenization, lemminization, stemming, vectorizing.
- Feed the data into the proper algorithm
- Evalute the alternatives
- Select the best model

## Section Five: XGBoost
- 3 Advantages:- High performance when dataset is huge, fast execution and we can keep all the interpretation of problem( i.e we dont need to apply feature scaling)
- No feature scaling because this model is based on decision trees
