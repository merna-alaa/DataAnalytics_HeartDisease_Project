# DataAnalytics_HeartDisease_Project
ML - Data Science -python

Data Cleaning: This involves handling missing values, correcting  inconsistent data, and dealing with outliers to ensure that the data is  accurate and reliable.
Data Integration: If data is collected from multiple sources, it may need to be  integrated into a single dataset to ensure consistency and coherence.
Data Transformation: This step involves converting data into a suitable format  for analysis. This may include standardizing data units, normalizing data  distributions, and encoding categorical variables.
Data Reduction: Data reduction techniques such as feature selection or  dimensionality reduction may be applied to reduce the complexity of the  dataset and improve computational efficiency.
Descriptive Statistics: Descriptive statistics such as mean, median, mode,  and standard deviation can provide a summary of the dataset and help  identify trends and patterns.
Data Visualization: Visualizations such as histograms, box plots, scatter plots,  and heatmaps can provide a graphical representation of the data, allowing  for better understanding of distributions, relationships, and outliers.
Exploratory Data Analysis: EDA techniques such as correlation analysis,  frequency analysis, and data profiling can help uncover patterns,  relationships, and anomalies in the data.
Data Interpretation: Based on the findings from data exploration, insights and  conclusions can be drawn, which can inform further analysis or decision-  making.
Supervised learning models after splitting the dataset and normilzing it such as:
LogisticRegression: This is a linear classification algorithm that models the probability of a binary target variable using a logistic function. It learns a set of weights for the features that maximize the likelihood of the target variable given the input features. Logistic regression is often used for binary classification problems and can be extended to handle multi-class problems.

KNeighborsClassifier: This is a non-parametric classification algorithm that classifies each data point based on the class of its k-nearest neighbors in the feature space. The value of k is a hyperparameter that can be tuned to optimize performance. K-nearest neighbors is a simple and effective algorithm that can handle both binary and multi-class classification problems.

DecisionTreeClassifier: This is a tree-based classification algorithm that recursively partitions the feature space into regions based on the values of the input features. At each level of the tree, the algorithm chooses the feature that best separates the data into the target classes. Decision trees are popular because they are easy to interpret and can handle both binary and multi-class problems.

SVC: This is a kernel-based classification algorithm that learns a decision boundary between the target classes by maximizing the margin between the nearest data points from each class. The kernel function can be chosen to transform the input features into a higher-dimensional space, allowing for non-linear decision boundaries. Support vector machines are versatile and effective, but can be computationally expensive.

GaussianNB: This is a probabilistic classification algorithm that models the conditional probability of the target variable given the input features using Gaussian distributions. It assumes that the features are independent and normally distributed within each class. Gaussian Naive Bayes is simple and efficient, but may not be appropriate for datasets with highly correlated features.

RandomForestClassifier: This is an ensemble classification algorithm that combines multiple decision trees into a single model. Each tree is trained on a random subset of the features and data points, and the final prediction is determined by aggregating the predictions of all the trees. Random forests are robust and effective for both binary and multi-class problems, and can handle high-dimensional feature spaces.



