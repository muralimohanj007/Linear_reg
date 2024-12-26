1.Divide the Dataset into Independent and Dependent Dataset:
  Separate the features (independent variables) and the target (dependent variable) for analysis.

2.Divide the Dataset into Train and Test:
  Split the data into training and testing sets (e.g., 80% for training and 20% for testing) to evaluate model performance.

3.Feature Scaling – Standardization:
  Normalize or standardize features to ensure they are on the same scale. This is particularly important for models like Ridge, Lasso, or Elastic Net that are sensitive to the scale of data.

4.Model Training:
  Train the regression model on the training dataset using a specific algorithm (e.g., Linear Regression, Ridge, Lasso).
5.Model Fit:
  Fit the model to the training data, enabling it to learn the relationships between the independent and dependent variables.

6.Coefficients and Intercept:
  Retrieve the coefficients (weights) and the intercept (bias) from the trained regression model, which indicate the relationship strength between features and the target.

7.Prediction:
  Use the trained model to predict outcomes on the test dataset.

8.Evaluation – MSE, MAE, RMSE:
  Calculate evaluation metrics like Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) to assess how well the model performs on the test dataset.
