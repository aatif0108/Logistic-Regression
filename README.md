## Logistic Regression
This project implements logistic regression from scratch using Python. The model leverages **Batch Gradient Descent** to optimize the parameters (weights) by minimizing the **Logistic Loss (Cross-Entropy) cost function**. The logistic regression model predicts binary outcomes (0 or 1) by applying the sigmoid function to the linear combination of input features. 

During training, the model iteratively adjusts the weights to reduce the cost, which measures the difference between the predicted probabilities and the actual labels. The optimization process is carried out through gradient descent, where the gradients of the cost function with respect to the model parameters are computed and used to update the weights.

Once the model is trained, it is evaluated using various performance metrics such as **accuracy**, **precision**, **recall**, and **F1-score**. A **confusion matrix** is also generated to give a detailed breakdown of the model's predictions. The goal is to minimize errors in prediction, especially false positives and false negatives, while ensuring that the model generalizes well to new data.
