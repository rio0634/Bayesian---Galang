#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Naive Bayes Simulation

Author: Rio Angel Galang
Date: April 17, 2026

This program demonstrates the Naive Bayes algorithm using a simple binary classification dataset.
It shows how a probabilistic model learns from training data and predicts outcomes based on feature likelihoods.
"""

from sklearn.naive_bayes import GaussianNB
import numpy as np

print("Naive Bayes Simulation")
print("\nThis program classifies whether an individual is FIT or NOT FIT")
print("based on Physical and Mental conditions using Naive Bayes.\n")

print("Feature Meaning:")
print("Physical: 1 = Fit, 0 = Not Fit")
print("Mental: 1 = Stable, 0 = Unstable\n")

# Training dataset (binary features)
X = np.array([
    [1, 1],
    [1, 0],
    [0, 1],
    [1, 1],
    [0, 0]
])

# Labels: 1 = Fit, 0 = Not Fit
y = np.array([1, 1, 0, 1, 0])

# Train Naive Bayes model
model = GaussianNB()
model.fit(X, y)

# Test input
test_data = [[1, 1]]

# Prediction
prediction = model.predict(test_data)
probabilities = model.predict_proba(test_data)

print("Input:", test_data)

print("\nClass Probabilities:")
print("P(Not Fit):", round(probabilities[0][0], 4))
print("P(Fit):", round(probabilities[0][1], 4))

print("\nFinal Prediction:", prediction[0])

# Interpretation
print("\nInterpretation:")

if prediction[0] == 1:
    print("The model predicts that the individual is FIT based on physical and mental conditions.")
else:
    print("The model predicts that the individual is NOT FIT based on physical and mental conditions.")


# In[ ]:




