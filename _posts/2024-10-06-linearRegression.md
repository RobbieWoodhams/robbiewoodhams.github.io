---
title: Exploring Linear Regression
date: 2024-10-05 18:00:00 +0000
categories: [Machine Learning]
tags: [Machine Learning, Linear Regression, Gradient Descent, ]
math: true
---

# Chapter 1: Linear Regression

## What is Linear Regression

Linear regression is a statistical method to model the relationship between a dependent variable (y) and one or more independent variables (x). Its purpose is to predict the value of y based on x, assuming a linear relationship between the two.

In simple terms, we try to fit a straight line to the data points, where the line represents the best possible prediction for y given any value of x. The relationship is captured by the formula:

$$y = mx + c$$

Linear regression is a key technique in machine learning and data analysis, used in applications ranging from predicting sales to understanding economic trends.

## Formula Breakdown

### Equation of a Line: $$y = mx + c$$
- m (slope): Represents the rate of change in y for a unit increase in x.
- c (intercept): The value of y when x equals zero.

### Input and Output Variables:
- Input variable (x): The independent variable, such as house size, experience, etc.
- Output variable (y): The dependent variable, such as house price, salary, etc.

### Prediction:
The goal of linear regression is to predict ŷ (the predicted value of y) based on the input x. The model learns m and c to make these predictions.

## Example: Predicting Exam Scores Based on Hours Studied

| Hours Studied (x) | Exam Score (y) |
|----------|----------|
| 1   | 52   |
| 2   | 56   |
| 3   | 60   |
| 4   | 63   |
| 5   | 67   |


So how do we try to fit a straight line through the data points, where the line represents the best possible prediction for y given any value of x? Given we do not yet know the optimal values of m and c we first start by initialising them with random values. For simplicity we can use m = 0 and c = 0.

This means our initial line equation is:

$$y = 0x + 0  ⟹ ŷ = 0$$

>**Note:**
> ŷ (y hat) is the predicted value of y 
{: .prompt-info }

Using the data set and our initial line equation we can figure out our initial predicted values for each point.

| Hours Studied (x) | Exam Score (y) | Predicted Score 
|----------|----------|----------|
| 1   | 52   | 0   |
| 2   | 56   | 0   |
| 3   | 60   | 0   |
| 4   | 63   | 0   |
| 5   | 67   | 0   |

Clearly, our predictions are wrong since we initialised m and c to 0. Let’s move on to how we would calculate the error of our predicted scores.

# Chapter 2: Cost Function

## What is the cost function?

The Cost Function (also called the error or loss function) measures how well a machine learning model's predictions match the actual data. In the case of linear regression, the cost function quantifies the difference between the predicted values (ŷ) and the actual values (y).

The objective of linear regression is to minimize the cost function, which means finding the line that fits the data points as closely as possible. This is done by adjusting the parameters m (slope) and c (intercept) to reduce the cost.

The most commonly used cost function in linear regression is the Mean Squared Error (MSE), which calculates the average of the squared differences between the actual and predicted values.

## Formula Breakdown

The formula for the Mean Squared Error (MSE) is:

$$ 
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 
$$

### Key Terms:
- $$n$$: The number of data points in your dataset.
- $$y_i$$: The actual value of the dependent variable for the i-th data point.
- $$\hat{y}_i$$: The predicted value for the i-th data point.
- $$(y_i - \hat{y}_i)^2$$: The squared difference between the actual and predicted values. This represents the error or residual for each data point.
- $$\sum_{i=1}^{n}$$: Summation notation (or sigma notation) allows us to write a long sum in a single expression.

### Why Squared Differences?
- Squaring the differences ensures that both positive and negative errors contribute equally to the total error. Without squaring, negative differences could cancel out positive ones, giving a misleading total error.
- Squaring also amplifies larger errors, so models that make significantly inaccurate predictions are penalized more.

### Interpretation:
- A smaller MSE means that the predictions are closer to the actual values, indicating a better fit.
- A larger MSE indicates larger discrepancies between the predicted and actual values, meaning the model needs improvement.

## Example: Predicting Exam Scores Based on Hours Studied

Following our example of where we predicted the exam scores based on hours studied: 

| Hours Studied (x) | Exam Score (y) | Predicted Score 
|----------|----------|----------|
| 1   | 52   | 0   |
| 2   | 56   | 0   |
| 3   | 60   | 0   |
| 4   | 63   | 0   |
| 5   | 67   | 0   |

Let’s calculate the error for each data point:

- For $$x_1 = 1$$: $$(y_1 - \hat{y}_1)^2 = (52 - 0)^2 = 2704$$
- For $$x_2 = 2$$: $$(y_2 - \hat{y}_2)^2 = (56 - 0)^2 = 3136$$
- For $$x_3 = 3$$: $$(y_3 - \hat{y}_3)^2 = (60 - 0)^2 = 3600$$
- For $$x_4 = 4$$: $$(y_4 - \hat{y}_4)^2 = (63 - 0)^2 = 3969$$
- For $$x_5 = 5$$: $$(y_5 - \hat{y}_5)^2 = (67 - 0)^2 = 4489$$

Now, we sum the squared errors:

$$2704 + 3136 + 3600 + 3969 + 4489 = 17898$$

The MSE is the average squared error:

$$MSE = \frac{17898}{5} = 3579.6$$

This is our initial error. Now, we need to minimise this error by updating m and c using Gradient Descent.

# Chapter 3: Gradient Descent

## What is Gradient Descent?
Gradient Descent is an iterative algorithm used to minimise the cost function by adjusting m and c. At each step, we calculate how much to adjust m and c based on the gradients (derivatives) of the cost function with respect to these parameters.

## Step-by-Step Process

To minimise the MSE, we need to calculate how the MSE changes with respect to m and c. This is done using partial derivatives.

### Step 1: Deriving the Gradients

Start with the MSE formula:

$$ MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - (mx_i + c))^2 $$

Lets start with differeniating the MSE with respect to m:

As the MSE has a function within a function we will use the chain rule to differentiate. 

The Chain Rule:

$$ \frac{\partial}{\partial x} [f(g(x))] = f'(g(x)) \cdot g'(x) $$

Putting the MSE formula into the chain rule we get:

$$ \frac{\partial MSE}{\partial m} [(y_i - (mx_i + c))^2] $$

This can be broken down into an inner function and an outer function which are:

- Inner Function g(x): $$y_i - (mx_i + c)$$
- Outer Function f(x): $$u^2$$

>**Note:**
> here u is just a placeholder for $$y_i - (mx_i + c)$$
{: .prompt-info }

Firstly lets find the derivative of the outer function. Using the power rule the derivative of $$u^2 = 2$$.

So, so far we have this:

$$ \frac{\partial MSE}{\partial m} [(y_i - (mx_i + c))^2] = 2(y_i - (mx_i + c))$$

Lets finish this by deriving the inner function.

Using the constant rule $$y_i$$ and c are 0. As a result we are left with $$-mx_i$$. 

Since we are finding the derivative of the MSE with respect to m, m = 1. So we are left with $$-x_i$$

Putting it altogether we have:

$$ \frac{\partial MSE}{\partial m} [(y_i - (mx_i + c))^2] = \frac{1}{n} \sum_{i=1}^{n} 2(y_i - (mx_i + c)) \cdot -x_i$$

When simplified equals:

$$ \frac{\partial MSE}{\partial m} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - (mx_i + c)) \cdot x_i$$

Lets now differeniate the MSE with respect to c:

Since both m and c share the same outer function we can use the same steps as before to get to this point:

$$ \frac{\partial MSE}{\partial m} [(y_i - (mx_i + c))^2] = 2(y_i - (mx_i + c))$$

We not just need to derivive the inner function with respect to c.

Using the constant rule $$y_i$$ and $$mx_i$$ are 0. As we are finding the derivative with respect to c, c = 1. As a result we are left with -1.

Putting it altogether we have:

$$ \frac{\partial MSE}{\partial m} [(y_i - (mx_i + c))^2] = \frac{1}{n} \sum_{i=1}^{n} 2(y_i - (mx_i + c)) \cdot -1$$

When simplified equals:

$$ \frac{\partial MSE}{\partial m} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - (mx_i + c))$$

### Step 2: Using Gradient Descent

Now that we have the partial derivatives of the Mean Squared Error (MSE) with respect to slope m and intercept c, we can use them to understand how the MSE changes as we adjust these parameters in the model using the update rule. The update rule is as follows:

$$ m_{\text{new}} = m_{\text{old}} - α \cdot \frac{\partial MSE}{\partial m} $$

$$ c_{\text{new}} = c_{\text{old}} - α \cdot \frac{\partial MSE}{\partial c} $$

Using the partial derivatives lets calculate the gradient from our data.

### Step 3: Gradient with Respect to m

Using formula $$ \frac{\partial MSE}{\partial m} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - (mx_i + c)) \cdot x_i$$

$$ \frac{\partial MSE}{\partial m} = -\frac{2}{n} [1(52 - 0) + 2(56 - 0) + 3(60 - 0) + 4(63 - 0) + 5(67 - 0)]$$

$$ = -\frac{2}{5} [1(52) + 2(56) + 3(60) + 4(63) + 5(67)]$$

$$ = -\frac{2}{5} [52 + 112 + 180 + 252 + 355]$$

$$ = -\frac{2}{5} \cdot 931 = -2 \cdot 186.2 = -372.4$$

### Step 4: Gradient with respect to c

Using formula $$ \frac{\partial MSE}{\partial c} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - (mx_i + c))$$

$$ \frac{\partial MSE}{\partial c} = -\frac{2}{n} [(52 - 0) + (56 - 0) + (60 - 0) + (63 - 0) + (67 - 0)]$$

$$ = -\frac{2}{5} [52 + 56 + 60 + 63 + 67]$$

$$ = -\frac{2}{5} \cdot 298 = -2 \cdot 59.6 = -119.2$$

### Step 5: Update m and c

These are the update formulas for m and c respectively:

$$m_{\text{new}} = m_{\text{old}} - α \cdot \frac{\partial MSE}{\partial m}$$

$$c_{\text{new}} = c_{\text{old}} - α \cdot \frac{\partial MSE}{\partial c}$$

Lets choose a learning rate α = 0.01 and use the gradients to update m and c:

>**Note:**
> the learning rate can be any number but 0.01 or 0.001 are generally good options
{: .prompt-info }

$$ m_{\text{new}} = 0 - 0.01 \cdot (-372.4) = 0 + 3.724 = 3.724$$

$$ c_{\text{new}} = 0 - 0.01 \cdot (-119.2) = 0 + 1.192 = 1.192 $$

So, the updated values of m and c are:

$$m = 3.724, c = 1.192$$

### Step 6: Repeat the Process

Now that we have updated m and c, we can use these new values to recalculate predictions, update the gradients, and continue iterating until the values of m and c converge to values that minimise the error.

With our new values we can create new predictions for the dataset.

| Hours Studied (x) | Exam Score (y) | Predicted Score 
|----------|----------|----------|
| 1   | 52   | 4.916   |
| 2   | 56   | 8.64   |
| 3   | 60   | 12.364   |
| 4   | 63   | 16.088   |
| 5   | 67   | 19.812   |

MSE of second iteration:

$$MSE = 2231$$

As we can see the predicted score still does not match as shown by the MSE being relatively high. However the MSE is lower than the previous iteration showing we are converging to the optimal linear regression.

We then fill out the partial derivatives of the MSE with respect to m and c again to get:

$$ \frac{\partial MSE}{\partial m} = -283.32$$

$$ \frac{\partial MSE}{\partial c} = -94.47$$

Now using the update formula with a learning rate α = 0.01:

$$ m_{\text{new}} = 3.724 - 0.01 \cdot (-283.32) = 3.724 + 2.8332 = 6.5572$$

$$ c_{\text{new}} = 1.192 - 0.01 \cdot (-94.47) = 1.192 + 0.9447 = 2.1367 $$

So, the second iteration of m and c are:

$$m = 6.5572, c = 2.1367$$

### Step 7: Final Predicted Equation

After 1000 iterations, the values of m and c will converge, and we will find the optimal values that minimise the MSE. This gives us the final linear regression equation: 

$$ y = 4.11x + 47$$

where:

$$m = 4.11, c = 47$$

This equation can now be used to predict exam scores based on hours studied.





# Conclusion

Throughout this blog we have followed a step-by-step process, that starts with a dataset, we then initialised random values for m and  c, and iteratively updated these parameters using Gradient Descent to minimise the MSE. This resulted in an optimal linear regression model that provides accurate predictions based on the input data.