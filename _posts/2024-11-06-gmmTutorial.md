---
title: Gaussian Mixture Model Matlab Tutorial
date: 2024-11-06 18:00:00 +0000
categories: [Machine Learning, Matlab Tutorial]
tags: [Machine Learning, Clustering, Gaussian Mixture Model, Matlab Tutorial, ]
math: true
---

# Introduction

Welcome to the Gaussian Mixture Model Matlab Tutorial blog. In this blog we will break down how to apply the Gaussian Mixture Model algorithm in Matlab while using toolboxes and with not using toolboxes. This blog aims to add a practical perspective on the Gaussian Mixture Model following the theory blog on Exploring Clustering Methods. This blog will explain all code and confusing jargon, however, if you need a deeper understanding before delving into the practical applications, read the blog on Exploring Clustering Methods. Lets begin with the task we will be covering. 

# Task Overview

You are a data analyst at a confectionery company, and you’ve been assigned to study the weights of candies in a mixed bowl. The bowl contains two types of candies:

- Type A candies, which are generally smaller.
- Type B candies, which are larger on average.

Unfortunately, the candies are mixed together, and the only available data is a list of individual candy weights. You suspect that each type of candy has a distinct average weight but with some variability. Your goal is to use a Gaussian Mixture Model (GMM) to analyse this data and identify the properties of each candy type.

>**Note:**
> Since we are using only one property of the candies (weight) this will be a 1D Gaussian distribution.
{: .prompt-info }

#### Background Data

Generate a dataset of candy weights based on two Gaussian distributions:

- Type A: Mean weight of 5 grams, standard deviation of 1 gram, with 150 samples.
- Type B: Mean weight of 10 grams, standard deviation of 1.5 grams, with 100 samples.

# Step-by-Step Process Without MATLAB Toolbox

## Step 1: Generate Dataset

In MATLAB, we can use the randn function to create normally distributed random values around a given mean with a specified standard deviation with a certain number of samples. Here’s how we generate data for each group:

### Step 1.A: Define the Parameters

- Define the parameters for the data
- Use randn to create an array of 100 and 150 samples
- Scale the random values by the standard deviation and add the mean to get a normal distribution
- Combine the two arrays for later use

```matlab
%% Generate Data

% Parameters for Type A candies
mu1 = 5;      % Mean of Type A
sigma1 = 1;   % Standard deviation of Type A
n1 = 150;     % Number of Type A candies

% Parameters for Type B candies
mu2 = 10;     % Mean of Type B
sigma2 = 1.5; % Standard deviation of Type B
n2 = 100;     % Number of Type B candies

% Generate data from each distribution
data1 = mu1 + sigma1 * randn(n1, 1); % Data points for Type A
data2 = mu2 + sigma2 * randn(n2, 1); % Data points for Type B

data = [data1; data2];
```

### Step 1.B: Find the PDF

The next step is to use the PDF formula to find the initial responsibilities of each data point. Basically, we want to find the probabilities that each sample belongs to a cluster (Type A or B candies). The formula for the PDF is as follows:

$$f(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left( -\frac{(x - \mu)^2}{2 \sigma^2} \right)$$

For code optimisation we can put this formula into a function within MATLAB, and call it when we need it. This will be saved within another .m file but within the same folder as our GMM.

```matlab
function val = gaussian1D(x, mu, sigma)
    %  x     - Input vector
    %  mu    - Mean
    %  sigma - Standard deviation

    %PDF Formula
    val = (1 / (sigma * sqrt(2 * pi))) * exp(-(x - mu).^2 ./ (2 * sigma^2));
end
```

### Step 1.B: Plot Results

Now that we have our PDF function we need to call it to find the PDFs of our two clusters. We can then plot the results to visualise our data.

```matlab
x = [0:0.1:20];
y1 = gaussian1D(x, mu1, sigma1);
y2 = gaussian1D(x, mu2, sigma2);

plot(x, y1, 'b-');
hold on;
plot(x, y2, 'r-');
plot(data1, zeros(size(data1)), 'bx', 'markersize', 10);
plot(data2, zeros(size(data2)), 'rx', 'markersize', 10);
```

#### Code Explanation

- x = [0:0.1:20];

The vector x provides a set of points over which you evaluate the Gaussian functions. Without this, you wouldn't have points to plot the continuous curves. It has the syntax [start:increment:end]. So in this case the vector contains values [0, 0.1, 0.2, 0.3, ..., 29.9, 30].

- plot(x, y1, 'b-');

    This plots the first Gaussian curve (PDF) in blue.
    - x is a vector representing the range of values on the x-axis.
    - y1 is the corresponding pdf values for the first Gaussian distribution.
    - 'b-' specifies the line style and color. 'b' for blue and '-' for a solid line

- hold on;

    This command tells MATLAB to retain the current plot and add new plots on top of it.

- plot(x, y2, 'r-');

    This plots the second Gaussian curve (PDF) in red.

- plot(data1, zeros(size(data1)), 'bx', 'markersize', 10);

    This plots the data points for the first distribution as blue crosses ('bx') along the x-axis.
    - data1 is a vector of data points sampled from the first Gaussian distribution.
    - zeros(size(data1)) places the data points along the x-axis (i.e., at y = 0), creating a horizontal alignment for the markers.
    - 'bx' specifies the marker style and color. 'b' for blue and 'x' for a cross marker
    - 'markersize', 10 sets the marker size for better visibility

- plot(data2, zeros(size(data2)), 'rx', 'markersize', 10);

    This lots the data points for the second distribution as red crosses ('rx') along the x-axis.

### Step 1.C: Entire Code

Here is the entire code to generate and visualise the two Gaussian curves.

```matlab
%% Generate Data
% Parameters for Type A candies
mu1 = 5;      % Mean of Type A
sigma1 = 1;   % Standard deviation of Type A
n1 = 150;     % Number of Type A candies

% Parameters for Type B candies
mu2 = 10;     % Mean of Type B
sigma2 = 1.5; % Standard deviation of Type B
n2 = 100;     % Number of Type B candies

% Generate data from each distribution
data1 = mu1 + sigma1 * randn(n1, 1); % Data points for Type A
data2 = mu2 + sigma2 * randn(n2, 1); % Data points for Type B

% Combine data
data = [data1; data2];

% Range of Values
x = [0:0.1:20];

y1 = gaussian1D(x, mu1, sigma1);
y2 = gaussian1D(x, mu2, sigma2);

plot(x, y1, 'b-');
hold on;
plot(x, y2, 'r-');
plot(data1, zeros(size(data1)), 'bx', 'markersize', 10);
plot(data2, zeros(size(data2)), 'rx', 'markersize', 10);
```

## Step 2: Initialise Parameters for the GMM

Now that we have our dataset we need to set up initial values for the parameters of our GMM. These initial values give the algorithm a starting point to iterate from when fitting the model to the data.

1. Choose the number of clusters (k) — this is how many Gaussian distributions we’ll fit to the data.

2. Initialise the means (mu) — random starting points from the data for each Gaussian’s center.

3. Initialise the variances (sigma) — initially set to the overall variance of the data.

4. Initialise the weights (phi) — set to equal values, assuming each Gaussian component is equally likely.

```matlab
k = 2; % Number of clusters

% Randomly initialise means by selecting k random points
initial_mu = datasample(data, k)';

% Initialize variances with the overall variance of the data
initial_sigma = std(data) * ones(1, k);

% Set equal initial weights for each Gaussian component
initial_phi = ones(1, k) / k;
```

#### Code Explanation

- k = 2; 

    This line sets the variable k to 2, indicating that we are using two Gaussian components (or clusters) in the Gaussian Mixture Model (GMM).

- initial_mu = datasample(data, k)';

    Here, we are initialising the means (mu) of our Gaussian components. 
    - datasample(data, k) selects k random samples from data, which serves as our dataset.
    - datasample(data, k) returns a column vector by default, and we use <strong> ' </strong> to transpose it into a row vector for consistency. The result is a 1-by-2 vector, where each element represents an initial mean for one of the Gaussian components.

- initial_sigma = std(data) * ones(1, k);
    
    This line initialises the variances (sigma) for each Gaussian component.
    - std(data) calculates the standard deviation of the entire dataset (data). Standard deviation is the square root of variance, and this gives a measure of the spread of the data.
    - ones(1, k) creates a 1-by-2 vector of ones, [1, 1]. This is used to replicate the standard deviation across both components.
    - std(data) * ones(1, k) scales each element in the [1, 1] vector by std(data), resulting in a 1-by-2 vector where each entry is equal to the standard deviation of the dataset.
    - This initial initial_sigma vector provides the starting variance values for each Gaussian. We assume both Gaussians have similar spread initially, represented by the overall data spread.

- initial_phi = ones(1, k) / k;

    This line initializes the weights (phi) for each Gaussian component.
    - ones(1, k) again creates a 1-by-2 vector of ones, [1, 1].
    - Dividing by k (/ k) results in a 1-by-2 vector where each element is 1 / k. Since k = 2, this division gives [0.5, 0.5].
    - The result, initial_phi, is a vector where each element represents the initial weight (or proportion) for each Gaussian component.
    - Setting initial_phi to equal values ([0.5, 0.5]) means we assume, initially, that each Gaussian component accounts for half of the data points. This is a neutral starting point and will be refined in later iterations of the EM algorithm.