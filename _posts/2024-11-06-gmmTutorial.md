---
title: Gaussian Mixture Model Matlab Tutorial
date: 2024-11-06 18:00:00 +0000
categories: [Machine Learning, Matlab Tutorial]
tags: [Machine Learning, Clustering, Gaussian Mixture Model, Matlab Tutorial, ]
math: true
---

# Introduction

Welcome to the Gaussian Mixture Model Matlab Tutorial blog. In this blog we will break down how to apply the Gaussian Mixture Model algorithm in Matlab with and without toolboxes. This blog aims to add a practical perspective on the Gaussian Mixture Model following the theory blog on Exploring Clustering Methods. This blog will explain all code and confusing jargon, however, if you need a deeper understanding before delving into the practical applications, read the blog on Exploring Clustering Methods. Lets begin with the task we will be covering. 

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

### Step 1.C: Entire Code and Graph

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

![GMM generated dataset](assets/gmm-fig1.jpg)

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

% Initialise variances with the overall variance of the data
initial_sigma = std(data) * ones(1, k);

% Set equal initial weights for each Gaussian component
initial_phi = ones(1, k) / k;
```

#### Code Explanation

- k = 2; 

    This line sets the variable k to 2, indicating that we are using two Gaussian components (or clusters) in the Gaussian Mixture Model (GMM).

- mu = datasample(data, k)';

    Here, we are initialising the means (mu) of our Gaussian components. 
    - datasample(data, k) selects k random samples from data, which serves as our dataset.
    - datasample(data, k) returns a column vector by default, and we use <strong> ' </strong> to transpose it into a row vector for consistency. The result is a 1-by-2 vector, where each element represents an initial mean for one of the Gaussian components.

- sigma = std(data) * ones(1, k);
    
    This line initialises the variances (sigma) for each Gaussian component.
    - std(data) calculates the standard deviation of the entire dataset (data). Standard deviation is the square root of variance, and this gives a measure of the spread of the data.
    - ones(1, k) creates a 1-by-2 vector of ones, [1, 1]. This is used to replicate the standard deviation across both components.
    - std(data) * ones(1, k) scales each element in the [1, 1] vector by std(data), resulting in a 1-by-2 vector where each entry is equal to the standard deviation of the dataset.
    - This initial sigma vector provides the starting variance values for each Gaussian. We assume both Gaussians have similar spread initially, represented by the overall data spread.

- phi = ones(1, k) / k;

    This line initializes the weights (phi) for each Gaussian component.
    - ones(1, k) again creates a 1-by-2 vector of ones, [1, 1].
    - Dividing by k (/ k) results in a 1-by-2 vector where each element is 1 / k. Since k = 2, this division gives [0.5, 0.5].
    - The result, initial_phi, is a vector where each element represents the initial weight (or proportion) for each Gaussian component.
    - Setting phi to equal values ([0.5, 0.5]) means we assume, initially, that each Gaussian component accounts for half of the data points. This is a neutral starting point and will be refined in later iterations of the EM algorithm.

## Step 3: Implementing the EM algorithm

We will now implement the Expectation-Maximisation (EM) algorithm to fit the GMM to the data. This involves two main steps:

1. Expectation Step (E-step): Calculate the probability that each data point belongs to each Gaussian component.
2. Maximisation Step (M-step): Update the parameters (mean, variance, and weight) of each Gaussian based on the probabilities calculated in the E-step.

### Step 3.A: Expectation Step (E-Step)

To make the code easier to read we can put the E-step in a function and save it in a .m file within the same folder as our GMM. In the E-step, we calculate the responsibility of each Gaussian for each data point, which tells us the probability that a given data point belongs to each component. Here is the code for E-step:

```matlab
function responsibilities = expectation(data, mu, sigma, phi, k)
    n = length(data);               % Number of data points
    responsibilities = zeros(n, k); % Initialise responsibility matrix

    for j = 1:k
        % Calculate the Gaussian pdf for each component
        responsibilities(:, j) = phi(j) * gaussian1D(data, mu(j), sigma(j));
    end

    % Normalise responsibilities to make them sum to 1 for each data point
    responsibilities = responsibilities ./ sum(responsibilities, 2);
end
```

#### Expectation Function Explanation

- <strong> function responsibilities = expectation(data, mu, sigma, phi, k) </strong>

    This function returns responsibilities, an n x k matrix where each entry represents the responsibility of a Gaussian component for each data point.
    - data: The vector containing the data points (e.g., weights of candies).
    - mu: A vector of means for each Gaussian component.
    - sigma: A vector of standard deviations for each Gaussian component.
    - phi: A vector of mixture weights (proportions) for each Gaussian component.
    - k: The number of Gaussian components (clusters).

- <strong> n = length(data); </strong>

    This value n is used to initialise the responsibility matrix. Each row will represent a data point, and each column will represent a Gaussian component. Since data = data1 + data2, n = 250.

- <strong> responsibilities = zeros(n, k); </strong>

    This creates an n x k matrix filled with zeros. It will store the responsibilities of each Gaussian component for each data point. For 250 data points and 2 Gaussian components, responsibilities will be a 250 x 2 matrix initialised with zeros.
    - n: Number of rows (one for each data point).
    - k: Number of columns (one for each Gaussian component). 

- <strong> for j = 1:k </strong>

    This loop iterates over each Gaussian component, calculating the weighted probability density for each data point with respect to that component.

- <strong> responsibilities(:, j) = phi(j) * gaussian1D(data, mu(j), sigma(j)); </strong>

    This stores the weighted pdf values in the j-th column of the responsibilities matrix. Where each row corresponds to a data point, and each column represents the responsibility of a specific Gaussian component.
    - gaussian1D(data, mu(j), sigma(j)); : This gives the probability density values for each data point, assuming it belongs to the j-th Gaussian.
    - phi(j) * gaussian1D(...): Multiplies each pdf value by the weight phi(j) of the j-th component. This weight represents the prior probability of a data point being generated by the j-th Gaussian component. The result is a weighted probability density for each data point under the j-th Gaussian component. 
    - responsibilities(:, j) = ...: Stores the weighted pdf values in the j-th column of the responsibilities matrix. Each row corresponds to a data point, and each column represents the responsibility of a specific Gaussian component.

- <strong> responsibilities = responsibilities ./ sum(responsibilities, 2); </strong>

    This normalises the responsibilities so that, for each data point, the sum of responsibilities across all components is 1. This is an essential step in soft clustering, where each data point has fractional membership in each Gaussian component.
    - sum(responsibilities, 2): Calculates the sum of responsibilities across all components (columns) for each data point (row). The result is an n x 1 column vector where each element is the sum of weighted pdf values for a particular data point across all components.
    - responsibilities = responsibilities ./ sum(..., 2); : Divides each entry in responsibilities by the sum of its row.

### Step 3.B: Maximisation Step (M-Step)

Now that we have our E-step we can create another function for the Maximisation step. In the M-step, we use the responsibilities to calculate new estimates for the parameters of each Gaussian. Here is the code for M-step:

```matlab
function [mu, sigma, phi] = maximisation(data, responsibilities, k) 
    % Initialise new parameters
    mu = zeros(1, k);
    sigma = zeros(1, k);
    phi = zeros(1, k);

    for j = 1:k
        % Update the weights (phi) for each component
        phi(j) = mean(responsibilities(:, j));

        % Update the mean (mu) for each component
        mu(j) = sum(responsibilities(:, j) .* data) / sum(responsibilities(:, j));

        % Update the variance (sigma) for each component
        sigma(j) = sqrt(sum(responsibilities(:, j) .* (data - mu(j)).^2) / sum(responsibilities(:, j)));
    end
end
```

#### Maximisation Function Explanation

- <strong> function [mu, sigma, phi] = maximization(data, responsibilities, k) </strong>

    This function returns updated values for mu (mean), sigma (standard deviation) and phi (weights) for each component using:
    - data: A vector of data points (weights of candies).
    - responsibilities: An n x k matrix where each entry responsibilities(i, j) represents the probability (responsibility) of the j-th Gaussian component for the i-th data point.
    - k: The number of Gaussian components (clusters).

- <strong> mu = zeros(1, k); </strong>

    This initialises a 1-by-k vector of zeros to store the new mean values for each component. Initially, all values are zero, and they will be updated within the loop.

- <strong> sigma = zeros(1, k); </strong>

    This initialises a 1-by-k vector of zeros for the new variances (standard deviations squared) for each component.

- <strong> phi = zeros(1, k); </strong>

    This initialises a 1-by-k vector of zeros for the new weights (proportions) of each component.
    These initialisations provide storage for the updated values that we will calculate in the following loop.

- <strong> for j = 1:k </strong>

    This loop iterates over each Gaussian component, calculating the weighted probability density for each data point with respect to that component.

- <strong> phi(j) = mean(responsibilities(:, j)); </strong>

    This line updates the phi (weight) of the j-th Gaussian component.
    - responsibilities(:, j): Selects all rows in the j-th column of responsibilities, giving the responsibility of the j-th component for each data point.
    - mean(responsibilities(:, j)): Calculates the average responsibility of the j-th component across all data points.
    - phi(j) = ...: Stores this value in phi(j), updating the weight for the j-th component.

- <strong> mu(j) = sum(responsibilities(:, j) .* data) / sum(responsibilities(:, j)); </strong>

    This line updates the mean mu(j) for the j-th Gaussian component.
    - responsibilities(:, j) .* data: Multiplies each data point by the responsibility of the j-th component for that data point. This is a weighted sum, where data points with higher responsibilities contribute more to the mean.
    - sum(responsibilities(:, j) .* data): Sums up these weighted values, resulting in the weighted sum of data points for the j-th component.
    - mu(j) = ...: Divides the weighted sum of data points by the sum of responsibilities to calculate the weighted average (mean) for the j-th component, and stores it in mu(j).

- <strong> sigma(j) = sqrt(sum(responsibilities(:, j) .* (data - mu(j)).^2) / sum(responsibilities(:, j))); </strong>

    This line updates the standard deviation sigma(j) for the j-th Gaussian component.
    - (data - mu(j)).^2: Calculates the squared difference between each data point and the current mean mu(j). This measures how far each data point deviates from the mean.
    - responsibilities(:, j) .* (data - mu(j)).^2: Multiplies each squared difference by the responsibility of the j-th component for that data point. This is a weighted squared deviation, where points with higher responsibility contribute more.
    - sum(responsibilities(:, j) .* (data - mu(j)).^2): Sums up these weighted squared deviations, giving the total weighted variance for the j-th component.
    - sum(responsibilities(:, j)): Sums up the responsibilities of the j-th component across all data points, serving as a normalizing factor.
    - sqrt(...): Takes the square root of the result to obtain the standard deviation (since sigma represents standard deviation, not variance).
    - sigma(j) = ...: Stores the calculated standard deviation for the j-th component in sigma(j).

### Step 3.C: EM Algorithm

Now that we have our E-step and M-step we can implement our EM algorithm.

```matlab
%% EM Algorithm
max_iters = 100;  % Maximum number of iterations
tolerance = 1e-4; % Convergence threshold

for iter = 1:max_iters
    % E-step: Calculate responsibilities
    responsibilities = expectation(data, mu, sigma, phi, k);
    
    % M-step: Update parameters based on responsibilities
    prev_mu = mu; % Store previous means for convergence check
    [mu, sigma, phi] = maximization(data, responsibilities, k);

    % Check for convergence
    if max(abs(mu - prev_mu)) < tolerance
        fprintf('Converged after %d iterations\n', iter);
        break;
    end
end
```

#### EM Algorithm Explanation

- <strong> max_iters = 100; </strong>

    Sets the maximum number of iterations for the EM algorithm. If convergence isn’t reached within these iterations, the loop will stop.

- <strong> tolerance = 1e-4; </strong>

    Sets the convergence threshold. This value is used to determine if the algorithm has converged by checking if the change in the means (mu) falls below this threshold.

- <strong> for iter = 1:max_iters </strong>

    This for loop iterates up to max_iters times. On each iteration, it performs an E-step followed by an M-step to update the model parameters.

- <strong> responsibilities = expectation(data, mu, sigma, phi, k); </strong>

    Calls the expectation function to calculate the responsibilities for each data point with respect to each Gaussian component.

- <strong> prev_mu = mu; </strong>

    Stores the current means (mu) in prev_mu so that we can check for convergence after updating the parameters.

- <strong> [mu, sigma, phi] = maximisation(data, responsibilities, k); </strong>

    Calls the maximisation function to update the parameters based on the current responsibilities.

- <strong> if max(abs(mu - prev_mu)) < tolerance </strong>

    This checks if the maximum change in the means (mu) between the current and previous iteration is less than the specified tolerance. If this maximum change is smaller than tolerance, the algorithm is considered to have converged.
    - mu - prev_mu calculates the difference between the updated and previous means.
    - abs(mu - prev_mu) takes the absolute value of these differences.
    - max(...) finds the largest difference.

- <strong> fprintf('Converged after %d iterations\n', iter); </strong>

    If convergence is reached, this line prints a message indicating the number of iterations it took to converge.

- <strong> break; </strong>

    This exits the loop early if convergence is detected.

## Step 4: Plot Results

Now that we have our converged GMM graph we can plot the data.

```matlab
%% Plot Results
figure;
x = linspace(min(data), max(data), 100); % Range of x values for plotting

% Calculate the pdf of each Gaussian component
pdf1 = phi(1) * normpdf(x, mu(1), sigma(1));
pdf2 = phi(2) * normpdf(x, mu(2), sigma(2));

% Combine the components to get the overall GMM pdf
gmm_pdf = pdf1 + pdf2;

% Plot each Gaussian component
plot(x, pdf1, 'r--', 'LineWidth', 1.5);
hold on;
plot(x, pdf2, 'b--', 'LineWidth', 1.5);

% Plot the combined GMM pdf
plot(x, gmm_pdf, 'k-', 'LineWidth', 2);

title('GMM Fit to Candy Weights');
xlabel('Weight (grams)');
ylabel('Density');
legend('Type A PDF', 'Type B PDF', 'GMM PDF');
hold off;
```

Here are the final results of the converged GMM.

![GMM Results](assets/gmm-fig2.jpg)