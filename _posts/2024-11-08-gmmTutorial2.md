---
title: Gaussian Mixture Model Matlab Tutorial - 2D
date: 2024-11-06 18:00:00 +0000
categories: [Machine Learning, Matlab Tutorial]
tags: [Machine Learning, Clustering, Gaussian Mixture Model, Matlab Tutorial, ]
math: true
---

# Introduction

Welcome to the second Gaussian Mixture Model Matlab Tutorial blog. In this blog we will break down how to apply a multi-dimensional Gaussian Mixture Model in Matlab without the GMM toolbox. This blog aims to add a practical perspective on the Gaussian Mixture Model following the theory blog on Exploring Clustering Methods. This blog will explain all code and confusing jargon, however, if you need a deeper understanding before delving into the practical applications, read the blog on Exploring Clustering Methods. Lets begin with the task we will be covering. 

# Task Overview

You are a data scientist at an environmental research institute studying the migratory patterns of birds. You’ve been assigned to analyse a dataset containing observations from three distinct bird species:

- Species X
    - Sightings: 361
    - Mean Coordinates: Latitude 37, Longitude 57
    - Covariance = [4, 1.2; 1.2, 3]

- Species Y
    - Sightings: 280
    - Mean Coordinates: Latitude 40, Longitude 50
    - Covariance = [3, -0.8; -0.8, 2.5]

- Species Z
    - Sightings: 197
    - Mean Coordinates: Latitude 48, Longitude 43
    - Covariance = [2.5, 0.6; 0.6, 1.8]

Unfortunately, the observations are combined, and you only have the coordinates (latitude and longitude) of each bird sighting during migration season. You suspect each species follows a distinct migration pattern, represented by clusters in 2D space. Your goal is to use a Gaussian Mixture Model (GMM) to analyse this data and identify the migration properties for each bird species.

>**Note:**
> Since we have two properties (latitude and longitude) this will be a 2D Gaussian distribution.
{: .prompt-info }

# Step-by-Step Process Without MATLAB Toolbox

## Step 1: Generate Dataset

In MATLAB, we can use the randn function to create normally distributed random values around a given mean with a specified standard deviation with a certain number of samples. Here’s how we generate data for each group:

### Step 1.A: Define the Parameters

```matlab
% Define parameters for the different species

% Species X
n1 = 367;                       % Number of points (sightings)
mu1 = [37, 57];                 % Mean for species X (Coordinates)
sigma1 = [4, 1.2; 1.2, 3];      % Covariance matrix (Migration Pattern)


% Species Y
n2 = 280;
mu2 = [40, 50];                
sigma2 = [3, -0.8; -0.8, 2.5]; 


% Species Z
n3 = 197;                   
mu3 = [48, 43];                 
sigma3 = [2.5, 0.6; 0.6, 1.8];  
```

### Step 1.B: Generate Data

```matlab
% Generate data for each species based on background data
dataX = mvnrnd(mu1, sigma1, n1);
dataY = mvnrnd(mu2, sigma2, n2);
dataZ = mvnrnd(mu3, sigma3, n3);

% Combine all data into one dataset
Dataset = [dataX; dataY; dataZ];
```

#### Code Explanation

- <strong> mvnrnd(mu1, sigma1, n1) </strong>

    The function mvnrnd takes the mean (mu), covariance (sigma), and the number of points (n) to generate a set of points that form a multivariate Gaussian distribution around the specified mean. The majority of the generated points will be clustered around the mean value (mu), representing the center of the Gaussian distribution but due to the Gaussian (bell curve) nature, some points will be further away from the mean, which are natural outliers in a Gaussian distribution. The covariance matrix (sigma) controls how spread out the points are around the mean in each dimension.

- <strong> dataX = mvnrnd(mu1, sigma1, n1); </strong>

    - Generates n1 random data points for Species X.
    - mu1 specifies the 2D mean (center) for Species X's sightings.
    - sigma1 is the 2x2 covariance matrix, defining the spread and directional relationship (correlation) between the two dimensions (e.g., latitude and longitude) for Species X.
    - The result, dataX, is an n1 x 2 matrix where each row represents one sighting with latitude and longitude values.

- <strong> dataY = mvnrnd(mu2, sigma2, n2); </strong>

    - Generates n2 random data points for Species Y.
    - mu2 provides the mean coordinates for Species Y's sightings.
    - sigma2 is the covariance matrix for Species Y, shaping its spatial spread and orientation.
    - The output, dataY, is an n2 x 2 matrix.

- <strong> dataZ = mvnrnd(mu3, sigma3, n3); </strong>

    - Generates n3 random data points for Species Z.
    - mu3 is the mean vector for Species Z
    - sigma3 is the covariance matrix defining its spread and orientation.
    - The output, dataZ, is an n3 x 2 matrix.

- <strong> Dataset = [dataX; dataY; dataZ]; </strong>

    This line combines the data points for all three species into a single matrix, Dataset. This combined dataset represents all the sightings, with each row in Dataset representing the latitude and longitude of a bird sighting from one of the three species.

### Step 1.C: Plot Data

```matlab
% Plot the generated data points for each cluster
figure;
scatter(dataX(:,1), dataX(:,2), 10, 'r', 'filled'); % Cluster 1 in red
hold on;
scatter(dataY(:,1), dataY(:,2), 10, 'b', 'filled'); % Cluster 2 in blue
hold on;
scatter(dataZ(:,1), dataZ(:,2), 10, 'g', 'filled'); % Cluster 3 in green
```

#### Code Explanation

- <strong> figure; </strong>

    This opens a new figure window for the plot and ensures that the plot is created in a separate window, allowing us to visualise the data points clearly.

- <strong> scatter(dataX(:,1), dataX(:,2), 10, 'r', 'filled'); </strong>

    This line plots each data point for Species X in red on the scatter plot.
    - <strong>scatter</strong>: Creates a scatter plot of data points.
    - <strong>dataX(:,1)</strong>: Selects all rows in the first column of dataX, representing the x-coordinates (latitude) of Species X.
    - <strong>dataX(:,2)</strong>: Selects all rows in the second column of dataX, representing the y-coordinates (longitude) of Species X.
    - <strong>10</strong>: Sets the marker size to 10.
    - <strong>'r'</strong>: Sets the marker color to red .
    - <strong>'filled'</strong>: Fills the scatter markers with the specified color (red).

- <strong> hold on; </strong>

    This tells MATLAB to keep the current plot and add new plots to it without clearing the existing data points. This allows us to plot data for multiple species (Species X, Y, and Z) on the same figure.


### Step 1.D: Calculate and Plot Gaussian PDF

```matlab
% Define grid for plotting Gaussian contours
x = linspace(min(Dataset(:,1))-5, max(Dataset(:,1))+5, 100);
y = linspace(min(Dataset(:,2))-5, max(Dataset(:,2))+5, 100);
[X, Y] = meshgrid(x, y);

% Calculate and plot Gaussian PDF contour for each cluster

% Cluster 1
Z1 = mvnpdf([X(:) Y(:)], mu1, sigma1);
Z1 = reshape(Z1, size(X));
contour(X, Y, Z1, 'LineColor', 'r', 'LineWidth', 1.5);

% Cluster 2
Z2 = mvnpdf([X(:) Y(:)], mu2, sigma2);
Z2 = reshape(Z2, size(X));
contour(X, Y, Z2, 'LineColor', 'b', 'LineWidth', 1.5);

% Cluster 3
Z3 = mvnpdf([X(:) Y(:)], mu3, sigma3);
Z3 = reshape(Z3, size(X));
contour(X, Y, Z3, 'LineColor', 'g', 'LineWidth', 1.5);

% Add labels and title
title('Initial Dataset for Gaussian Mixture Model');
xlabel('Feature 1');
ylabel('Feature 2');
legend('Cluster 1 Data', 'Cluster 2 Data', 'Cluster 3 Data', 'Cluster 1 PDF', 'Cluster 2 PDF', 'Cluster 3 PDF');
hold off;
```

#### Code Explanation

- <strong> linspace </strong>

    The function linspace in MATLAB generates a vector of linearly spaced values between two specified endpoints. It has the syntax linspace(start, end, num_points) where start is the starting value of the range, end is the ending value of the range, and num_points is the number of points to generate between start and end. For example:

    ```matlab
    x = linspace(0, 10, 5)
    ```

    This line will create a vector x with 5 linearly spaced points from 0 to 10:

    ```matlab
    x = [0, 2.5, 5, 7.5, 10]
    ```

- <strong> x = linspace(min(Dataset(:,1))-5, max(Dataset(:,1))+5, 100); </strong>

    By using linspace to generate 100 points along both the x and y axes, we get a high-resolution grid across the plot’s range i.e. helps define the smoothness and resolution of the contours. Without it the contours would be jagged.
    - <strong>min(Dataset(:,1))</strong>: Finds the minimum x-coordinate (first column) in Dataset.
    - <strong>max(Dataset(:,1))</strong>: Finds the maximum x-coordinate in Dataset.
    - The -5 and +5 extend the range a bit, ensuring the contour plot covers the entire dataset with some padding.

- <strong> y = linspace(min(Dataset(:,2))-5, max(Dataset(:,2))+5, 100); </strong>

    Similar to x, but for the y-dimension (second column) of Dataset.

- <strong> meshgrid </strong>

    meshgrid in MATLAB is a function that creates a grid of coordinates based on two input vectors, which is useful for evaluating functions over a 2D space. It takes two 1D arrays (vectors) and generates two 2D matrices that represent all possible combinations of those input values. It has the syntax [X, Y] = meshgrid(x, y), where x is a vector defining the x-coordinates of the grid, y is a vector defining the y-coordinates of the grid, and [X, Y] is a 2D matrix where each element in X represents an x-coordinate, and each element in Y represents the corresponding y-coordinate at that position. For example:

    ```matlab
    x = [1, 2, 3];
    y = [4, 5, 6];
    [X, Y] = meshgrid(x, y);
    ```

    The output will be:

    ```matlab
    X = 
        1     2     3
        1     2     3
        1     2     3

    Y = 
        4     4     4
        5     5     5
        6     6     6
    ```

    Here:
    - X has each row as [1, 2, 3], representing all x-coordinates across each row of the grid.
    - Y has each column as [4; 5; 6], representing all y-coordinates across each column of the grid.

    So in our example each (X(i, j), Y(i, j)) pair represents a unique point in 2D space where the Gaussian PDF will be calculated, enabling us to plot contours (levels of constant density) across the entire grid.

- <strong> Z1 = mvnpdf([X(:) Y(:)], mu1, sigma1); </strong>

    This computes the multivariate Gaussian probability density at each point in the grid defined by X and Y, based on a Gaussian distribution with mean mu1 and covariance sigma1.
    - <strong>X(:) and Y(:)</strong>: The (:) operator converts the X and Y matrices from meshgrid into column vectors. For example if X and Y are each 100x100 matrices (from a 100-point grid defined by linspace), X(:) and Y(:) will each become 10,000x1 column vectors.
    - <strong>[X(:) Y(:)]</strong>: This concatenates the two column vectors horizontally to create a 10,000x2 matrix where each row in this matrix represents an (x, y) coordinate on the 2D grid.
    - <strong>mvnpdf</strong>: Computes the multivariate Gaussian PDF for each point on the grid (x, y coordinates), based on mu1 (mean of Species X) and sigma1 (covariance of Species X).
    - <strong>Output Z1</strong>: mvnpdf returns a 10,000x1 column vector Z1, where each element contains the computed probability density for each corresponding (x, y) coordinate in the grid. Higher values in Z1 indicate points closer to the mean mu1, while lower values indicate points further from the mean, where the density drops off.

- <strong> Z1 = reshape(Z1, size(X)); </strong>

    The reshape function converts Z1 from a 10,000x1 column vector back into a 100x100 matrix (the same size as X and Y). By reshaping, we map each probability density back to its corresponding location in the X and Y grid. This makes it easier to plot contours, as we can now use the grid matrices X, Y, and Z1 directly in functions like contour.

- <strong> contour(X, Y, Z1, 'LineColor', 'r', 'LineWidth', 1.5); </strong>

    This function plots contour lines, which are lines that connect points of equal value. In this context, the contour lines represent levels of constant probability density in the Gaussian distribution.
    - <strong>X and Y</strong>: These are the 2D grid matrices created by meshgrid. They represent all x and y coordinates over a defined range.
    - <strong>Z1</strong>: This is a matrix containing the probability density values calculated at each (x, y) grid point using the Gaussian PDF (mvnpdf). Each element Z1(i, j) represents the probability density at the location (X(i, j), Y(i, j)). contour uses these values to plot contour lines where the density is constant, forming ellipses around the mean of the Gaussian distribution.
    - <strong>LineColor, 'r'</strong>: Sets the contour line color to red for Species X.
    - <strong>LineWidth, 1.5</strong>: Sets the width of the contour lines to 1.5.


### Step 1.E: Entire Code and Graph

```matlab
% Set random seed for reproducibility
rng(0);

% Define parameters for the different species

% Species X
n1 = 367;                       % Number of points (sightings)
mu1 = [37, 57];                 % Mean for species X (Coordinates)
sigma1 = [4, 1.2; 1.2, 3];      % Covariance matrix (Migration Pattern)


% Species Y
n2 = 280;
mu2 = [40, 50];                
sigma2 = [3, -0.8; -0.8, 2.5]; 


% Species Z
n3 = 197;                   
mu3 = [48, 43];                 
sigma3 = [2.5, 0.6; 0.6, 1.8];  

% Generate data for each species based on background data
dataX = mvnrnd(mu1, sigma1, n1);
dataY = mvnrnd(mu2, sigma2, n2);
dataZ = mvnrnd(mu3, sigma3, n3);

% Combine all data into one dataset
Dataset = [dataX; dataY; dataZ];




% Plot the generated data points for each cluster
figure;
scatter(dataX(:,1), dataX(:,2), 10, 'r', 'filled'); % Cluster 1 in red
hold on;
scatter(dataY(:,1), dataY(:,2), 10, 'b', 'filled'); % Cluster 2 in blue
hold on;
scatter(dataZ(:,1), dataZ(:,2), 10, 'g', 'filled'); % Cluster 3 in green


% Define grid for plotting Gaussian contours
x = linspace(min(Dataset(:,1))-5, max(Dataset(:,1))+5, 100);
y = linspace(min(Dataset(:,2))-5, max(Dataset(:,2))+5, 100);
[X, Y] = meshgrid(x, y);



% Calculate and plot Gaussian PDF contour for each cluster

% Cluster 1
Z1 = mvnpdf([X(:) Y(:)], mu1, sigma1);
Z1 = reshape(Z1, size(X));
contour(X, Y, Z1, 'LineColor', 'r', 'LineWidth', 1.5);

% Cluster 2
Z2 = mvnpdf([X(:) Y(:)], mu2, sigma2);
Z2 = reshape(Z2, size(X));
contour(X, Y, Z2, 'LineColor', 'b', 'LineWidth', 1.5);

% Cluster 3
Z3 = mvnpdf([X(:) Y(:)], mu3, sigma3);
Z3 = reshape(Z3, size(X));
contour(X, Y, Z3, 'LineColor', 'g', 'LineWidth', 1.5);

% Add labels and title
title('Initial Dataset for Gaussian Mixture Model');
xlabel('Latitude');
ylabel('Longitude');
legend('Species X Data', 'Species Y Data', 'Species Z Data', 'Species X PDF', 'Species Y PDF', 'Species Z PDF');
hold off;
```

![GMM generated dataset](assets/clustering-tutorial/gmm2d-fig1.jpg)

## Step 2: Initialise Parameters for the GMM

Now that we have our dataset we need to set up initial values for the parameters of our GMM. These initial values give the algorithm a starting point to iterate from when fitting the model to the data.

1. Choose the number of clusters (k) — this is how many Gaussian distributions we’ll fit to the data.

2. Initialise the means (mu) — random starting points from the data for each Gaussian’s center.

3. Initialise the variances (sigma) — initially set to the overall variance of the data.

4. Initialise the weights (phi) — set to equal values, assuming each Gaussian component is equally likely.

```matlab
% Set the number of clusters
k = 3;

% Initialise mixing coefficients (phi) equally
phi = ones(1, k) / k;  

% Initialise means (mu) by selecting random data points
mu = Dataset(randperm(size(Dataset, 1), k), :);  

% Initialise covariance matrices (sigma) as identity matrices
d = size(Dataset, 2);  
sigma = repmat(eye(d), [1, 1, k]);  
```

#### Code Explanation

- <strong> k = 3 </strong>

    This defines the number of clusters (components) for the Gaussian Mixture Model (GMM). Here, we are assuming that we have three clusters in the dataset, so k = 3.

- <strong> phi = ones(1, k) / k; </strong>

    This line initializes the mixing coefficients (weights) for each cluster.
    - <strong>ones(1, k)</strong>: Creates a row vector of ones with length k for example [1, 1, 1] for k = 3.
    - <strong>ones(1, k) / k</strong>: Divides each element by k, resulting in equal weights for each cluster.
    - For k = 3, phi will be [1/3, 1/3, 1/3] (or approximately [0.33, 0.33, 0.33]), indicating that each cluster initially has an equal probability of containing a data point.

- <strong> mu = Dataset(randperm(size(Dataset, 1), k), :); </strong>

    This initializes the means for each cluster by randomly selecting points from the dataset.
    - <strong>size(Dataset, 1)</strong>: Returns the number of rows (data points) in Dataset
    - <strong>randperm(size(Dataset, 1), k)</strong>: Generates k random unique indices (integers) between 1 and the number of data points. This selects k random rows from Dataset
    - <strong>Dataset(randperm(size(Dataset, 1), k), :)</strong>: Uses these random indices to select k rows from Dataset, with each row representing a data point.
    - Result: mu is a k x d matrix, where each row is the mean of a cluster, and d is the number of dimensions in the dataset. This initializes the cluster centers to be near data points. For example, if Dataset contains 2D data (e.g., latitude and longitude) and we select three random points, mu might look like:

    ```matlab
    mu = [
        36, 57;
        40, 50;
        48, 43
    ];
    ```

- <strong> d = size(Dataset, 2); </strong>

    This determines the number of dimensions in the dataset.
    - <strong>size(Dataset, 2)</strong>: Returns the number of columns in Dataset, which corresponds to the dimensionality (d) of each data point. For a 2D dataset, d would be 2.

- <strong> sigma = repmat(eye(d), [1, 1, k]); </strong>

    Initialises the covariance matrices for each cluster as identity matrices.
    - <strong>eye(d)</strong>: Creates a d x d identity matrix, which has 1s along the diagonal and 0s elsewhere. This represents a spherical Gaussian distribution (equal variance in all directions).
    - <strong>repmat(eye(d), [1, 1, k])</strong>: Replicates the identity matrix k times along a third dimension to create a d x d x k array. Each d x d slice along the third dimension represents the initial covariance matrix for a cluster.
    - Result: sigma is a d x d x k array, where each slice sigma(:, :, i) is a covariance matrix for the i-th cluster. For example a 2D dataset (d = 2) with three clusters (k = 3), the initialised sigma would look like this:

    ```matlab
    sigma(:, :, 1) = [1 0; 0 1]; % Covariance matrix for the first cluster
    sigma(:, :, 2) = [1 0; 0 1]; % Covariance matrix for the second cluster
    sigma(:, :, 3) = [1 0; 0 1]; % Covariance matrix for the third cluster
    ```

## Step 3: Implementing the EM algorithm

Using the parameters we have initialised we will now implement the Expectation-Maximisation (EM) algorithm to fit the GMM to the data. This involves two main steps:

1. Expectation Step (E-step): Calculate the probability that each data point belongs to each Gaussian component (cluster).
2. Maximisation Step (M-step): Update the parameters (mean, variance, and weight) of each Gaussian based on the probabilities calculated in the E-step.

### Step 3.A: Expectation Step (E-Step)

To make the code easier to read we can put the E-step in a function and save it in a .m file within the same folder as our GMM. In the E-step, we calculate the responsibility of each Gaussian for each data point, which tells us the probability that a given data point belongs to each component. Here is the code for E-step:

```matlab
function responsibilities = expectation(Dataset, mu, sigma, phi, k)
    n = length(Dataset);                 Number of data points
    responsibilities = zeros(n, k);     % Initialise responsibility matrix

    for i = 1:k
        % Calculate the Gaussian pdf for each component
        responsibilities(:, i) = phi(i) * mvnpdf(Dataset, mu(i, :), sigma(:, :, i));
    end

    responsibilities = responsibilities ./ sum(responsibilities, 2);  % Normalize responsibilities

end
```

#### Expectation Function Explanation

- <strong> function responsibilities = expectation(...) </strong> 

    This defines a function called expectation that returns the responsibility matrix.
    - <strong>Dataset</strong> : The matrix containing all data points, where each row is a data point.
    - <strong>mu</strong> : The matrix of means for each cluster (each row corresponds to a cluster mean).
    - <strong>sigma</strong> : The 3D array of covariance matrices for each cluster.
    - <strong>phi</strong> : The vector of mixing coefficients for each cluster.
    - <strong>k</strong> : The number of clusters.
    - <strong>Result</strong> : responsibilities, an n x k matrix where each element responsibilities(j, i) holds the probability of data point j belonging to cluster i.

- <strong> n = length(Dataset); </strong> 

    Calculates the number of data points in Dataset by finding gives number of rows.

- <strong> responsibilities = zeros(n, k); </strong> 

    This initialises the responsibility matrix responsibilities as an n x k matrix of zeros.

- <strong> for i = 1:k </strong> 

    Loops through each cluster i from 1 to k

- <strong> responsibilities(:, i) = phi(i) * mvnpdf(Dataset, mu(i, :), sigma(:, :, i)); </strong> 

    - <strong>mvnpdf(Dataset, mu(i, :), sigma(:, :, i))</strong> : Computes the Gaussian probability density for each data point in Dataset with respect to cluster i’s mean (mu(i, :)) and covariance (sigma(:, :, i)).
    - <strong>phi(i) * mvnpdf(...)</strong> : Multiplies this density by the cluster’s mixing coefficient (phi(i)), creating a weighted probability for each data point belonging to cluster i.
    - <strong>responsibilities(:, i) = ...</strong> : Stores these probabilities in the i-th column of responsibilities, where each entry represents the responsibility of cluster i for a particular data point.

- <strong> responsibilities = responsibilities ./ sum(responsibilities, 2); </strong> 

    This normalises the responsibilities for each data point so that each row in responsibilities sums to 1.
    - <strong>sum(responsibilities, 2)</strong>: Computes the sum of each row in responsibilities. The 2 as the second argument tells MATLAB to sum along the rows, giving an n x 1 vector where each entry is the sum of probabilities for a data point across all clusters. For example: 

    ```matlab
    % Assume responsibilities is a 3x3 matrix for simplicity
    responsibilities = [
        0.1, 0.2, 0.3;
        0.4, 0.3, 0.2;
        0.2, 0.5, 0.3
    ];

    % sum(responsibilities, 2) would yield:
    % [0.6; 0.9; 1.0]
    ```

    - <strong>responsibilities ./ sum(...)</strong>: This part divides each element in responsibilities by the sum of its row. To ensure that each row of responsibilities sums to 1. For example: 

    ```matlab
    % Normalized responsibilities:
    responsibilities = [
        0.1 / 0.6, 0.2 / 0.6, 0.3 / 0.6;  % Row sums to 1
        0.4 / 0.9, 0.3 / 0.9, 0.2 / 0.9;  % Row sums to 1
        0.2 / 1.0, 0.5 / 1.0, 0.3 / 1.0   % Row sums to 1
    ];
    ```

### Step 3.B: Maximisation Step (M-Step)

Now that we have our E-step we can create another function for the Maximisation step. In the M-step, we use the responsibilities to calculate new estimates for the parameters of each Gaussian (cluster). Here is the code for M-step:

```matlab
function [mu, sigma, phi] = maximisation(Dataset, responsibilities, k)
   
    [n, d] = size(Dataset); % n = number of data points, d = number of dimensions

    % Initialise new parameters
    mu = zeros(k, d);            
    sigma = zeros(d, d, k);         
    phi = zeros(1, k);             

    for j = 1:k
        % Update the mixing coefficient (phi) for each component
        phi(j) = mean(responsibilities(:, j));

        % Update the mean (mu) for each component
        mu(j, :) = sum(responsibilities(:, j) .* Dataset) / sum(responsibilities(:, j));

        % Update the covariance matrix (sigma) for each component
        diff = Dataset - mu(j, :);  % Difference from mean
        sigma(:, :, j) = (responsibilities(:, j) .* diff)' * diff / sum(responsibilities(:, j)); 
    end
end
```

#### Maximisation Function Explanation

- <strong> function [mu, sigma, phi] = maximisation(...) </strong> 

    This defines a function called maximisation that returns mu (the means), sigma (the covariances), and phi (the mixing coefficients) for each cluster.
    - <strong>Dataset</strong>: An n x d matrix containing n data points, each with d features or dimensions.
    - <strong>responsibilities</strong>: An n x k matrix where each entry responsibilities(j, i) represents the probability that data point j belongs to cluster i.
    - <strong>k</strong>: The number of clusters.

- <strong> [n, d] = size(Dataset); </strong> 

    This extracts the size of Dataset for further use.
    - <strong>n</strong>: The number of rows in Dataset, representing the total number of data points.
    - <strong>d</strong>: The number of columns in Dataset, representing the number of dimensions (features) in each data point.

- <strong> mu = zeros(k, d); </strong> 

    This initialises mu as a k x d matrix to store the mean of each cluster. Each row in mu will be the mean vector of one cluster. For example if k = 3 and d = 2, mu will look like:

    ```matlab
        mu = [
            0, 0;
            0, 0;
            0, 0
        ]
    ```

- <strong> sigma = zeros(d, d, k); </strong> 

    This initialises sigma as a d x d x k array to store the covariance matrix for each cluster. Each d x d slice along the third dimension will represent one cluster’s covariance matrix. For example if k = 3 and d = 2, sigma will look like:

    ```matlab
        sigma = [
            [1, 0;    % Covariance matrix for cluster 1
            0, 1],

            [1, 0;    % Covariance matrix for cluster 2
            0, 1],

            [1, 0;    % Covariance matrix for cluster 3
            0, 1]
        ]
    ```

- <strong> phi = zeros(1, k); </strong> 

    This initialises phi as a 1 x k vector to store the mixing coefficients for each cluster. For example if k = 3, sigma will look like:

    ```matlab
        phi = [0, 0, 0;]
    ```

- <strong> for j = 1:k </strong> 

    This loop iterates over each cluster j from 1 to k. Within each iteration, it updates the parameters for the current cluster j.

- <strong> phi(j) = mean(responsibilities(:, j)); </strong> 

    This calculates the mixing coefficient (weight) for cluster j
    - <strong>responsibilities(:, j)</strong>: Selects the j-th column of responsibilities, which contains the responsibility values (probabilities) of each data point for cluster j.
    - <strong>mean(responsibilities(:, j))</strong>: Averages the responsibilities for cluster j. This gives the proportion of the data points assigned to that cluster, which becomes the updated mixing coefficient phi(j)

- <strong> mu(j, :) = sum(responsibilities(:, j) .* Dataset) / sum(responsibilities(:, j)); </strong> 

    This calculates the mean for cluster j
    - <strong>mu(j, :)</strong>: Refers to the j-th row of the mu matrix, which represents the mean vector for cluster j. The : selects all columns, so mu(j, :) is a 1D vector (a row vector) containing the mean values for each dimension in cluster j.
    - <strong>responsibilities(:, j) .* Dataset</strong>: This is an element-wise multiplication, where each data point in Dataset is weighted by its responsibility for cluster j. This results in a weighted dataset, where points more strongly associated with cluster j contribute more to the mean.
    - <strong>sum(responsibilities(:, j) .* Dataset)</strong>: Sums the weighted data points for cluster j across all data points, producing a 1 x d vector.
    - <strong>/ sum(responsibilities(:, j))</strong>: Divides by the sum of the responsibilities for cluster j to calculate the weighted average, giving the new mean vector for cluster j.


- <strong> diff = Dataset - mu(j, :); </strong>

    This calculates the difference between each data point and the mean of cluster j.
    - <strong>Dataset</strong>: An n x d matrix of data points.
    - <strong>mu(j, :)</strong>: A 1 x d vector representing the mean of cluster j, which MATLAB automatically expands to n x d to match the dimensions of Dataset.
    - <strong>diff</strong>: An n x d matrix where each row represents the difference between a data point and the cluster mean.
    
    For example if we have:

    ```matlab
    Dataset = [
        2, 3;
        4, 5;
        6, 7
    ];  % 3x2 matrix

    mu(j, :) = [3, 4];  % 1x2 vector
    ```

    Then diff = Dataset - mu(j, :); produces:

    ```matlab
    diff = [
        -1, -1;  % [2, 3] - [3, 4]
        1,  1;  % [4, 5] - [3, 4]
        3,  3   % [6, 7] - [3, 4]
    ];  % 3x2 matrix
    ```

    This diff matrix, now n x d, represents the deviation of each data point from the cluster mean in each dimension. These deviations are then used to calculate the covariance matrix for the cluster in the M-step.

- <strong> sigma(:, :, j) = (responsibilities(:, j) .* diff)' * diff / sum(responsibilities(:, j)); </strong> 

    This calculates the covariance matrix of cluster j.
    - <strong>(responsibilities(:, j) .* diff)' * diff</strong>: Multiplies each row of diff by the responsibility for cluster j, weighting the deviation of each data point based on its likelihood of belonging to cluster j. Then ' transposes the weighted diff matrix to d x n.
    - <strong>\* diff</strong>: Multiplies d x n by n x d, resulting in a d x d matrix that represents the weighted outer product of diff, which calculates the covariance for cluster j.
    - <strong>/ sum(responsibilities(:, j))</strong>: Divides by the sum of responsibilities for cluster j to normalize the covariance matrix. This gives the average weighted covariance matrix for cluster j, taking into account the spread of data points around the mean.

### Step 3.C: EM Algorithm

Now that we have our E-step and M-step we can implement our EM algorithm.

```matlab
%% EM Algorithm
max_iters = 100;  % Maximum number of iterations
tolerance = 1e-4; % Convergence threshold

for iter = 1:max_iters
    % E-step: Calculate responsibilities
    responsibilities = expectation(Dataset, mu, sigma, phi, k);
    
    % M-step: Update parameters based on responsibilities
    prev_mu = mu; % Store previous means for convergence check
    [mu, sigma, phi] = maximisation(Dataset, responsibilities, k);

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

- <strong> responsibilities = expectation(Dataset, mu, sigma, phi, k); </strong>

    Calls the expectation function to calculate the responsibilities for each data point with respect to each Gaussian component.

- <strong> prev_mu = mu; </strong>

    Stores the current means (mu) in prev_mu so that we can check for convergence after updating the parameters.

- <strong> [mu, sigma, phi] = maximisation(Dataset, responsibilities, k); </strong>

    Calls the maximisation function to update the parameters based on the current responsibilities.

- <strong> if max(abs(mu - prev_mu)) < tolerance </strong>

    This checks if the maximum change in the means (mu) between the current and previous iteration is less than the specified tolerance. If this maximum change is smaller than tolerance, the algorithm is considered to have converged.
    - <strong>mu - prev_mu</strong> : calculates the difference between the updated and previous means.
    - <strong>abs(mu - prev_mu)</strong> : takes the absolute value of these differences.
    - <strong>max(...)</strong> : finds the largest difference.

- <strong> fprintf('Converged after %d iterations\n', iter); </strong>

    If convergence is reached, this line prints a message indicating the number of iterations it took to converge.

- <strong> break; </strong>

    This exits the loop early if convergence is detected.

## Step 4: Plot Results

Now that we have our converged GMM graph we can plot the data.

```matlab
% Assign each data point to the cluster with the highest responsibility
[~, cluster_assignments] = max(responsibilities, [], 2);

% Plot the generated data points for each cluster based on assignments
figure;
scatter(Dataset(cluster_assignments == 3, 1), Dataset(cluster_assignments == 3, 2), 10, 'r', 'filled'); % Species X in red
hold on;
scatter(Dataset(cluster_assignments == 1, 1), Dataset(cluster_assignments == 1, 2), 10, 'b', 'filled'); % Species Y in blue
hold on;
scatter(Dataset(cluster_assignments == 2, 1), Dataset(cluster_assignments == 2, 2), 10, 'g', 'filled'); % Species Z in green
hold on;


% Define grid for plotting Gaussian contours
x = linspace(min(Dataset(:,1))-5, max(Dataset(:,1))+5, 100);
y = linspace(min(Dataset(:,2))-5, max(Dataset(:,2))+5, 100);
[X, Y] = meshgrid(x, y);


% Calculate and plot Gaussian PDF contour for each cluster

% Species X
Z1 = mvnpdf([X(:) Y(:)], mu1, sigma1);
Z1 = reshape(Z1, size(X));
contour(X, Y, Z1, 'LineColor', 'r', 'LineWidth', 1.5);

% Species Y
Z2 = mvnpdf([X(:) Y(:)], mu2, sigma2);
Z2 = reshape(Z2, size(X));
contour(X, Y, Z2, 'LineColor', 'b', 'LineWidth', 1.5);

% Species Z
Z3 = mvnpdf([X(:) Y(:)], mu3, sigma3);
Z3 = reshape(Z3, size(X));
contour(X, Y, Z3, 'LineColor', 'g', 'LineWidth', 1.5);


% Add labels and title
title('Final Gaussian Mixture Model Clusters after EM Algorithm');
xlabel('Latitude');
ylabel('Longitude');
legend('Species X Data', 'Species Y Data', 'Species Z Data', 'Species X PDF', 'Species Y PDF', 'Species Z PDF');
hold off;
```

#### Code Explanation

- <strong> [~, cluster_assignments] = max(responsibilities, [], 2);</strong>

    - <strong>responsibilities</strong>: This is an N x k matrix, where N is the number of data points and k is the number of clusters. Each element responsibilities(i, j) represents the probability (or responsibility) that data point i belongs to cluster j.
    - <strong>max(responsibilities, [], 2)</strong>: This finds the maximum value along each row of responsibilities.
    - <strong>[~, cluster_assignments]</strong>: cluster_assignments is an N x 1 vector where each element cluster_assignments(i) contains the index of the cluster with the highest responsibility for data point i.

- <strong> scatter(Dataset(cluster_assignments == 3, 1), Dataset(cluster_assignments == 3, 2), 10, 'r', 'filled'); </strong>

    - <strong>Dataset(cluster_assignments == 3, 1)</strong>: Selects all data points in Dataset that have been assigned to Cluster 3 ( where cluster_assignments == 3) and extracts the first feature (dimension) for these data points.
    - <strong>Dataset(cluster_assignments == 3, 2)</strong>: Extracts the second feature for the same data points.
    - <strong>scatter(..., 'r', 'filled')</strong>: plots these points in red with filled markers, representing Species X.

Here are the final results of the converged GMM.

![GMM Results](assets/clustering-tutorial/gmm2d-fig2.jpg)