---
title: KMeans Matlab Tutorial
date: 2024-11-02 18:00:00 +0000
categories: [Machine Learning, Matlab Tutorial]
tags: [Machine Learning, Clustering, K-Means, Matlab Tutorial, ]
math: true
---

# Introduction

Welcome to the K-Means Matlab Tutorial blog. In this blog we will break down how to apply the K-Means algorithm in Matlab using toolboxes and not using toolboxes. This blog aims to add a practical perspective on the K-Means algorithm following the theory blog on Exploring Clustering Methods. This blog will explain all code and confusing jargon, however, if you need a deeper understanding before delving into the practical applications, read the blog on Exploring Clustering Methods. Lets begin with the task we will be covering. 

# Task Overview

You are a data analyst working for a real estate company. Your team wants to understand the neighbourhood clusters based on housing characteristics to better target marketing efforts for specific types of properties. The goal is to segment neighbourhoods into clusters based on average property prices and proximity to city amenities (like parks, schools, and public transport).

Using K-Means clustering, segment these neighbourhoods into clusters to help the marketing team understand the types of neighbourhoods the company should target for different types of properties.

For this task we will be generating our own dataset to enhance our MATLAB capabilities, however, in a real-world scenario you would use the dataset provided. Lets start the task without using MATLAB toolboxes.

# Step-by-Step Process Without MATLAB Toolbox

## Step 1: Generate Dataset

### Step 1.A: Define Features

As the task asks we want to create a dataset with two key features for each neighbourhood:

- Average Property Price: The average cost of housing in a neighbourhood.
- Proximity to Amenities: A score representing how close a neighbourhood is to key amenities like schools, parks, and public transport. This score ranges from 1 to 10, with higher values indicating a better proximity to amenities.

Given this is a fictitious task lets assume there are 2000 neighbourhoods

### Step 1.B: Split Data into Two Groups

To make the dataset realistic, we will divide it into two groups that mimic real-world clusters. Each group will have different average property prices and proximity scores to reflect varying neighbourhood types.

Group 1: Lower-Priced Neighbourhoods with Lower Proximity Scores
- Number of Neighbourhoods: 1000
- Average Property Price:
    - Mean: £150,000
    - Standard Deviation: £30,000
- Proximity to Amenities:
    - Mean: 3 (on a scale from 1 to 10)
    - Standard Deviation: 1

This group represents neighbourhoods where property prices are more affordable, and proximity to amenities is lower. These might be suburban or rural areas where the housing is cheaper and further away from major amenities.

Group 2: Higher-Priced Neighbourhoods with Higher Proximity Scores
- Number of Neighbourhoods: 1000
- Average Property Price:
    - Mean: £300,000
    - Standard Deviation: £50,000
- Proximity to Amenities:
    - Mean: 7 (on a scale from 1 to 10)
    - Standard Deviation: 1.5

This group represents neighbourhoods where property prices are higher, and proximity to amenities is better. These could be urban areas or more desirable suburbs where housing is costlier but closer to facilities.

### Step 1.C: Generate the Data

In MATLAB, we can use the randn function to create normally distributed random values around a given mean with a specified standard deviation. Here’s how we generate data for each group:

#### Generate Group 1 

- Using randn to create an array of 1000 points for each feature 
- Scale the random values by the standard deviation and add the mean

```matlab
%Group 1
mean_price1 = 150000; % Mean property price for Group 1
std_price1 = 30000; % Standard deviation for property price
prices1 = mean_price1 + (std_price1 * randn(1000, 1)); % Generate 1000 property prices

mean_amenity1 = 3; % Mean proximity score for Group 1
std_amenity1 = 1.0; % Standard deviation for proximity
amenities1 = mean_amenity1 + std_amenity1 * randn(1000, 1); % Generate 1000 proximity scores
```

#### Generate Group 2 

- Using randn to create an array of 1000 points for each feature 
- Scale the random values by the standard deviation and add the mean

```matlab
%Group 2
mean_price2 = 300000; % Mean property price for Group 2
std_price2 = 50000; % Standard deviation for property price
prices2 = mean_price2 + (std_price2 * randn(1000, 1)); % Generate 1000 property prices

mean_amenity2 = 7; % Mean proximity score for Group 1
std_amenity2 = 1.5; % Standard deviation for proximity
amenities2 = mean_amenity2 + std_amenity2 * randn(1000, 1); % Generate 1000 proximity scores
```

### Step 1.D: Combine Both Groups into One Dataset

We now have two groups: Group 1 with prices1 and amenities1 and Group2 with prices2 and amenities2. To create the final dataset lets concatenate the data from both groups:

```matlab
% Combine both groups into one dataset
amenities1 = min(max(amenities1, 0), 10); % Ensure amenities1 values are within [0, 10]
amenities2 = min(max(amenities2, 0), 10); % Ensure amenities2 values are within [0, 10]
amenities = [amenities1; amenities2];
prices = [prices1; prices2];
Dataset = [prices, amenities];
```

Now we have our generated dataset lets plot the data to visualise our two groups:

```matlab
% Plot Group 1 and Group 2 with different colours
figure;
scatter(prices1, amenities1, 'r'); % Group 1 in red
hold on;
scatter(prices2, amenities2, 'b'); % Group 2 in blue
title('Generated Neighbourhood Dataset');
xlabel('Average Property Price');
ylabel('Proximity to Amenities');
legend('Lower-priced neighbourhoods', 'Higher-priced neighbourhoods');
hold off;
```

### Step 1.E: Results

Here is the entire code to generate and plot the dataset:

```matlab
% Group 1 (Lower-priced neighbourhoods)
mean_price1 = 150000; % Mean property price
std_price1 = 30000; % Standard deviation for property price
prices1 = mean_price1 + std_price1 * randn(1000, 1); % Generate 1000 property prices

mean_amenity1 = 3; % Mean proximity score
std_amenity1 = 1.0; % Standard deviation for proximity
amenities1 = mean_amenity1 + std_amenity1 * randn(1000, 1); % Generate 1000 proximity scores

% Group 2 (Higher-priced neighbourhoods)
mean_price2 = 300000; % Mean property price
std_price2 = 50000; % Standard deviation for property price
prices2 = mean_price2 + std_price2 * randn(1000, 1); % Generate 1000 property prices

mean_amenity2 = 7; % Mean proximity score
std_amenity2 = 1.5; % Standard deviation for proximity
amenities2 = mean_amenity2 + std_amenity2 * randn(1000, 1); % Generate 1000 proximity scores

% Combine both groups into one dataset
amenities1 = min(max(amenities1, 0), 10); % Ensure amenities1 values are within [0, 10]
amenities2 = min(max(amenities2, 0), 10); % Ensure amenities2 values are within [0, 10]
amenities = [amenities1; amenities2];
prices = [prices1; prices2];
Dataset = [prices, amenities];

% Plot the two groups
figure;
scatter(prices1, amenities1, 'r'); % Group 1 in red
hold on;
scatter(prices2, amenities2, 'b'); % Group 2 in blue
title('Generated Neighbourhood Dataset');
xlabel('Average Property Price');
ylabel('Proximity to Amenities');
legend('Lower-priced neighbourhoods', 'Higher-priced neighbourhoods');
hold off;
```

![KMeans generated dataset](assets/kmeans-fig1.jpg)

## Step 2: Implement K-Means Algorithm

### Step 2.A: Define the Parameters

Lets now define the parameters for our K-Means algorithm.

```matlab
k = 2;  % Number of clusters we want to identify
max_iterations = 100; % Maximum number of iterations to allow

[num_points, num_features] = size(Dataset); % Get the number of points and features in the dataset

% Randomly select k data points from the dataset as the initial centroids
centroids = Dataset(randperm(num_points, k), :);
```

Where:
- <strong> k </strong>: Number of clusters (2 clusters for the two neighbourhood types)
- <strong> max_iterations </strong>: The maximum number of iterations to avoid infinite loops if the algorithm doesnt converge.
- <strong> [num_points, num_features] = size(Dataset) </strong>: This line extracts the number of rows and columns in the matrix dataset and assigns them to two variables num_points and num_features.
- <strong> centroids = Dataset(randperm(num_points, k), :) </strong>: This line selects k random rows from the dataset to serve as the initial centroids for the K_Means clustering algorithm. Lets break it down further.

#### Line "centroids = Dataset(randperm(num_points, k), :)" Breakdown
- randperm(num_points, k): randperm is a function that generates k random row indicies from num_points. For example, if num_points = 2000 and k = 2, randperm creates a randomly shuffled list of all integers from 1 to 2000 and then pick the first 2 numbers from the list. For instance, it may produce:

```matlab
[876, 1537, 125, 1892, 543, ..., 45]
```

So the values taken from this function in this instance would be [876, 1537].

- <strong> Dataset(...) </strong>: Using these randomly chosen values [876, 1537], we select these specific rows from the Dataset.
- The colon <strong> : </strong> means select all columns. So, Dataset(randperm(num_points, k), :) takes all columns for each chosen row.
- <strong> centroids </strong> : If Dataset has 2000 rows and 2 columns (e.g., price and amenities), then Dataset(randperm(num_points, k), :) will return a k x 2 matrix. Given k = 2, centroids is a 2 x 2 matrix.

### Step 2.B: Create the loop

First we create a loop that iterates up until a max iteration count (max_iterations).

```matlab
for iter = 1:max_iterations
    % Our k-Means algorithm will go in here
end
```
Where:
- <strong> iter </strong> is the loop counter

### Step 2.C: Euclidean Distance

Next we want to find the pairwise Euclidean distances between each data point and each centroid.

```matlab
    distances = pdist2(Dataset, centroids);
```

Where:
- <strong> distances </strong> is a matrix where each row corresponds to a data point and each column corresponds to a centroid.
- <strong> pdist2 </strong> is a function that calculates the pairwise Euclidean distances between each point in Dataset and each centroid.
- <strong> Dataset </strong> is a matrix that contains 2000 rows (data points) and 2 columns (features: price and proximity).
- <strong> centroids </strong> is a matrix with k rows (number of clusters) and 2 columns (features for each centoid).

### Step 2.C: Find the Closest Centroid

After finding all the Euclidean distances between each data point and centroid, we want to find which centroid is closest to each data point.

```matlab
    [~, cluster_assignments] = min(distances, [], 2);
```

Where:
- <strong> min(distances, [], 2) </strong> : Finds the minimum value along each row of the distances matrix. 

Since distances is a 2000 x 2 matrix where each row represents a data point and each column represents the distance to the centroid, this line goes through each row and finds the smallest value i.e. the cloest centroid. [] is an empty placeholder. 2 is the dimension argument. Its just the syntax for the min function.

- <strong> [~, cluster_assignments] </strong> : Used to store the index of the minimum value.

When min is called with two output arguments, it returns the value and the index. For example:

```matlab
[value, index] = min([3, 5, 2]);
```

Here, value would be 2 (the minimum), and index would be 3 (the position of 2 in the array).

Since we dont need the actual distance between the data point and the centroids we can ignore the value by using <strong> ~ </strong> and continue to store the closest cluster.

### Step 2.C: Centroid Update

Now that we have the new closest centroids we need to update the centroids by finding the mean of each feature.

```matlab
new_centroids = arrayfun(@(i) mean(Dataset(cluster_assignments == i, :), 1), 1:k, 'UniformOutput', false);
new_centroids = cell2mat(new_centroids');
```

Where:
- <strong> arrayfun </strong> applies a function to each element of an array, with the syntax B = arrayfun(func,A). In this case it is applying the anonymous function <strong> @(i) ... </strong>  to each cluster index from 1 to k.

- <strong> @(i) </strong> is an anonymous function that can accept multiple inputs and return one output. Here takes i (the cluster index) as an input (which comes from the array 1:k specified in the arrayfun function) and outputs the mean of all data points assigned to cluster i.

- <strong> Dataset(cluster_assignments == i, :) </strong> selects all rows in Dataset where cluster_assignments == i.

- <strong> mean(..., 1) </strong> calculates the mean of the selected rows along dimension 1 (columns), which gives the average values for each feature (e.g., property price and proximity). The result is a single row vector representing the mean position (centroid) for cluster i.

- <strong> 'UniformOutput', false </strong> instructs MATLAB to store each output in a cell array, which can handle non-scalar outputs because the output is a row vector and not a scalar (single number), MATLAB cannot store it in a regular array without some adjustments.

- <strong> cell2mat(new_centroids') </strong> converts the transposed cell array into a regular k x num_features matrix.

#### Code Walkthrough

Suppose k = 2, so 1:k is [1, 2].
<strong> arrayfun </strong> calls the anonymous function <strong> @(i) mean(Dataset(cluster_assignments == i, :), 1) </strong> twice.

- First call (i = 1): arrayfun passes i = 1 into the function
    - <strong> Dataset(cluster_assignments == 1, :) </strong> selects all data points assigned to cluster 1.
    - <strong> mean(..., 1) </strong> calculates the mean of these points along each feature (columns), giving the new centroid for cluster 1.

- Second call (i = 2): arrayfun passes i = 2 into the function
    - <strong> Dataset(cluster_assignments == 2, :) </strong> selects all data points assigned to cluster 2.
    - <strong> mean(..., 1) </strong> calculates the mean of these points along each feature (columns), giving the new centroid for cluster 2.

After both calls, arrayfun collects the results into a cell array where each cell contains the centroid for one cluster. Finally, cell2mat converts this cell array into a regular matrix of centroids.

### Step 2.D: Convergence Check

Now that we have our new centroids we need to check whether the centroids have stopped moving between iterations. In K-Means clustering, this is a signal that the algorithm has converged.

```matlab
if isequal(centroids, new_centroids)
    break;
end

centroids = new_centroids;
```

Where:
- <strong> isequal </strong> is a function that checks if two arrays are exactly the same. It returns true if the arrays are identical and false if not. Here it checks whether the centroids have converged and will exit the loop if they have by using <strong> break </strong>.

- <strong> centroids = new_centroids </strong> updates the current centroids to the values in new_centroids. If the centroids have not converged this line assigns the newly calculated centroids for the next iteration.


### Step 2.E: K-Means Algorithm

Here is the entire code of the K-Means algorithm:

```matlab
% Step 3: Implement K-Means algorithm
for iter = 1:max_iterations
    distances = pdist2(Dataset, centroids);
    [~, cluster_assignments] = min(distances, [], 2);

    new_centroids = arrayfun(@(i) mean(Dataset(cluster_assignments == i, :), 1), 1:k, 'UniformOutput', false);
    new_centroids = cell2mat(new_centroids');

    if isequal(centroids, new_centroids)
        break;
    end

    centroids = new_centroids;
end
```

### Step 3: Visualise the Results

Now that we have our K-Means algorithm we want to visualise our results. To do this we create a graph as we did before with the generated data set.

```matlab
figure;
gscatter(Dataset(:,1), Dataset(:,2), cluster_assignments, 'rb', '.', 20); % Plot points by cluster
hold on;
scatter(centroids(:,1), centroids(:,2), 100, 'kx', 'LineWidth', 2); % Plot centroids as large black X's
title('K-Means Clustering Results on Neighbourhood Data');
xlabel('Average Property Price');
ylabel('Proximity to Amenities');
legend('Cluster 1', 'Cluster 2', 'Centroids');
hold off;
```

Where: 
- <strong> figure </strong> : This creates a new figure window in MATLAB
- <strong> gscatter(Dataset(:,1), Dataset(:,2), cluster_assignments, 'rb', '.', 20); </strong>
    - gscatter function: Creates a scatter plot where data points are grouped and coloured by category (cluster assignments, in this case).
    - Dataset(:,1): Selects all rows in the first column of Dataset, representing the x-coordinates of the data points.
    - Dataset(:,2): Selects all rows in the second column of Dataset, representing the y-coordinates of the data points.
    - cluster_assignments: A vector that specifies the cluster assignment (1 or 2) for each data point in Dataset. MATLAB uses this vector to assign different colours to each cluster.
    - 'rb': Specifies the colours for each cluster, with r for red and b for blue. The first cluster will be red, and the second cluster will be blue.
    - '.': Specifies the marker style. '.' uses dot markers for the data points.
    - '20': Specifies the dot thickness of each data point.
- <strong> hold on </strong> :  This command keeps the existing plot (data points) so that we can add the centroids without erasing the current data points.
- <strong> scatter(centroids(:,1), centroids(:,2), 100, 'kx', 'LineWidth', 2); </strong>
    - scatter function: Creates a scatter plot for the centroids with more customisation.
    - centroids(:,1): Selects all rows in the first column of centroids, representing the x-coordinates of the centroids.
    - centroids(:,2): Selects all rows in the second column of centroids, representing the y-coordinates of the centroids.
    - 100: Specifies the marker size. Here, the centroids are displayed as larger points to stand out.
    - 'kx': Specifies the colour and marker type for the centroids. k means black, x means cross markers.
    - 'LineWidth', 2: Sets the line width of the markers to 2, making the X’s bolder and more visible.
- <strong> title('K-Means Clustering Results on Neighbourhood Data'); </strong> : Sets the title for the plot.
- <strong> xlabel('Average Property Price'); </strong> : Adds a label to the x-axis of the plot.
- <strong> ylabel('Proximity to Amenities'); </strong> : Adds a label to the y-axis of the plot.
- <strong> legend('Cluster 1', 'Cluster 2', 'Centroids'); </strong> : Adds a legend to the plot
- <strong> hold off </strong> : Releases the current figure, allowing future plotting commands to replace the existing plot.

### Step 3.A: Results

![KMeans results](assets/kmeans-fig2.jpg)