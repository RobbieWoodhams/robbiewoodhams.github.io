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

![KMeans generated dataset](assets/Kmeans-fig1.jpg)

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
