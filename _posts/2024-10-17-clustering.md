---
title: Exploring Clustering Methods
date: 2024-10-17 18:00:00 +0000
categories: [Machine Learning]
tags: [Machine Learning, Clustering, Gaussian Mixture Model, K-Means, ]
math: true
---

# Introduction

Welcome to the blog on Clustering techniques used in unsupervised machine learning algorithms. In this blog, we will delve into unsupervised learning and its applications, exploring two common forms of clustering techniques: K-means clustering and Gaussian Mixture Model (GMM) clustering. Furthermore, we will look into various metrics for evaluating cluster performance. Let us begin with a background on unsupervised learning.

# Unsupervised Learning

## What is Unsupervised Learning

IBM defines unsupervised learning as a machine learning algorithm that analyses and clusters unlabelled data sets by discovering hidden patterns or data groupings within the data without requiring human intervention. In simpler terms, the algorithm follows a set of rules to examine and classify data into different groups or clusters, even though the data has no predefined categories. 

### Example: Customer Identification

Imagine a company with a large dataset of customer purchasing behaviours but no information about the type of customer each individual is. Using unsupervised learning, the algorithm could analyse the data and automatically group customers based on their spending habits, identifying high spenders, occasional buyers, and bargain shoppers without needing prior labels or categories.

## What is Unsupervised Learning Used For?

Given the ability to identify patterns in large volumes of data quickly unsupervised learning algorithms have found many real-world applications. Some examples include:

- Market Segmentation: As mentioned previously unsupervised learning algorithms are used to identify distinct groups of customers based on their purchasing behaviour, demographics, or other characteristics. 

- Anomaly Detection: Unsupervised learning algorithms are used for anomaly detection in various domains including cybersecurity, fraud detection, and equipment maintenance. These algorithms can identify unusual patterns or outliers in data that deviate significantly from normal behaviour, helping to detect fraudulent transactions, security breaches, or equipment failures.

- Recommendation Systems: Unsupervised learning algorithms are also used in recommendation systems to provide personalised content and recommendations for users based on user behaviour and preferences. 

## What is Clustering?

Unsupervised learning models are used for three main tasks - clustering, association, and dimensionality reduction. In this blog, we will be focusing on clustering.  

Clustering is a data mining technique which groups unlabelled data based on their similarities or differences. Clustering algorithms are used to process raw, unclassified data objects into groups represented by structures or patterns in the information. Let’s explore the common methods.

# K-Means Clustering

## What is K-Means Clustering?

K-Means is a hard clustering algorithm, meaning each data point belongs to exactly one cluster. Data points are assigned into K groups, where K represents the number of clusters based on the distance from each group's centroid.  The data points closest to a given centroid will be clustered under the same category. A larger K value will be indicative of smaller groupings with more granularity whereas a smaller K value will have larger groupings and less granularity. 

K-Means is revolved around the Euclidean Distance formula, which measures the distance between two points $$P(x_1, y_1)$$ and $$Q(x_2, y_2)$$ and is given by:

$$d(P, Q) = \sqrt{\sum_{i=1}^{n} (q_i - p_i)^2}$$

Where:
- $$d(P, Q)$$ is the distance between two points
- $$\sum_{i=1}^{n}$$ is the sum of all Euclidean vectors squared
- $$(q_i - p_i)^2$$ are the Euclidean vectors (coordinates, features, etc)

Let us go through the step-by-step process of using K-Means with an example.

## Example: Customer Segmentation

Suppose we have a retail company's customer dataset with the following features for each customer:

- Age: The age of the customer
- Annual Income (in thousands): The customer's yearly income
- Spending Score (1 - 100): A score assigned based on customer behaviour and engagement, where 100 means a high spender and 1 means a low spender.

Our goal is to use K-Means clustering to segment customers into groups based on these features.

### Dataset Example

| Customer | Age | Annual Income (£k) | Spending Score |
|----------|----------|----------|----------|
| 1   | 25   | 40   | 75 |
| 2   | 34   | 85   | 50 |
| 3   | 22   | 60   | 95 |
| 4   | 45   | 30   | 40 |
| 5   | 50   | 200  | 30 |
| 6   | 31   | 150  | 60 |
| 7   | 23   | 45  | 88 |
| 8   | 37   | 90  | 20 |
| 9   | 29   | 70  | 77 |
| 10   | 43   | 120  | 10 |

Lets aim to cluster these customers into three groups (K = 3) using K-Means clustering. K = 3 was chosen because in this scenario grouping customers into three categories reflect typical real-world segmentation for example a low spender, a medium spender, and a high spender. There are techniques for selecting the number of clusters in K-Means such as the Silhouette Score and the Elbow method but we will not be covering them in this blog.

### Step 1: Initialise K and Select Initial Centroids

Since we have chosen to cluster the customers into three groups (K = 3) we will randomly initialise three centroids by picking any three customers. Lets arbitrarily choose:

- Centroid 1: Customer 1 (25, 40, 75)
- Centroid 2: Customer 5 (50, 200, 30)
- Centroid 3: Customer 3 (22, 60, 95)

### Step 2: Calculate the Distance to Centroids

Now we use the Euclidean distance formula to find the Euclidean distance between each customer and all three centroids, using all three features (age, annual income, spending score)

The formula for Euclidean distance in a 3D space is:

$$d(P, Q) = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2 + (z_2 - z_1)^2}$$

Where x, y, and z are the three features.

Let us calculate the distance from the customers to each centroid.

#### Customer 1 (25, 40, 75)

- Distance to Centroid 1 (25, 40, 75)

$$d = \sqrt{(25 - 25)^2 + (40 - 40)^2 + (75 - 75)^2} = 0$$

- Distance to Centroid 2 (50, 200, 30)

$$d = \sqrt{(25 - 50)^2 + (40 - 200)^2 + (75 - 30)^2}$$

$$d = \sqrt{(-25)^2 + (-160)^2 + (45)^2}$$

$$d = \sqrt{625 + 25600 + 2025}$$

$$d = \sqrt{28250}$$

$$d \approx 168.08$$

- Distance to Centroid 3 (22, 60, 95)

$$d = \sqrt{(25 - 22)^2 + (40 - 60)^2 + (75 - 95)^2}$$

$$d = \sqrt{(3)^2 + (-20)^2 + (-20)^2}$$

$$d = \sqrt{9 + 400 + 400}$$

$$d = \sqrt{809}$$

$$d \approx 28.44$$

#### Customer 2 (34, 85, 50)

- Distance to Centroid 1 (25, 40, 75)

$$d = \sqrt{(34 - 25)^2 + (85 - 40)^2 + (50 - 75)^2}$$

$$d = \sqrt{(9)^2 + (45)^2 + (-25)^2}$$

$$d = \sqrt{81 + 2025 + 625}$$

$$d = \sqrt{2731}$$

$$d \approx 52.26$$

- Distance to Centroid 2 (50, 200, 30)

$$d = \sqrt{(34 - 50)^2 + (85 - 200)^2 + (50 - 30)^2}$$

$$d = \sqrt{(-16)^2 + (-115)^2 + (20)^2}$$

$$d = \sqrt{256 + 13225 + 400}$$

$$d = \sqrt{13881}$$

$$d \approx 117.83$$

- Distance to Centroid 3 (22, 60, 95)

$$d = \sqrt{(34 - 22)^2 + (85 - 60)^2 + (50 - 95)^2}$$

$$d = \sqrt{(12)^2 + (25)^2 + (-45)^2}$$

$$d = \sqrt{144 + 625 + 2025}$$

$$d = \sqrt{2794}$$

$$d \approx 52.87$$

#### Customer 3 (22, 60, 95)

- Distance to Centroid 1 (25, 40, 75)

$$d = \sqrt{(22 - 25)^2 + (60 - 40)^2 + (95 - 75)^2}$$

$$d = \sqrt{(-3)^2 + (20)^2 + (20)^2}$$

$$d = \sqrt{9 + 400 + 400}$$

$$d = \sqrt{809}$$

$$d \approx 28.44$$

- Distance to Centroid 2 (50, 200, 30)

$$d = \sqrt{(22 - 50)^2 + (60 - 200)^2 + (95 - 30)^2}$$

$$d = \sqrt{(-28)^2 + (-140)^2 + (65)^2}$$

$$d = \sqrt{784 + 19600 + 4225}$$

$$d = \sqrt{13881}$$

$$d \approx 156.87$$

- Distance to Centroid 3 (22, 60, 95)

$$d = \sqrt{(22 - 22)^2 + (60 - 60)^2 + (95 - 95)^2} = 0$$

#### Customer 4 (45, 30, 40)

- Distance to Centroid 1 (25, 40, 75)

$$d = \sqrt{(45 - 25)^2 + (30 - 40)^2 + (40 - 75)^2}$$

$$d = \sqrt{(20)^2 + (-10)^2 + (-35)^2}$$

$$d = \sqrt{400 + 100 + 1225}$$

$$d = \sqrt{1725}$$

$$d \approx 41.53$$

- Distance to Centroid 2 (50, 200, 30)

$$d = \sqrt{(45 - 50)^2 + (30 - 200)^2 + (40 - 30)^2}$$

$$d = \sqrt{(-5)^2 + (-170)^2 + (10)^2}$$

$$d = \sqrt{25 + 28900 + 100}$$

$$d = \sqrt{29025}$$

$$d \approx 170.37$$

- Distance to Centroid 3 (22, 60, 95)

$$d = \sqrt{(45 - 22)^2 + (30 - 60)^2 + (40 - 95)^2}$$

$$d = \sqrt{(23)^2 + (-30)^2 + (-55)^2}$$

$$d = \sqrt{529 + 900 + 3025}$$

$$d = \sqrt{4454}$$

$$d \approx 66.74$$

#### Customer 5 (50, 200, 30)

- Distance to Centroid 1 (25, 40, 75)

$$d = \sqrt{(50 - 25)^2 + (200 - 40)^2 + (30 - 75)^2}$$

$$d = \sqrt{(25)^2 + (160)^2 + (-45)^2}$$

$$d = \sqrt{625 + 25600 + 2025}$$

$$d = \sqrt{28250}$$

$$d \approx 168.08$$

- Distance to Centroid 2 (50, 200, 30)

$$d = \sqrt{(50 - 50)^2 + (200 - 200)^2 + (30 - 30)^2} = 0$$

- Distance to Centroid 3 (22, 60, 95)

$$d = \sqrt{(50 - 22)^2 + (200 - 60)^2 + (30 - 95)^2}$$

$$d = \sqrt{(28)^2 + (140)^2 + (-65)^2}$$

$$d = \sqrt{784 + 19600 + 4225}$$

$$d = \sqrt{24609}$$

$$d \approx 156.87$$

#### Customer 6 (31, 150, 60)

- Distance to Centroid 1 (25, 40, 75)

$$d = \sqrt{(31 - 25)^2 + (150 - 40)^2 + (60 - 75)^2}$$

$$d = \sqrt{(5)^2 + (110)^2 + (-15)^2}$$

$$d = \sqrt{25 + 12100 + 225}$$

$$d = \sqrt{12350}$$

$$d \approx 111.13$$

- Distance to Centroid 2 (50, 200, 30)

$$d = \sqrt{(31 - 50)^2 + (150 - 200)^2 + (60 - 30)^2}$$

$$d = \sqrt{(-19)^2 + (-50)^2 + (30)^2}$$

$$d = \sqrt{361 + 2500 + 900}$$

$$d = \sqrt{3761}$$

$$d \approx 61.33$$

- Distance to Centroid 3 (22, 60, 95)

$$d = \sqrt{(31 - 22)^2 + (150 - 60)^2 + (60 - 95)^2}$$

$$d = \sqrt{(9)^2 + (90)^2 + (-35)^2}$$

$$d = \sqrt{81 + 8100 + 1225}$$

$$d = \sqrt{9406}$$

$$d \approx 96.98$$

#### Customer 7 (23, 45, 88)

- Distance to Centroid 1 (25, 40, 75)

$$d = \sqrt{(23 - 25)^2 + (45 - 40)^2 + (88 - 75)^2}$$

$$d = \sqrt{(-2)^2 + (5)^2 + (13)^2}$$

$$d = \sqrt{4 + 25 + 169}$$

$$d = \sqrt{198}$$

$$d \approx 14.07$$

- Distance to Centroid 2 (50, 200, 30)

$$d = \sqrt{(23 - 50)^2 + (45 - 200)^2 + (88 - 30)^2}$$

$$d = \sqrt{(-27)^2 + (-155)^2 + (58)^2}$$

$$d = \sqrt{729 + 24025 + 3364}$$

$$d = \sqrt{28118}$$

$$d \approx 167.68$$

- Distance to Centroid 3 (22, 60, 95)

$$d = \sqrt{(23 - 22)^2 + (45 - 60)^2 + (88 - 95)^2}$$

$$d = \sqrt{(1)^2 + (-15)^2 + (-7)^2}$$

$$d = \sqrt{1 + 225 + 49}$$

$$d = \sqrt{270}$$

$$d \approx 16.43$$

#### Customer 8 (37, 90, 20)

- Distance to Centroid 1 (25, 40, 75)

$$d = \sqrt{(37 - 25)^2 + (90 - 40)^2 + (20 - 75)^2}$$

$$d = \sqrt{(12)^2 + (50)^2 + (-55)^2}$$

$$d = \sqrt{144 + 2500 + 3025}$$

$$d = \sqrt{5669}$$

$$d \approx 75.29$$

- Distance to Centroid 2 (50, 200, 30)

$$d = \sqrt{(37 - 50)^2 + (90 - 200)^2 + (20 - 30)^2}$$

$$d = \sqrt{(-13)^2 + (-110)^2 + (-10)^2}$$

$$d = \sqrt{169 + 12100 + 100}$$

$$d = \sqrt{12369}$$

$$d \approx 111.22$$

- Distance to Centroid 3 (22, 60, 95)

$$d = \sqrt{(37 - 22)^2 + (90 - 60)^2 + (20 - 95)^2}$$

$$d = \sqrt{(15)^2 + (30)^2 + (75)^2}$$

$$d = \sqrt{225 + 900 + 5625}$$

$$d = \sqrt{6750}$$

$$d \approx 82.16$$

#### Customer 9 (29, 70, 77)

- Distance to Centroid 1 (25, 40, 75)

$$d = \sqrt{(29 - 25)^2 + (70 - 40)^2 + (77 - 75)^2}$$

$$d = \sqrt{(4)^2 + (30)^2 + (2)^2}$$

$$d = \sqrt{16 + 900 + 4}$$

$$d = \sqrt{920}$$

$$d \approx 30.33$$

- Distance to Centroid 2 (50, 200, 30)

$$d = \sqrt{(29 - 50)^2 + (70 - 200)^2 + (77 - 30)^2}$$

$$d = \sqrt{(-21)^2 + (-130)^2 + (47)^2}$$

$$d = \sqrt{441 + 16900 + 2209}$$

$$d = \sqrt{19550}$$

$$d \approx 139.82$$

- Distance to Centroid 3 (22, 60, 95)

$$d = \sqrt{(29 - 22)^2 + (70 - 60)^2 + (77 - 95)^2}$$

$$d = \sqrt{(7)^2 + (10)^2 + (18)^2}$$

$$d = \sqrt{49 + 100 + 324}$$

$$d = \sqrt{473}$$

$$d \approx 21.75$$

#### Customer 10 (43, 120, 10)

- Distance to Centroid 1 (25, 40, 75)

$$d = \sqrt{(43 - 25)^2 + (120 - 40)^2 + (10 - 75)^2}$$

$$d = \sqrt{(18)^2 + (80)^2 + (-65)^2}$$

$$d = \sqrt{3246 + 6400 + 4225}$$

$$d = \sqrt{13871}$$

$$d \approx 117.78$$

- Distance to Centroid 2 (50, 200, 30)

$$d = \sqrt{(43 - 50)^2 + (120 - 200)^2 + (10 - 30)^2}$$

$$d = \sqrt{(-7)^2 + (-80)^2 + (-20)^2}$$

$$d = \sqrt{49 + 6400 + 400}$$

$$d = \sqrt{6849}$$

$$d \approx 82.76$$

- Distance to Centroid 3 (22, 60, 95)

$$d = \sqrt{(43 - 22)^2 + (120 - 60)^2 + (10 - 95)^2}$$

$$d = \sqrt{(21)^2 + (60)^2 + (-85)^2}$$

$$d = \sqrt{441 + 3600 + 7225}$$

$$d = \sqrt{11266}$$

$$d \approx 106.14$$

### Step 3: Assign Each Customer to the Nearest Centroid

After calculating the distances, we assign each customer to the closest centroid based on the smallest distance.

- Customer 1 is closest to Centroid 1 
- Customer 2 is closest to Centroid 1 
- Customer 3 is closest to Centroid 3 
- Customer 4 is closest to Centroid 1 
- Customer 5 is closest to Centroid 2 
- Customer 6 is closest to Centroid 2
- Customer 7 is closest to Centroid 1 
- Customer 8 is closest to Centroid 1 
- Customer 9 is closest to Centroid 3 
- Customer 10 is closest to Centroid 2

We can now cluster the customers to each centroid:

- Cluster 1: Customers 1, 2, 4, 7, 8 (closer to centroid 1)
- Cluster 2: Customers 5, 6, 10 (closer to centroid 2)
- Cluster 3: Customers 3, 9 (closer to centroid 3)

### Step 4: Update Centroids

Now that we have the clusters we can update the centroids by finding the mean of each feature.

#### Cluster 1 (Customers 1, 2, 4, 7, 8)

- Mean age = $$ \frac{25, 34, 45, 23, 37}{5} = 32.8$$
- Mean income = $$ \frac{40, 85, 30, 45, 90}{5} = 58$$
- Mean spending score = $$ \frac{75, 50, 40, 88, 20}{5} = 54.6$$

Updated Centroid 1 becomes (32.8, 58, 54.6)

#### Cluster 2 (Customers 5, 6, 10)

- Mean age = $$ \frac{50, 31, 43}{3} = 41.33$$
- Mean income = $$ \frac{200, 150, 120}{3} = 156.67$$
- Mean spending score = $$ \frac{30, 60, 10}{3} = 33.33$$

Updated Centroid 2 becomes (41.33, 156.67, 33.33)

#### Cluster 3 (Customers 3, 9)

- Mean age = $$ \frac{22, 29}{2} = 25.5$$
- Mean income = $$ \frac{60, 70}{2} = 65$$
- Mean spending score = $$ \frac{95, 77}{2} = 86$$

Updated Centroid 3 becomes (25.5, 65, 86)

### Step 5: Repeat Until Convergence

We repeat the process of recalculating the distances to the updated centroids, reassigning customers to the nearest centroid, and updating the centroids. This process continues until the centroids no longer change, or a maximum number of iterations is reached.

### Final Clusters

After 3 iterations the centroids converge. here are the final cluster and centroid assignments:

- Cluster 1: Customers 2, 4, 8
- Cluster 2: Customers 5, 6, 10
- Cluster 3: Customers 1, 3, 7, 9

- Centroid 1: (38.67, 68.33, 36.67)
- Centroid 2: (41.33, 156.67, 33.33)
- Centroid 3: (24.75, 53.75, 83.75)


# Gaussian Mixture Model (GMM) Clustering

Gaussian Mixture Model (GMM) Clustering is a soft clustering algorithm, meaning that each data point is assigned to clusters based on the probability or likelihood that it belongs to each cluster, rather than assigning each data point to exactly one cluster like K-Means. GMM assumes that the data is generated from a mixture of several Gaussian distributions, each representing a cluster. Instead of assigning a data point to the nearest centroid, GMM assigns a probability to each point for belonging to each distribution (cluster).

The primary formula for GMM Clustering is the Gaussian PDF formula as seen below, however, GMM also contains various other formulas and parameters which I will explain as we follow through an example.

$$N(x \mid \mu, \Sigma) = \frac{1}{(2\pi)^{d/2} \lvert \Sigma \rvert^{1/2}} \exp\left( -\frac{1}{2} (x - \mu)^\top \Sigma^{-1} (x - \mu) \right)$$

Where:
- $$x$$ is the data point (a d-dimensional vector)
- $$\mu$$ is the mean vector (center of the Gaussian distribution)
- $$\Sigma$$ is the covariance matrix (describes the spread and shape of the distribution)
- $$\lvert \Sigma \rvert$$ is the determinent of the covariance matrix
- $$(x - \mu)^\top \Sigma^{-1} (x - \mu)$$ is the Mahalanobis distance between $$x$$ and the mean $$\mu$$

We will break down the formula and its parameters more as we go through the example.



## Example: Customer Segmentation

Let us use the same data set as we did with K-Means.

| Customer | Age | Annual Income (£k) | Spending Score |
|----------|----------|----------|----------|
| 1   | 25   | 40   | 75 |
| 2   | 34   | 85   | 50 |
| 3   | 22   | 60   | 95 |
| 4   | 45   | 30   | 40 |
| 5   | 50   | 200  | 30 |
| 6   | 31   | 150  | 60 |
| 7   | 23   | 45  | 88 |
| 8   | 37   | 90  | 20 |
| 9   | 29   | 70  | 77 |
| 10   | 43   | 120  | 10 |

Lets also cluster these customers into two groups (K = 2).

### Step 1: Initialisation of Parameters

Firstly, we need to initialise the means $$\mu_k$$, covariances $$\sigma_k$$, and weights $$\alpha_k$$ for each of the three Gaussian components (since K = 2). The initialisation provides a starting point for the iterative process.

#### Means $$(\mu_1, \mu_2)$$

The means represent the initial centres of the clusters. Like K-means, we can initialise the means by selecting some actual data points or we can use the K-Means centroids we figured out earlier. For this example lets choose two customers from different parts of the dataset to ensure a good spread.

- $$\mu_1 = (25, 40, 75)$$ (Customer 1)
- $$\mu_2 = (50, 200, 30)$$ (Customer 5)


These choices ensures that the means are spread far apart, representing distinct parts of the data space. Customer 1 represents lower income and higher spending, while Customer 5 represents high income and low spending.

#### Covariances $$(\sigma_1, \sigma_2)$$

The covariance matrix describes the spread of the Gaussian distribution around each mean. Initially, we assume that each feature is independent (no correlation between age, income, and spending score), so we start with a diagonal covariance matrix where the values on the diagonal represent the variance of each feature. As a result the Covariance Matrix Structure will look like this:

$$\Sigma = \begin{bmatrix} \sigma_{\text{Age}}^2 & 0 & 0 \\ 0 & \sigma_{\text{Income}}^2 & 0 \\ 0 & 0 & \sigma_{\text{Score}}^2 \end{bmatrix}$$

We can compute the variance for each feature using the formula:

$$\sigma^2 = \frac{1}{N} \Sigma(x_i - \mu)^2$$ 

Where:
- $$N$$ is the number of samples
- $$X_i$$ are the individual data points
- $$\mu$$ is the mean

Lets compute the variances for each feature using the dataset.

##### Age Variance:

$$\mu_{\text{Age}} = \frac{25 + 34 + 22 + 45 + 50 + 31 + 23 + 37 + 29 + 43}{10}$$

$$\mu_{\text{Age}} \approx 34 $$

$$\sigma_{\text{Age}}^2 = \frac{1}{10}[(25 - 34)^2 + (34 - 34)^2 + (22 - 34)^2 + (45 - 34)^2 + (50 - 34)^2 + (31 - 34)^2 + (23 - 34)^2 + (37 - 34)^2 + (29 - 34)^2 + (43 - 34)^2]$$ 

$$= \frac{1}{10}[81 + 0 + 144 + 121 + 256 + 9 + 121 + 9 + 25 + 81]$$ 

$$= \frac{1}{10}[847]$$ 

$$\sigma_{\text{Age}}^2 \approx 85$$ 

##### Income Variance:

$$\mu_{\text{Income}} = \frac{40 + 85 + 60 + 30 + 200 + 150 + 45 + 90 + 70 + 120}{10}$$

$$\mu_{\text{Income}} = 89 $$

$$\sigma_{\text{Income}}^2 = \frac{1}{10}[(40 - 89)^2 + (85 - 89)^2 + (60 - 89)^2 + (30 - 89)^2 + (200 - 89)^2 + (150 - 89)^2 + (45 - 89)^2 + (90 - 89)^2 + (70 - 89)^2 + (120 - 89)^2]$$ 

$$= \frac{1}{10}[2401 + 16 + 841 + 3481 + 12321 + 3721 + 1936 + 1 + 361 + 961]$$ 

$$= \frac{1}{10}[26040]$$ 

$$\sigma_{\text{Income}}^2 = 2604$$ 

##### Spending Score Variance:

$$\mu_{\text{Score}} = \frac{75 + 50 + 95 + 40 + 30 + 60 + 88 + 20 + 77 + 10}{10}$$

$$\mu_{\text{Score}} \approx 55 $$

$$\sigma_{\text{Score}}^2 = \frac{1}{10}[(75 - 55)^2 + (50 - 55)^2 + (95 - 55)^2 + (40 - 55)^2 + (30 - 55)^2 + (60 - 55)^2 + (88 - 55)^2 + (20 - 55)^2 + (77 - 55)^2 + (10 - 55)^2]$$ 

$$= \frac{1}{10}[400 + 25 + 1600 + 225 + 625 + 25 + 1089 + 1225 + 484 + 2025]$$ 

$$= \frac{1}{10}[7723]$$ 

$$\sigma_{\text{Score}}^2 \approx 772$$ 

We now have the Initial Covariances:

$$\Sigma_1 = \begin{bmatrix} 85 & 0 & 0 \\ 0 & 2604 & 0 \\ 0 & 0 & 772 \end{bmatrix}$$

$$\Sigma_2 = \begin{bmatrix} 85 & 0 & 0 \\ 0 & 2604 & 0 \\ 0 & 0 & 772 \end{bmatrix}$$

#### Weights $$(\alpha_1, \alpha_2)$$

We start with equal weights for each Gaussian component, meaning we assume each cluster initially represents 50% of the data.

$$\alpha_1 = \alpha_2 = 0.5$$


### Step 2: Expectation Step (E-Step)

The E-Step calculates the responsibility or probability that each point belongs to each cluster. It is calculated using this formula.

$$r_{nk} = \frac{\pi_k \mathcal{N}(x_n | \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \alpha_j \mathcal{N}(x_n | \mu_j, \Sigma_j)}$$

Where:
- $$r_{nk}$$ is the responsibility that component k takes for data point n 
- $$\pi_k$$ is the mixing coefficient for component k 
- $$\mathcal{N}(x_n | \mu_j, \Sigma_j)$$ is the Gaussian PDF for component k evaluated at data point $$X_n$$ with mean $$\mu_k$$ and covariance $$\Sigma_k$$
- $$\sum_{j=1}^{K}$$ represents the summation over all components j of the Gaussian Mixture Model (GMM)
- $$\alpha_j$$ is the weights

#### Gaussian PDF 

$$N(x \mid \mu, \Sigma) = \frac{1}{(2\pi)^{d/2} \lvert \Sigma \rvert^{1/2}} \exp\left( -\frac{1}{2} (x - \mu)^\top \Sigma^{-1} (x - \mu) \right)$$




# Metrics for Evaluating Cluster Performance

# Conclusion

