---
title: Exploring Bayesian Classification
date: 2024-10-10 18:00:00 +0000
categories: [Machine Learning]
tags: [Machine Learning, Bayesian Classification, Probability, ]
math: true
---

# Introduction

# Statistics Overview

Before we embark upon this blog's core concepts, let us refresh ourselves with an overview of sets, probability rules, and how we combine different probabilities.

## Sets and Venn Diagrams

- Union (A ∪ B): This operation combines all elements from sets A and B. For example:

$$ A = {cat, dog, fish} $$

$$ B = {dog, snake, turtle} $$

$$(A ∪ B) = {cat, dog, fish, snake, turtle}$$

-	Intersection (A ∩ B): This operation finds common elements in both sets. Using the same sets:

$$A ∩ B = {dog}$$

-	Mutually Exclusive Events: If two events can't happen simultaneously. For example, getting heads and tails in the same toss. Then their intersection is zero: 

$$P(A ∩ B)  = 0$$

## Probability Rules:

- Sum Rule: The sum of all the probabilities of all possible outcomes of an event equals 1. For example, in a coin toss, P(heads) + P(tails) = 1, where each has a probability of 0.5.

- Chain Rule (product rule): This calculates the joint probability of two events happening together:

$$P(A,B)=P(A ∣ B)×P(B)$$

Here, P(A∣B) is the conditional probability of A happening given that B has already happened.

# Conditional Probability and Bayes' Rule

This is the foundation of Bayesian Classification. Conditional Probability answers the question: "What is the probability of one event given that another event has already occured?". Lets explore the two formulas.

- Conditional Probability: If A and B are two events, the conditional probability of A given B is: 

$$ P(A ∣ B) = \frac{P(A ∩ B)}{P(B)} $$

This formula helps calculate the probability of an event A when we know event B has occured.

- Bayes' Rule: This is a key tool in Bayesian statistics, used to update the probability of a hypothesis based on new evidence:

$$ P(B ∣ A) = \frac{P(A | B) \cdot P(B)}{P(A)} $$

Where:
- P(B ∣ A) is the posterior probability (the probability of B after seeing A).
- P(A ∣ B) is the likelihood (the probability of A happening, given B is true).
- P(B) is the prior probability of B (how likely B was before seeing A).
- P(A) is the marginal likelihood of A (the total probability of A happening).

## Example: Medical Testing (COVID Test)

Let's walk through the Bayes' rule with a real world example to calculate the probability of someone having COVID given a positive test result. First lets look at the definitions of the various symbols and their meaning:

- \+ (test positive)
- \- (test negative)
- COV (has COVID)
- NCOV (does not have COVID)

Here is the necessary information:

- Sensitivity (True Positive Rate): The probability of testing positive (A) if you have COVID is 95% (B). Since were looking for the likelihood of A happening given B we use P(A ∣ B) in other words P(\+ ∣ COV) = 0.95.

- Specificity (True Negative Rate): The probability of testing negative (A) if you dont have COVID is 99% (B). This means the probability of a false positive is 1%, so P(\+ ∣ NCOV) = 0.01.

- Prevalence: The probability that someone in the population has COVID (prior probability) is 1%. So P(COV) = 0.01.

- Negative Prevalence: The probability that someone does not have COVID is 99%. So P(NCOV) = 0.99.

Our goal is to calculate the posterior probability (the probability of an event occuring after considering new information)of someone having COVID given a positive test result, this is denoted as P(COV ∣ +).

To find this we use Bayes' Rule:

$$ P(COV ∣ +) = \frac{P(+ | COV) \cdot P(COV)}{P(+)} $$

Where:
- P(+ ∣ COV) is the sensitivity.
- P(COV) is the prevalence.
- P(+) is the total probability of testing positive.

### Step 1: Calculate P(\+) 

We want to find the total probability of testing positive. This can happen in two ways:

1. You test positive and have COVID P(\+ ∣ COV) $$\cdot$$ P(COV)
2. You test positive but do not have COVID P(+ ∣ NCOV) $$\cdot$$ P(NCOV)

So, P(\+) is the sum of these two possibilities:

$$P(+) = P(+ ∣ COV) \cdot P(COV) + P(+ ∣ NCOV) \cdot P(NCOV)$$

Substituting the values:

$$P(+) = (0.95 \cdot 0.01) + (0.01 \cdot 0.99)$$

$$P(+) = (0.0095) + (0.0099) = 0.0194$$

### Step 2: Apply Bayes' Rule to calculate P(COV ∣ \+)

Now that we have P(\+) = 0.0194, we can substitute the values into Bayes' Rule:

$$ P(COV ∣ +) = \frac{0.95 \cdot 0.01}{0.0194} $$

$$ P(COV ∣ +) = \frac{0.0095}{0.0194} = 0.4897$$

### Interpretation

The result is approximately 0.49 or 49%.

This means that even with a positive test result, there is only a 49% chance that the person actually has COVID. The reason this probability is lower than expected, despite the high sensitivity and specificity, is due to the low prevalence of COVID with only 1% of the population having it.

# Bayesian Classification

Bayesian classification is a method used to classify data based on Bayes' Rule, which calculates the probability that a given instance belongs to a particular class based on prior knowledge and evidence. For example, imagine you’re trying to figure out whether an email is spam or not spam. Bayesian classification helps by looking at the features of the email like certain keywords and uses probability to decide the most likely class spam or not spam.

## Example: Classifying Spam or not Spam Emails

### 1. Prior Probability

The prior probability is the initial belief about how likely each class is, before seeing any new data. This is based on historical information for example, if in your email inbox, 30% of all emails are spam and 70% are not spam, the prior probabilities are:

- P(spam) = 0.30
- P(not spam) = 0.70

### 2. Likelihood

The likelihood is how probable it is to observe certain features or evidence given that the instance belongs to a specific class (how often it appears as spam or not spam).

Let’s say we’re trying to classify an email based on the word “lottery". We know from experience that:

- In spam emails, the word "lottery" appears 80% of the time:

$$P(lottery ∣ spam) = 0.80$$

- In non-spam emails, the word "lottery" appears only 5% of the time:

$$P(lottery ∣ not spam) = 0.05$$

This tells us that if an email contains the word "lottery," it’s much more likely to be spam than not spam, based on past data.

### 3. Evidence (Marginal Probability)

The evidence is the overall probability of seeing the feature (in this case, the word “lottery”) across all classes. It’s the total likelihood of observing that feature, regardless of the class. This is computed as:

$$P(lottery) = P(lottery ∣ spam) \cdot P(spam) + P(lottery ∣ not spam) \cdot P(not spam)$$

### 4. Posterior Probability

The posterior probability is the updated probability of a class given the new evidence (the feature we observed). This is what we calculate using Bayes’ Rule:

$$ P(spam ∣ lottery) = \frac{P(lottery | spam) \cdot P(spam)}{P(lottery)} $$

This equation tells us the probability that an email is spam given that it contains the word “lottery”.

We also calculate:

$$ P(not spam ∣ lottery) = \frac{P(lottery | not spam) \cdot P(not spam)}{P(lottery)} $$

Whichever class (spam or not spam) has the higher posterior probability is chosen as the predicted class.

### 5. Classification Decision

Once we have the posterior probabilities for all possible classes, we choose the class with the highest probability. In this case, if P(spam ∣ lottery) is higher than P(not spam ∣ lottery), we classify the email as spam.

### Example Breakdown

Now we have all our data we can discern whether an email we receive with the word "lottery" is spam or not. Using the data we discussed:

- P(lottery ∣ spam) = 0.80
- P(lottery ∣ not spam) = 0.05
- P(spam) = 0.30
- P(not spam) = 0.70

First lets calculate the marginal probability P(lottery):

$$P(lottery) = P(lottery ∣ spam) \cdot P(spam) + P(lottery ∣ not spam) \cdot P(not spam)$$

$$P(lottery) = (0.80 \cdot 0.30) + (0.05 \cdot 0.70)$$

$$P(lottery) = 0.24 + 0.035 = 0.275$$

Now, using Bayes' Rule to calculate the probability that this email is spam:

$$P(spam ∣ lottery) = \frac{0.80 \cdot 0.30}{0.275} = \frac{0.24}{0.275} = 0.872$$

This means that when "lottery" appears in the email, there is an 87% chance that the email is spam.

Similarly, the probability that the email is not spam:

$$P(not spam ∣ lottery) = \frac{0.05 \cdot 0.70}{0.275} = \frac{0.035}{0.275} = 0.13$$

This shows that there is only a 13% chance that the email is not spam if it contains the word "lottery".

Thus since $$P(spam ∣ lottery) > P(not spam ∣ lottery)$$, we classify the email as spam.

# Naive Bayesian Classification

Naive Bayesian Classification refers to the simplifying assumption that all features are conditionally independent of each other, given the class label. This assumption is called naïve because it simplifies the real-world scenario where features like words in a document might actually be correlated, but the model ignores those dependencies to make the computation easier.

To put simply normal Bayesian Classification accounts for the possible relationships between words, making it more complex and computationally intensive but more accurate when features are related. On the other hand Naive Bayes assumes that each word contributes independently and ignores the words potential relationships.

Here is the formula for Naive Bayes Classification:

$$P(C ∣ X) \propto P(C) \prod_{i=1}^{n} P(X_i ∣ C)$$

Where: 
- P(C ∣ X) is the posterior probability of class C given the feature set $$X = (X_1, X_2, ..., X_n)$$
- P(C) is the prior probability of class C
- $$P(X_i ∣ C)$$ is the likelihood of feature $$X_i$$ given C
- $$\prod_{i=1}^{n}$$ represents the product of all the likelihoods for each feature $$X_i$$, from 1 to n (the total number of features)

The formula basically says the probability of class C given feature set X is directly proportional to the product of all features likelihoods times the prior probability.

## Example: Classifying Spam or not Spam Emails with multiple words

Lets go through another example to understand how to use this formula. Again we have an email and want to determine whether it is spam or not spam based on the occurence of these words:

- Winner
- Lottery
- Friend
- Money

### Step 1: Prior probabilities

We will start with the same prior probabilities as before for each class:

- P(spam) = 0.30
- P(not spam) = 0.70

### Step 2: Likelihood

Next, we calculate the likelihoods for each word in both the spam and not spam categories. These likelihoods are learned from previous data, for this example we will make up random values.

#### Spam:

- P(winner ∣ spam) = 0.80
- P(lottery ∣ spam) = 0.90
- P(friend ∣ spam) = 0.40
- P(money ∣ spam) = 0.70

#### Not Spam:

- P(winner ∣ not spam) = 0.10
- P(lottery ∣ not spam) = 0.05
- P(friend ∣ not spam) = 0.60
- P(money ∣ not spam) = 0.20

### Step 3: Calculate Posterior Probability for Each Class

Using Naive Bayes, we assume the words are conditionally independent, so we multiply the probabilities of each word appearing, given the class (spam or not spam). We use Bayes' Rule to compute the posterior probabilities of each class.

#### For Spam:

$$P(spam ∣ winner, lottery, friend, money) = \frac{P(spam) \cdot P(winner ∣ spam) \cdot P(lottery ∣ spam) \cdot P(friend ∣ spam) \cdot P(money ∣ spam)}{P(winner, lottery, friend, money)}$$

#### Substituting values:

$$P(spam ∣ winner, lottery, friend, money) = \frac{0.30 \cdot 0.80 \cdot 0.90 \cdot 0.40 \cdot 0.70}{P(winner, lottery, friend, money)}$$

#### Calculating the Numerator:

$$0.30 \cdot 0.80 \cdot 0.90 \cdot 0.40 \cdot 0.70 = 0.06048$$

#### For Not Spam:

$$P(not spam ∣ winner, lottery, friend, money) = \frac{P(not spam) \cdot P(winner ∣ not spam) \cdot P(lottery ∣ not spam) \cdot P(friend ∣ not spam) \cdot P(money ∣ not spam)}{P(winner, lottery, friend, money)}$$

#### Substituting values:

$$P(not spam ∣ winner, lottery, friend, money) = \frac{0.70 \cdot 0.10 \cdot 0.05 \cdot 0.60 \cdot 0.20}{P(winner, lottery, friend, money)}$$

#### Calculating the Numerator:

$$0.70 \cdot 0.10 \cdot 0.05 \cdot 0.60 \cdot 0.20 = 0.00042$$

### Step 4: Calculate the Evidence (Denominator)

The denominator P(winner, lottery, friend, money) is the same for both classes. Since we only care about comparing two classes, we dont need to compute it directly. This is why the denominator is missing in the formula mentioned above. As a result we only compare the numerators.

### Step 5: Compare the Posterior probabilities

$$P(spam ∣ winner, lottery, friend, money) \propto 0.06048$$

$$P(not spam ∣ winner, lottery, friend, money) \propto 0.00042$$

Since 0.06048 is much larger than 0.00042, the email is far more likely to be classified as spam.