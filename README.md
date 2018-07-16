
# Discrete Form of Bayesian Classification


## Problem Description

In this lab, we'll learn how to use (Naive) Bayesian Inference with discrete data.  


## Objectives
* Understand and use each part of Bayes' Theorem, including the **_Prior_**, **_Posterior_**, **_Likelihood_**, and **_Normalization Constant_**.
* Apply Bayes' Theorem to make inferences about a dataset containing discrete data.  

### Review of Probability Terms

Joint Probability:  $ P(A \cap B)$

The probability of two events occuring together.  $ P(A \cap B)$ equals $ P(B \cap A)$.

Conditional Probability: $P(A \mid B)$  The probability of event A given event B.

Marginal Probability: $P(A)$  The probability of occurrence of the of the single event A. 
<br>
<br>
<center>**_Practice Questions: Probability_**</center>

| **Customer** |    **1**   |    **2**   |    **3**   |    **4**   |   **5**   |    **6**   |    **7**   |    **8**   |    **9**   |   **10**   |
|:--------:|:------:|:------:|:------:|:------:|:-----:|:------:|:------:|:------:|:------:|:------:|
|   **Food**   |  Donut |  Donut |  Donut | Muffin | Donut | Muffin |  Donut |  Donut | Muffin |  Donut |
|   **Drink**  | Coffee |   Tea  | Coffee |   Tea  |  Tea  | Coffee | Coffee | Coffee |   Tea  | Coffee |


Assume this sample is representative of the overall purchasing patterns seen at a coffee shop.  Fill in the following contingency table, and then use it to answer the probability questions below:

|        | **Donut** | **Muffin** | **Total** |
|:------:|:-----:|:------:|:-----:|
| **Coffee** |      |       |      |
|   **Tea**  |      |       |      |
|  **Total** |      |       |      |  

**1.)** The marginal probability of a person ordering a donut.   
**Answer:** 

**2.)** The joint probability of a person ordering coffee and a donut.  
**Answer:**   

**3.)** The joint probability of a person ordering donut and a coffee.  
**Answer:**   

**4.)** The conditional probability of a person ordering a coffee, given that they have already ordered a donut.    
**Answer:**   

**5.)** The conditional probability of a person ordering a donut, given that they have already ordered a coffee.    
**Answer:**   


### Bayes' Theorem

As you noticed in the practice problems above, $P(A \mid B)$ does not equal $P(B \mid A)$!  However, if we know $P(A \mid B)$, we can use **_Bayes' Theorem_** (also called Bayes' Rule) to calculate $P(B \mid A)$.  

Recall that the formula for Bayes' Theorem is:

$$ \LARGE P(A \mid B) = \frac{P(B \mid A) \, P(A)}{P(B)} $$

and that the following vocabulary corresponds to each of the terms in the equation:

$P(A \mid B)$: **_Posterior_**.  The probability we're trying to solve for, given some observation.    

$P(B \mid A)$  **_Likelihood_**.  Also called the **_update_**.  

$P(A) $ -- **_Prior_**.  The marginal probability of A.  

$P(B) $ -- **_Normalization Constant_**.  The marginal probability of B.  Dividing by this number ensures that our posterior will always be scaled to between 0 and 1. 

### The Dataset

The file `food_and_drink_sales.csv` contains records of 100,000 different customer transactions from a coffee shop, for customers who bought both a food and drink.  The choices are as follows:

|  Food  |  Drink |
|:------:|:------:|
|  donut | coffee |
| muffin |   tea  |
|  fruit |  water |

For the remainder of this lab, we'll use python and Bayes' Theorem to answer the following question: _Given that a customer has ordered a coffee, what is the probability they will order a donut?_

In the cell below, import the data from `food_and_drink_sales.csv` into a DataFrame.  


```python
import pandas as pd
import numpy as np 

df = None
print(len(df)) #100000
```

  

### A Quick Note on Priors

There are two types of priors we can use: **_Informative Priors_** and **_Uninformative Priors_**. 

An **_Informative Prior_** is a prior you choose, which encompasses information that you want to incorporate in your model.  

An **_Uninformative Prior_** is when you you do not have information you want to include as a prior, so you let the data "speak for itself".  

There are many different ways to determine how to set a prior, but the easiest way to set an uninformative prior is to simply use the **_Marginal Probability_** of the condition in question--that is, the number of times a value occurs in a dataset, divided by the total number of observations in the dataset.  


### Calcuating Marginal Probability

In the cell below, write a function that takes in a dataset, value, and column name for that value, and return the marginal probability of that value in the dataset.


```python
def get_marginal_probability(dataset, a, a_col_name):
    pass

test_df = df[:10]
display(test_df)
print(get_marginal_probability(test_df, 'donut', "Food Sales")) # 0.7
```

### Calculating the Conditional Probability

Complete the function in the cell below.  The function should take in a DataFrame, values for A and B, and the column names for A and B (to make accessing everything a little bit easier). 

This function should return the conditional probability of A given B. 


```python
def get_conditional_probability(dataset, a, a_col_name, b, b_col_name):
    pass

get_conditional_probability(test_df, 'donut', 'Food Sales', 'tea', 'Drink Sales') # 0.5
```

### Calculating the Normalization Constant

The denominator is called the **_Normalization Constant_**.  This is used to scale the value output by Bayes's Theorem to between 0 and 1.  

The Normalization Constant is made up of the **_Marginal Probability_** of the observed data.  In practice, this means that the denominator is made up of  $P(B\mid A) * P(A)$ (the conditional probability of B given A times the prior of A)as we see in the numerator, as well as $P(B \mid !A) * P(!A)$ (the conditional probability of B given not A, times the prior of not A). 

Written out long form, it looks like this:

$$ \LARGE P(A \mid B) = \frac{P(B \mid A) \, P(A)}{P(B \mid A) * P(A) + P(B \mid !A) * P(!A)} $$

In the cell below, write a function that takes in a dataset and a hypothesis, and returns the normalization constant. The function should also take an optional default parameter called `prior`, that is set to `None` by default.  If this parameter is passed in, it should use this prior in the calculations. If it is `None`, the function should then calculate the priors based on the contents of `dataset`. 

**_Hint_**: There's a short cut to calculating things $P(B \mid !A)$. Remember that $ P(B \mid !A) = 1 - P(B \mid A) $. Similarly, $P(!A) = 1 - P(A)$.


```python
def get_norm_constant(dataset, a, a_col_name, b, b_col_name, prior=None):
    pass
 
get_norm_constant(test_df, 'donut', 'Food Sales', 'coffee', 'Drink Sales')
```

```
Expected Output:
P(A): 0.7
P (!A): 0.30000000000000004
P(B|A): 0.8571428571428572
P(B|!A): 0.1428571428571428
0.6428571428571428
```

### Bringing It All Together: Calculating the Posterior


```python
def get_posterior(dataset, a, a_col_name, b, b_col_name, prior=None):
    pass

a = 'donut'
a_col_name = 'Food Sales'
b = 'coffee'
b_col_name = "Drink Sales"

get_posterior(df, a, a_col_name, b, b_col_name)
```

```
Expected Output:
P(A): 0.60157
P (!A): 0.39842999999999995
P(B|A): 0.5999301826886314
P(B|!A): 0.4000698173113686
0.6936385291560131
```

### Interpreting our Results

Given the results from the cell above, answer the following question:

A customer has ordered a coffee.  What is the probability that they will also order a donut?

**_Answer:_** 

### How to Use Bayes' Theorem For Inference

Bayes' Theorem can most easily be used for classification by setting a threshold for the probabilty, and assuming that the answer is true if the probability is greater than our threshold.  

Bayes' Theorem is most powerful when used iteratively, where the posterior for a given round becomes our prior for the next round.  For instance, let's say our company opens a new coffee shop in another town, and that shop does not have much data yet.  Purchasing habits may be different in that part of town, but we won't know until we have a sufficient amount of data to compare the two.  To use Bayesian Inference to make predictions with a smaller dataset from the new store, we can incorporate **_Informative Priors_** calculated as posteriors from the dataset at our main store.  In this way, if we wanted to know the probability of a customer at the new store buying a donut given that they have ordered a coffee, we can use the code we've written on the new dataset, but with $P(A) = 0.6936385$.

# Conclusion

In this lab, we learned:
* How to use a table of data to calculate the **_Prior_**
* How to use a table of data to calculate the **_Likelihood_**
* How to use a table of data to calculate the **_Normalization Constant_**
* How to calculate the **_Posterior_** for a dataset of discrete data using Bayes' Theorem.
