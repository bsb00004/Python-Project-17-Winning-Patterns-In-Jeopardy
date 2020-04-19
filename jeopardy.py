#!/usr/bin/env python
# coding: utf-8

# ## Python Project 17: Winning Patterns In Jeopardy
# 
# Jeopardy is a popular TV show in the US where participants answer questions to win money. It's been running for a few decades, and is a major force in popular culture.
# 
# Let's say we want to compete on Jeopardy, and we are looking for any edge to get to win. In this project, we'll work with a dataset of Jeopardy questions to figure out some patterns in the questions that could help you win.
# 
# The dataset is named `jeopardy.csv`, and contains `20000` rows from the beginning of a full dataset of Jeopardy questions, which you can download [here](https://www.reddit.com/r/datasets/comments/1uyd0t/200000_jeopardy_questions_in_a_json_file). Here's the beginning of the file:
# 
# <img src="https://dq-content.s3.amazonaws.com/Nlfu13A.png">
# 
# As you can see, each row in the dataset represents a single question on a single episode of Jeopardy. Here are explanations of each column:
# 
# - __Show Number__ -- the Jeopardy episode number of the show this question was in.
# - __Air Date__ -- the date the episode aired.
# - __Round__ -- the round of Jeopardy that the question was asked in. Jeopardy has several rounds as each episode progresses.
# - __Category__ -- the category of the question.
# - __Value__ -- the number of dollars answering the question correctly is worth.
# - __Question__ -- the text of the question.
# - __Answer__ -- the text of the answer.

# In[35]:


import pandas
import csv

jeopardy = pandas.read_csv("jeopardy.csv")

jeopardy.head()


# In[36]:


jeopardy.columns


# In[37]:


# Removing spaces in front of columns & assigning results back to 'jeopardy'
jeopardy.columns = ['Show Number', 'Air Date', 'Round', 'Category', 'Value', 'Question', 'Answer']


# ### Normalize Text Columns
# Before you can start doing analysis on the Jeopardy questions, you need to normalize all of the text columns (the Question and Answer columns).
# 
# Writing a function to normalize questions and answers. It should:
# 
# - Take in a string.
# - Convert the string to lowercase.
# - Remove all punctuation in the string.
# - Return the string.

# In[38]:


import re

def normalize_text(text):
    text = text.lower()
    text = re.sub("[^A-Za-z0-9\s]", "", text)
    return text

def normalize_values(text):
    text = re.sub("[^A-Za-z0-9\s]", "", text)
    try:
        text = int(text)
    except Exception:
        text = 0
    return text


# ### Normalize the `Question` column.
# - Using the Pandas __Series.apply__ method to apply the function to each item in the `Question` column.
# - Assign the result to the `clean_question` column.
# 
# ### Normalize the `Answer` column.
# - Using the Pandas __Series.apply__ method to apply the function to each item in the `Answer` column.
# - Assign the result to the `clean_answer` column.

# In[39]:


jeopardy["clean_question"] = jeopardy["Question"].apply(normalize_text)
jeopardy["clean_answer"] = jeopardy["Answer"].apply(normalize_text)
jeopardy["clean_value"] = jeopardy["Value"].apply(normalize_values)

jeopardy.head()


# Using the __pandas.to_datetime__ function to convert the `Air Date` column to a datetime column.

# In[40]:


jeopardy["Air Date"] = pandas.to_datetime(jeopardy["Air Date"])

jeopardy.dtypes


# ### Answers In Questions
# In order to figure out whether to study past questions, study general knowledge, or not study it all, it would be helpful to figure out two things:
# 
# - How often the answer is deducible from the question.
# - How often new questions are repeats of older questions.
# 
# You can answer the second question by seeing how often complex words (> 6 characters) reoccur. You can answer the first question by seeing how many times words in the answer also occur in the question. We'll work on the first question now, and come back to the second.
# 
# Writing a function that takes in a row in jeopardy, as a Series.

# In[41]:


def count_matches(row):
    # Split the clean_answer column around spaces and assign to the variable
    split_answer = row["clean_answer"].split(" ")
    
    # Split the clean_question column around spaces and assign to the variable
    split_question = row["clean_question"].split(" ")
    if "the" in split_answer:
        split_answer.remove("the")
    if len(split_answer) == 0:
        return 0
    match_count = 0
    for item in split_answer:
        if item in split_question:
            match_count += 1
    return match_count / len(split_answer)

# Pass the axis=1 argument to apply the function across each row.
jeopardy["answer_in_question"] = jeopardy.apply(count_matches, axis=1)

# mean
jeopardy["answer_in_question"].mean()


# ### Answer terms in the question
# The answer only appears in the question about 6% of the time. This isn't a huge number, and means that we probably can't just hope that hearing a question will enable us to figure out the answer. We'll probably have to study.

# ### Recycled Questions
# Let's say we want to investigate how often new questions are repeats of older ones. We can't completely answer this, because we only have about 10% of the full Jeopardy question dataset, but we can investigate it at least.
# 
# To do this, we can:
# 
# - Sort `jeopardy` in order of ascending air date.
# - Maintain a set called `terms_used` that will be empty initially.
# - Iterate through each row of `jeopardy`.
# - Split `clean_question` into words, remove any word shorter than 6 characters, and check if each word occurs in `terms_used`.
#     - If it does, increment a counter.
#     - Add each word to `terms_used`.
#     
# This will enable us to check if the terms in questions have been used previously or not. Only looking at words with six or more characters enables us to filter out words like `the` and `than`, which are commonly used, but don't tell us a lot about a question.

# In[42]:


question_overlap = []
terms_used = set()

# ort jeopardy by ascending air date
jeopardy = jeopardy.sort_values("Air Date")

for i, row in jeopardy.iterrows():
        split_question = row["clean_question"].split(" ")
        split_question = [q for q in split_question if len(q) > 5]
        match_count = 0
        for word in split_question:
            if word in terms_used:
                match_count += 1
        for word in split_question:
            terms_used.add(word)
        if len(split_question) > 0:
            match_count /= len(split_question)
        question_overlap.append(match_count)

jeopardy["question_overlap"] = question_overlap

jeopardy["question_overlap"].mean()


# ### Question overlap
# There is about 70% overlap between terms in new questions and terms in old questions. This only looks at a small set of questions, and it doesn't look at phrases, it looks at single terms. This makes it relatively insignificant, but it does mean that it's worth looking more into the recycling of questions.
# 
# ### Low Value Vs High Value Questions
# Let's say we only want to study questions that pertain to high value questions instead of low value questions. This will help us earn more money when we are on Jeopardy.
# 
# We can actually figure out which terms correspond to high-value questions using a chi-squared test. We'll first need to narrow down the questions into two categories:
# 
# - Low value -- Any row where __Value__ is less than `800`.
# - High value -- Any row where __Value__ is greater than `800`.

# In[43]:


# Creating a function that takes in a row from a Dataframe
def determine_value(row):
    value = 0
    if row["clean_value"] > 800:
        value = 1
    return value

# Determining which questions are high and low value
jeopardy["high_value"] = jeopardy.apply(determine_value, axis=1)


# We'll now be able to loop through each of the terms from the, `terms_used`, and:
# 
# - Find the number of low value questions the word occurs in.
# - Find the number of high value questions the word occurs in.
# - Find the percentage of questions the word occurs in.
# - Based on the percentage of questions the word occurs in, find expected counts.
# - Compute the chi squared value based on the expected counts and the observed counts for high and low value questions.
# 
# We can then find the words with the biggest differences in usage between high and low value questions, by selecting the words with the highest associated chi-squared values. Doing this for all of the words would take a very long time, so we'll just do it for a small sample now.

# In[44]:


# Create a function that takes in a word
def count_usage(term):
    low_count = 0
    high_count = 0
    for i, row in jeopardy.iterrows():
        if term in row["clean_question"].split(" "):
            if row["high_value"] == 1:
                high_count += 1
            else:
                low_count += 1
    return high_count, low_count

# Converting `terms_used` into a list using the list function, 
# and assigning the first 5 elements to `comparison_terms`
comparison_terms = list(terms_used)[:5]

# Creating an empty list
observed_expected = []
# Running the function on the term to get the high value and low value counts
# Append the result of running the function to `observed_expected` list
for term in comparison_terms:
    observed_expected.append(count_usage(term))

observed_expected


# Now that you've found the observed counts for a few terms, you can compute the expected counts and the chi-squared value.

# In[46]:


from scipy.stats import chisquare
import numpy as np

# Finding the number of rows in `jeopardy` where high_value is 1, and assigning to `high_value_count`
high_value_count = jeopardy[jeopardy["high_value"] == 1].shape[0]

# Finding the number of rows, where high_value is 0, and assign to `low_value_count`
low_value_count = jeopardy[jeopardy["high_value"] == 0].shape[0]

# Creating an empty list
chi_squared = []

# 
for obs in observed_expected:
    total = sum(obs) # add up both items in the list
    total_prop = total / jeopardy.shape[0] # Divide total by the number of rows
    high_value_exp = total_prop * high_value_count # to get the expected term count for high value rows
    low_value_exp = total_prop * low_value_count # to get the expected term count for low value rows
    
    # Using the `scipy.stats.chisquare` function to compute the chi-squared value and 
    # p-value given the expected and observed counts
    observed = np.array([obs[0], obs[1]])
    expected = np.array([high_value_exp, low_value_exp])
    chi_squared.append(chisquare(observed, expected))

chi_squared


# ### Chi-squared results
# None of the terms had a significant difference in usage between high value and low value rows. Additionally, the frequencies were all lower than 5, so the chi-squared test isn't as valid. It would be better to run this test with only terms that have higher frequencies.
