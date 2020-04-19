#!/usr/bin/env python
# coding: utf-8

### Python Project 17: Winning Patterns In Jeopardy
# In this project, we'll work with a dataset of Jeopardy questions to figure out some patterns in the questions that could help us win.
 The dataset is named `jeopardy.csv`, and contains `20000` rows from the beginning of a full dataset of Jeopardy questions, which you can download [here](https://www.reddit.com/r/datasets/comments/1uyd0t/200000_jeopardy_questions_in_a_json_file). Here's the beginning of the file:

import pandas
import csv
jeopardy = pandas.read_csv("jeopardy.csv")
jeopardy.head()
jeopardy.columns

# Removing spaces in front of columns & assigning results back to 'jeopardy'
jeopardy.columns = ['Show Number', 'Air Date', 'Round', 'Category', 'Value', 'Question', 'Answer']

#### Normalize Text Columns
# Writing a function to normalize questions and answers. It should:
# - Take in a string.
# - Convert the string to lowercase.
# - Remove all punctuation in the string.
# - Return the string.

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

#### Normalize the `Question` & `answer` column:
jeopardy["clean_question"] = jeopardy["Question"].apply(normalize_text)
jeopardy["clean_answer"] = jeopardy["Answer"].apply(normalize_text)
jeopardy["clean_value"] = jeopardy["Value"].apply(normalize_values)

jeopardy.head()

# Using the __pandas.to_datetime__ function to convert the `Air Date` column to a datetime column.
jeopardy["Air Date"] = pandas.to_datetime(jeopardy["Air Date"])

jeopardy.dtypes

#### Answers In Questions
# In order to figure out whether to study past questions, study general knowledge, or not study it all, 
 
# Writing a function that takes in a row in jeopardy, as a Series.
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

#### Recycled Questions
# Let's say we want to investigate how often new questions are repeats of older ones.  
# This will enable us to check if the terms in questions have been used previously or not. 
# Only looking at words with six or more characters enables us to filter out words like `the` and `than`, 
# which are commonly used, but don't tell us a lot about a question.

question_overlap = []
terms_used = set()

# sorting jeopardy by ascending air date
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
 
#### Low Value Vs High Value Questions
# Let's say we only want to study questions that pertain to high value questions instead of low value questions. 
# We can actually figure out which terms correspond to high-value questions using a chi-squared test. 
# - Low value -- Any row where __Value__ is less than `800`.
# - High value -- Any row where __Value__ is greater than `800`.

# Creating a function that takes in a row from a Dataframe
def determine_value(row):
    value = 0
    if row["clean_value"] > 800:
        value = 1
    return value

# Determining which questions are high and low value
jeopardy["high_value"] = jeopardy.apply(determine_value, axis=1)

# We'll now be able to loop through each of the terms from the, `terms_used`, and then find the words with the biggest differences in usage between high and low value questions, by selecting the words
# with the highest associated chi-squared values. Doing this for all of the words would take a very long time, so we'll just do it for a small sample now.

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

# Now that you've found the observed counts for a few terms, we can compute the expected counts and the chi-squared value.
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
