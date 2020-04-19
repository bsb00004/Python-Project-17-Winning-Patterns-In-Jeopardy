## Python Project 17: Winning Patterns In Jeopardy

Jeopardy is a popular TV show in the US where participants answer questions to win money. It's been running for a few decades, and is a major force in popular culture.

Let's say we want to compete on Jeopardy, and we are looking for any edge to get to win. In this project, we'll work with a dataset of Jeopardy questions to figure out some patterns in the questions that could help you win.

The dataset is named `jeopardy.csv`, and contains `20000` rows from the beginning of a full dataset of Jeopardy questions, which you can download [here](https://www.reddit.com/r/datasets/comments/1uyd0t/200000_jeopardy_questions_in_a_json_file). Here's the beginning of the file:

<img src="https://dq-content.s3.amazonaws.com/Nlfu13A.png">

As you can see, each row in the dataset represents a single question on a single episode of Jeopardy. Here are explanations of each column:

- __Show Number__ -- the Jeopardy episode number of the show this question was in.
- __Air Date__ -- the date the episode aired.
- __Round__ -- the round of Jeopardy that the question was asked in. Jeopardy has several rounds as each episode progresses.
- __Category__ -- the category of the question.
- __Value__ -- the number of dollars answering the question correctly is worth.
- __Question__ -- the text of the question.
- __Answer__ -- the text of the answer.

### Steps
  1. Normalize Text Columns
    - Normalize the Question column
    - Normalize the Answer column
  2. Writing function to determine if we can find answers in questions.
  3. Writing function to determine if we can find answers in recyled questions.
  4. Study questions that pertain to high value questions instead of low value questions.
  5. computing the expected counts and the chi-squared value.
  
### Chi-squared results
None of the terms had a significant difference in usage between high value and low value rows. Additionally, the frequencies were all lower than 5, so the chi-squared test isn't as valid. It would be better to run this test with only terms that have higher frequencies.
