# Rating Product & Sorting Reviews of Amazon

Rating Product & Sorting Reviews of Amazon Homework of Miuul Data Science & Machine Learning Bootcamp

## Business Problem

One of the most important problems in e-commerce is the correct calculation of the points given to the products after
sales. <br>The solution to this problem means providing greater customer satisfaction for the e-commerce site,
prominence of the product for the sellers and a seamless shopping experience for the buyers. <br>Another problem is the
correct ordering of the comments given to the products. Since misleading comments will directly affect the sale of the
product, it will cause both financial loss and loss of customers.<br> In the solution of these 2 basic problems, the
customers will complete the purchasing journey without any problems, while the e-commerce site and the sellers will
increase their sales.

## Story of Dataset

This dataset, which includes Amazon product data, includes product categories and various metadata. The product with the
most reviews in the electronics category has user ratings and reviews.

| Variable       |Description|
|:---------------|:----|
| reviewerID     |User ID|
| asin           |Product ID|
| reviewerName   |User’s name|
| helpful        |Rating of useful evaluation|
| reviewText     |Review|
| overall        |Product’s rating|
| summary        |Summary of review|
| unixReviewTime |Review Time|
| reviewTime     |Review Time Raw|
| day_diff       |Number of days since review|
| helpful_yes    |Number of times the review was found useful|
| total_vote     |Number of votes for the review|

## TASKS

1. Calculate the Average Rating according to the current comments and compare it with the existing average rating.
    1. Calculate the average score of the product.
    2. Calculate the weighted average score by date.
    3. Compare and interpret the average of each time period in weighted scoring.
2. Specify 20 reviews for the product to be displayed on the product detail page.
    1. Create the helpful_no variable.
        1. total vote is the total number of up-downs given to a comment.
        2. up means helpful.
        3. There is no helpful_no variable in the data set, it must be generated over existing variables.
        4. Find the number of votes that are not helpful (helpful_no) by subtracting the number of helpful votes (
           helpful_yes) from the total number of votes (total_vote).
    2. Calculate score_pos_neg_diff, score_average_rating and wilson_lower_bound scores and add them to the data.
        1. Define score_pos_neg_diff, score_average_rating and wilson_lower_bound functions to calculate
           score_pos_neg_diff, score_average_rating and wilson_lower_bound scores.
        2. Create scores according to score_pos_neg_diff. Next; Save as score_pos_neg_diff in df.
        3. Create scores according to score_average_rating. Next; Save it as score_average_rating in df.
        4. create scores based on wilson lower_bound. Next; Save it under the name wilson_lower_bound.
    3. Identify 20 Interpretations and interpret the results.
        1. Identify and rank the top 20 comments according to wilson_lower_bound.
        2. Interpret the results.
