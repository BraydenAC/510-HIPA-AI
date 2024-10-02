# 510-HIPA-AI
Repository to hold code relating to the Main Project being completed by Brayden and Zoiya


# The Dataset
All data is sourced from reddit threads.  The data itself consists of An individual
csv file for each of unique textual block, whether it be a main post or a following comment.

Data was collected via scraper in a randomized collection method, with only text blocks
that met a predetermined search criteria being collected.  

This Dataset consists of 2084 seperate elements

# How to replicate data collection

In order to use the scraper to complete the same procedure for data collection,
simply place the name of a subreddit into line 32 of scraper_script.py, in the 
format of 'subredditName+subredditName+etc...'

Once you run the script, the scraper will randomly retrieve a number of random
posts up to the limit specified in line 35 of scraper_script.py.  These posts will be
added to an existing directory within the project as csv files, or if the directory does
not exist, one will be created automatically.

Once data has been collected run reddit_merger.py in order to take all collected bits of
data and consolidate into a single csv file, with the format of each text block being a 
row, and the columns being 'title', 'selftext', and 'created_utc.'


# Time of collection
9-30-2024, 2:10-2:22 pm--collected 1199 posts.
10/1/2024, 4:30pm -- increased collected posts to 2084

# Annotation time estimate
10 - 25 seconds







