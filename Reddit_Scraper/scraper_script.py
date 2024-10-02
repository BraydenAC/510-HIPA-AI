import praw
import re
import csv
import os
import random

# Function to remove image URLs from text
def remove_image_urls(text):
    return re.sub(r'http\S+\.(jpg|jpeg|png|gif)', '', text)

# Function to check if any of the keywords are in the text
def contains_keywords(text, keywords):
    return any(keyword.lower() in text.lower() for keyword in keywords)

# Authenticate with Reddit
reddit = praw.Reddit(
    client_id='3GlV4o_NdjsbuFARNIcTug',
    client_secret='izjIRlCWazYWMM4q7CUEQWqX08PA6A',
    user_agent='HIPAAI/0.0.1/Fearless-Remote-855',
    username='Fearless-Remote-855',
    password=':r8G5ArnrdwrRy7'
)

# Create a directory to save the CSV files
output_dir = 'reddit_posts'
os.makedirs(output_dir, exist_ok=True)

# Define the keywords to filter by
keywords = ["patient", "physician", "doctor", "case"]

# Collect posts from the 'nursing' and 'medicine' subreddits
subreddit = reddit.subreddit('nursing+nursepractitioner')
posts = []

for post in subreddit.hot(limit=500):
    if not post.is_self:  # Check if the post is a self-post (text-only)
        continue
    if contains_keywords(post.title, keywords) or contains_keywords(post.selftext, keywords):
        posts.append({
            'title': post.title,
            'selftext': remove_image_urls(post.selftext),
            'created_utc': post.created_utc
        })

        # Collect comments for each post
        post.comments.replace_more(limit=10)
        for comment in post.comments.list():
            if contains_keywords(comment.body, keywords):
                posts.append({
                    'title': f"RE: {post.title}",
                    'selftext': remove_image_urls(comment.body),
                    'created_utc': comment.created_utc
                })

# Randomize the collection of posts
random.shuffle(posts)

# Handle missing values
for post in posts:
    if not post['selftext']:
        post['selftext'] = 'No content'

# Save each post and its comments as separate CSV files
for i, post in enumerate(posts):
    file_path = os.path.join(output_dir, f'post_{i+1}.csv')
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['title', 'selftext', 'created_utc']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow(post)

print(f"Data collection complete. Posts and comments saved to '{output_dir}' directory.")
