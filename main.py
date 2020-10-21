import praw
from praw import Reddit


def show_subreddit(reddit: Reddit, subreddit_name: str) -> None:
    subreddit = reddit.subreddit(subreddit_name)
    print(f'subreddit.display_name: {subreddit.display_name}')
    print(f'subreddit.title: {subreddit.title}')
    print(f'subreddit.description: {subreddit.description}')
    top = 10
    print(f'Top {top} hottest submission titles')
    for submission in subreddit.hot(limit=top):
        print(submission.title)


reddit = Reddit('bot', config_interpolation='basic')
print(f'reddit.read_only: {reddit.read_only}')

show_subreddit(reddit, 'learnpython')
