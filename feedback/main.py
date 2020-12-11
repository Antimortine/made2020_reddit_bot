import pathlib
import configparser
from datetime import datetime

from statistics import median

from praw import Reddit
from praw.models import Submission, Comment, Redditor

from pandas import DataFrame

HOME_DIR = str(pathlib.Path.home())
CURRENT_DIR = pathlib.Path(__file__).parent.absolute()

# submission
# first level coment_count
# comment
# replies
# average replies in submission first level
# median replies in submission first level
# score
# average score in submission first level
# median score in submission first level

def get_submission_stat(submission: Submission, bot_comment: Comment) -> dict:
    submission.comments.replace_more(limit=0)

    result = {}

    result['submission_title'] = submission.title.replace('\n', '\\n')
    result['submission_body'] = submission.selftext.replace('\n', '\\n')
    result['submission_num_comments'] = submission.num_comments 
    result['submission_url'] = submission.url
    
    scores = []
    for comment in submission.comments:
        if comment.id == bot_comment.id:
            continue
        scores.append(comment.score)
    result['mean_score'] = sum(scores) / len(scores) if scores else None
    result['median_score'] = median(scores) if scores else None
    result['scores'] = scores

    replies_count = []
    for comment in submission.comments:
        if comment.id == bot_comment.id:
            continue
        replies_count.append(len(comment.replies.list()))
    result['mean_replies_count'] = sum(replies_count) / len(replies_count) if replies_count else None
    result['median_replies_count'] = median(replies_count) if replies_count else None
    result['replies_counts'] = replies_count

    return result

def get_comment_stat(comment: Comment) -> dict:
    comment.refresh()
    comment.replies.replace_more(limit=0)

    result = {}
    result['comment_body'] = comment.body.replace('\n', '\\n')
    result['created_utc'] = datetime.fromtimestamp(comment.created_utc).strftime('%Y-%m-%d %H:%M')
    result['replies_count'] = len(comment.replies.list())
    result['score'] = comment.score

    return result

def get_bot_stat(redditor: Redditor) -> DataFrame:
    result = []
    for comment in redditor.comments.new(limit=None):
        if comment.submission.title == 'test title':
            continue
        if comment.created_utc < 1606078800:
            continue
        comment_stat = get_comment_stat(comment)
        submission_stat = get_submission_stat(comment.submission, comment)
        submission_stat.update(comment_stat)
        result.append(submission_stat)
    return DataFrame(result)


def main():
    config = configparser.ConfigParser()
    config.read(f'{HOME_DIR}/.config/praw.ini')
    bots = config['DEFAULT']['bots'].split(',')

    for bot in bots:
        reddit = Reddit(bot, config_interpolation='basic')
        bot_name = reddit.config.custom['bot_name']
        redditor = reddit.redditor(bot_name)
        get_bot_stat(redditor).to_csv(f'{CURRENT_DIR}/{bot_name}.tsv', sep='\t', index=False, )
        print(f'{bot} done')

if __name__ == '__main__':
    main()

