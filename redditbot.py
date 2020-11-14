from typing import List, Set
import random
from datetime import datetime
import time
import logging

from praw import Reddit
from praw.models import Subreddit, Submission

from model import Model


SLEEP_BETWEEN_REPLIES = 3

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class RedditBot():
    def __init__(self, praw_bot_name: str, is_silent: bool = True) -> None:
        self.reddit = Reddit(praw_bot_name, config_interpolation='basic')
        self.model = Model(self.reddit.config.custom['model_name'])
        self.is_silent = is_silent

    def _get_subreddits(self) -> List[Subreddit]:
        subreddits = []
        for subreddit_name in self.reddit.config.custom['subreddits'].split(','):
            subreddit = self.reddit.subreddit(subreddit_name)
            subreddits.append(subreddit)
        return subreddits

    def _get_replied_submissions(self) -> Set[str]:
        redditor = self.reddit.redditor(self.reddit.config.custom['bot_name'])
        summissions_id = set()
        for comment in redditor.comments.hot():
            summissions_id.add(comment.submission.id)
        return summissions_id

    def _get_submissions(self, subreddits: List[Subreddit],
                         check_hot_count: int = 500, max_submission_count: int = 5) -> List[Submission]:
        replied_submissions_id = self._get_replied_submissions()

        submissions = []
        for subreddit in subreddits:
            for submission in subreddit.hot(limit=check_hot_count):
                if submission.id in replied_submissions_id:
                    continue
                submissions.append(submission)

        if len(submissions) == 0:
            logger.debug(f'no submissions found')

        random.shuffle(submissions)

        submissions = submissions[:max_submission_count]
        logger.debug(f'found submissions {submissions}')
        return submissions

    def _prepare_reply(self, title: str, text: str) -> str:
        return self.model.generate_text(title, text)

    def _make_reply(self, submission: Submission) -> None:
        reply_text = self._prepare_reply(submission.title, submission.selftext)
        if not self.is_silent:
            submission.reply(reply_text)
        logger.debug(f'made reply submission={submission.title} reply={reply_text}')

    def _make_replies(self, submissions: List[Submission]) -> None:
        for submission in submissions:
            self._make_reply(submission)
            time.sleep(SLEEP_BETWEEN_REPLIES) 

    def run(self) -> None:
        subreddits = self._get_subreddits()
        submissions = self._get_submissions(subreddits)
        self._make_replies(submissions)
