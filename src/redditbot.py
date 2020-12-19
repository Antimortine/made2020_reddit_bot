from typing import List, Set
import random
from datetime import datetime
import time
import logging

from praw import Reddit
from praw.models import Subreddit, Submission

from model import Model

from utils import *


SLEEP_BETWEEN_REPLIES = 3 * 60

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class RedditBot():
    def __init__(self, praw_bot_name: str, is_silent: bool = True,
                 test_submission: bool = True, manual: bool = False) -> None:
        self.reddit = Reddit(
            praw_bot_name,
            config_interpolation='basic'
        )
        self.model = Model(self.reddit.config.custom['model_name'])
        self.is_silent = is_silent
        self.test_submission = test_submission
        self.manual = manual
        logger.debug(f'model {self.reddit.config.custom["model_name"]}')

    def _get_subreddits(self) -> List[Subreddit]:
        subreddits = []
        for subreddit_name in self.reddit.config.custom['subreddits'].split(','):
            subreddit = self.reddit.subreddit(subreddit_name)
            subreddits.append(subreddit)
        return subreddits

    def _get_replied_submissions(self) -> Set[str]:
        redditor = self.reddit.redditor(self.reddit.config.custom['bot_name'])
        summissions_id = set()
        for comment in redditor.comments.new(limit=None):
            summissions_id.add(comment.submission.id)
        return summissions_id

    def _get_submissions(self, subreddits: List[Subreddit],
                         check_hot_count: int = 150,
                         max_submission_count: int = 5) -> List[Submission]:
        replied_submissions_id = self._get_replied_submissions()

        submissions = []
        for subreddit in subreddits:
            for submission in subreddit.hot(limit=check_hot_count):
                if submission.id in replied_submissions_id:
                    continue
                if contains_stop_words(submission.title):
                    continue
                submissions.append(submission)

        if len(submissions) == 0:
            logger.debug(f'no submissions found')

        submissions = sorted(submissions, key=lambda x: x.created_utc, reverse=True)

        submissions = submissions[:max_submission_count*4]
        submissions = random.sample(submissions, min(len(submissions), max_submission_count))
        logger.debug(f'found submissions: {submissions}')
        return submissions

    def _prepare_reply(self, title: str, text: str, num_return_sequences: int = 1) -> [str]:
        replies = self.model.generate_text(title, text, num_return_sequences)
        processed_replies = [process_output(reply) for reply in replies]
        return processed_replies, replies

    def _make_reply(self, submission: Submission, reply_text: str = None) -> None:
        url = submission.url
        title = submission.title
        selftext = submission.selftext

        if reply_text is None:
            reply_text, generated_text = self._prepare_reply(title, selftext)
            reply_text, generated_text = reply_text[0], generated_text[0]
        

        if not reply_text or reply_text == ' ': 
            logger.debug(f'FAIL: generated reply to submission={title} was very bad ({generated_text})')
            return

        if self.test_submission:
            submission = self.reddit.submission(self.reddit.config.custom['test_submission'])
            if not self.is_silent:
                submission.reply(f'{url}\n\n{title}\n\n{selftext}\n\n\nREPLY:\n\n' + reply_text)
        else:
            if not self.is_silent:
                submission.reply(reply_text)

        logger.debug(f'made reply: submission={title} reply={reply_text}')
    
    def _choose_reply_manual(self, submission: Submission, reply_texts: str) -> str:
        url = submission.url
        title = submission.title
        selftext = submission.selftext
        print('SUBMISSION DATA:')
        print(f'{url}\n{title}\n{selftext}')
        print('Possible options:')
        for i, reply_text in enumerate(reply_texts):
            print(f'{i}) {reply_text}')
            print('-----')
        reply_index = input('Your choice:')
        try:
            reply_index = int(reply_index)
            if reply_index < 0 or reply_index >= len(reply_texts):
                return None 
            print(f'selected option: {reply_index}')
            return reply_texts[reply_index]
        except ValueError:
            return None

    def _make_replies(self, submissions: List[Submission]) -> None:
        if not self.manual:
            for submission in submissions:
                self._make_reply(submission)
                time.sleep(SLEEP_BETWEEN_REPLIES) 
        else:
            reply_texts_list = []
            for submission in submissions:
                reply_text, _ = self._prepare_reply(submission.title, submission.selftext, 10)
                reply_texts_list.append(reply_text)
            replies = []
            for submission, reply_texts in zip(submissions, reply_texts_list):
                replies.append(self._choose_reply_manual(submission, reply_texts))
            for submission, reply_text in zip(submissions, replies):
                if reply_text is None:
                    continue
                self._make_reply(submission, reply_text)
                time.sleep(SLEEP_BETWEEN_REPLIES) 

    def run(self) -> None:
        subreddits = self._get_subreddits()
        if self.manual:
            submissions = self._get_submissions(subreddits, max_submission_count=10)
        else:
            submissions = self._get_submissions(subreddits)
        self._make_replies(submissions)
