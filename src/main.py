import argparse
import configparser
import random
import logging
import pathlib

from redditbot import RedditBot


# CURRENT_DIR = pathlib.Path(__file__).parent.absolute()
PARENT_DIR = pathlib.Path(__file__).parent.parent.absolute()
HOME_DIR = str(pathlib.Path.home())

FORMAT = '%(levelname)s: %(name)s %(asctime)s %(message)s'
logging.basicConfig(filename=f'{PARENT_DIR}/log.log', level=logging.ERROR, format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is-silent', type=int, choices=[0, 1], default=1, required=False,
                        help='if is-silent = 1 bot will generate replies but not post them')
    parser.add_argument('--test-submission', type=int, choices=[0, 1], default=0, required=False,
                        help='if test-submission = 1 bot will make replies to custom test submission')
    parser.add_argument('--manual', type=int, choices=[0, 1], default=0, required=False,
                        help='if manual = 1 you will choose what to post')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(f'{HOME_DIR}/.config/praw.ini')
    bots = config['DEFAULT']['bots'].split(',')
    bot_id = random.choice(bots)
    bot = RedditBot(bot_id, args.is_silent, args.test_submission, args.manual)
    logger.debug(f'created bot {bot_id}')
    
    bot.run()


if __name__ == '__main__':
    main()
