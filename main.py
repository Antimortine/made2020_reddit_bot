import argparse
import configparser
import random
import logging

from redditbot import RedditBot


FORMAT = '%(levelname)s: %(name)s %(asctime)s %(message)s'
logging.basicConfig(filename='log.log', level=logging.ERROR, format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is-silent', type=int, choices=[0, 1], default=1, required=False,
                        help='if is-silent = 1 bot will generate replies but not post them')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read("praw.ini")
    bots = config['DEFAULT']['bots'].split(',')

    bot_id = random.choice(bots)

    bot = RedditBot(bot_id, args.is_silent)
    logger.debug(f'created bot {bot_id}')

    bot.run()


if __name__ == "__main__":
    main()
