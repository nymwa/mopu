from argparse import ArgumentParser
import tweepy

def gen_api(CK, CS, AK, AS):
    auth = tweepy.OAuthHandler(CK, CS)
    auth.set_access_token(AK, AS)
    api = tweepy.API(auth)
    return api


class Waso:
    def __init__(self, CK, CS, AK, AS):
        self.api = gen_api(CK, CS, AK, AS)

    def tweet(self, x):
        self.api.update_status(x)

