class PunctNormalizer:
    def __call__(self, sent):
        if len(sent) > 0 and (sent[-1] not in {'!.?'}):
            sent += '.'
        return sent

