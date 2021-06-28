class JanSoweliConverter:
    def __init__(self, vocab):
        self.vocab = vocab
        self.jan_id = vocab.indices['jan']
        self.soweli_id = vocab.indices['soweli']

    def __call__(self, source):
        target = []
        for token in source:
            if token == self.jan_id:
                target.append(self.soweli_id)
            elif token == self.soweli_id:
                target.append(self.jan_id)
            else:
                target.append(token)
        return target

