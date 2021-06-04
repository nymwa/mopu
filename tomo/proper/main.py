from tunimi.proper import ProperGenerator

if __name__ == '__main__':
    pg = ProperGenerator()
    for _ in range(30):
        print(pg())

