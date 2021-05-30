import re
import sys

def main():
    punct_pattern = re.compile(r' [!,.:?]')
    quot_pattern = re.compile(r'" .* "')

    for x in sys.stdin:
        x = x.strip()
        x = punct_pattern.sub('\\1', x)
        x = quot_pattern.sub('"\\1"', x)
        print(x)

