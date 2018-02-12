import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('--val', '-v', type=float, required=True)
parser.add_argument('--val', '-v', type=float, default=1.25)
ag = parser.parse_args()
print ag.val