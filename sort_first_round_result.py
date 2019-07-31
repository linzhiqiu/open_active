import os, glob, argparse


parser = argparse.ArgumentParser(description='Sort Result file')
parser.add_argument('--result_dir',
                    default="first_round",
                    help='The directory of results')
parser.add_argument('--index',
                    default=0, type=int,
                    help='The index of the accuracy to be sorted. From high to low.')
args = parser.parse_args()

files = glob.glob(os.path.join(args.result_dir, "*.txt"))
for file_path in files:
    with open(file_path, "r") as file:
        lines = file.readlines()
    lines[1:] = sorted(lines[1:], reverse=True, key=lambda x: float(x.split("|")[args.index]))
    with open(file_path, "w+") as file:
        for line in lines: file.write(line)
