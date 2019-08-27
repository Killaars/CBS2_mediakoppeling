##%%
#import argparse
#
#parser = argparse.ArgumentParser(description='Script to calculate and returns the probability of a match between a new child and the existings CBS articles')
#parser.add_argument('child_file', type=str, help='Filename of the new child article')
#parser.add_argument('cutoff', type=float, help='% cutoff boundary for automatic matches',default=0.9)
#parser.add_argument('nr_matches', type=int, help='Number of matches to return',default=5)
#args = parser.parse_args()
#print(args)

# myls.py
# Import the argparse library
import argparse

import os
import sys

# Create the parser
my_parser = argparse.ArgumentParser(description='List the content of a folder')
my_parser.add_argument('child_file',
                       type=str,
                       help='Filename of the new child article')
my_parser.add_argument('cutoff',
                       type=float,
                       help='% cutoff boundary for automatic matches')
my_parser.add_argument('nr_matches',
                       type=int,
                       help='Number of matches to return')

args = my_parser.parse_args()

input_path = args.child_file

print(input_path)