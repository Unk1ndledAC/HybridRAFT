import argparse
import os
import torch

def print_args(args):
    print("-----------------  ARGS  -----------------")
    for k, v in sorted(vars(args).items()):
        print(f"{k}: {v}")
    print("------------------------------------------")
