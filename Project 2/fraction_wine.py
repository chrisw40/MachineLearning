import os
import numpy as np
import pandas as pd
import argparse

def select_random_p_from_n(p, n) :
  permutation = np.random.permutation(n)
  k = int(p*n + 0.5)
  firstk = permutation[:k]
  return firstk

def out_name(in_name, p, seed) :
    s = str(seed) if seed >= 0 else ""
    r = str(int(p*100))
    basename = os.path.basename(in_name)
    name, extension = os.path.splitext(basename)
    return(name + "_" + s + "_" + r + extension)

if __name__ == "__main__":
    # Read user input
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type = str, default = 'wine.csv', help = "Input dataset")
    parser.add_argument("--frac", type = float, default = 0.5, help = "Fraction of the dataset")
    parser.add_argument("--seed", type = int, default = -1, help = "Seed for random data generation")
    args = parser.parse_args()
    dataset = args.dataset
    p = args.frac
    seed = args.seed
    if seed >=0:
      print('Set the seed to ', seed, '.')
      np.random.seed(seed)
    # Read dataset  
    df = pd.read_csv(dataset)
    num_lines = df.shape[0]
    selection = select_random_p_from_n(p, num_lines)
    df_select = df.iloc[selection]
    print(df_select.head(5))
    outfile_name = out_name(dataset,p, seed)
    print('Output file is', outfile_name)
    df_select.to_csv(outfile_name, index=False)
    
