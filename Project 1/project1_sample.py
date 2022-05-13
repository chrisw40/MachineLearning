import math
import numpy as np
import pandas as pd
import argparse

def cal_best_partition(df, partion):
   print(partition)
   print('Partition Z was replaced with partitions Z1,Z2 using Feature A3')
   return partition

if __name__ == "__main__":
    # Read user input
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type = str, default = 'dataset-1.csv', help = "Input csv file name of dataset")
    parser.add_argument("--input", type = str, default = 'partition-2.txt', help = "Input partition")
    parser.add_argument("--output", type = str, default = 'partition-3.txt', help = "Output partition")
    args = parser.parse_args()
    dataset = args.dataset
    input_partition_filename = args.input
    output_partition_filename = args.output

    # Read input dataset and partition
    df = pd.read_csv(dataset)
    print(df.head(2))
    outfile = open(output_partition_filename,"w")
    partition = {}
    with open(input_partition_filename) as f:
        for line in f.readlines():
            word = line.split(' ')
            key = word[0]
            value =[]
            for i in range(1,len(word)):
                value.append(int(word[i]))
            partition[key] = value

    # processing
    result_partition = cal_best_partition(df, partition)

    # writing output
    for key, value in result_partition.items():
        value_str=''
        for val in value:
            value_str = value_str+' '+ str(val)
        line = str(key)+' '+value_str+'\n'
        outfile.writelines(line)
    outfile.close()