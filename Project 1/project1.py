from collections import Counter, defaultdict
from dataclasses import dataclass
from operator import attrgetter
import math
import numpy as np
import pandas as pd
import argparse

def cal_entropy(labels):
    """Calculate the entropy"""
    total_count = len(labels)
    probs = [count / total_count for count in Counter(labels).values()]
    return sum(p * math.log(1/p, 2) for p in probs if p > 0)

def partition_attr(inputs, attribute):
    """Partition based on the attribute"""
    partitions = defaultdict(list)
    for input in inputs:
        key = getattr(input, attribute)
        partitions[key].append(input)
    return partitions

def cal_partition_entropy(inputs, attribute, label_attribute):
    """Calculate the entropy of the partition"""
    partitions = partition_attr(inputs, attribute)

    labels = [[getattr(input, label_attribute) for input in partition]
              for partition in partitions.values()]

    total_count = sum(len(label) for label in labels)

    return sum(cal_entropy(label) * len(label) / total_count for label in labels)

@dataclass
class Gain:
    id: str
    attribute: str
    gain: float

@dataclass
class F:
    id: str
    attribute: str
    f: float

def cal_best_partition(df, partition):
    df.columns = df.columns.str.strip() # strip whitspace
    instances = list(df.itertuples(name='Instance', index=True)) # pandas df to namedtuple
    split_attributes = list(df.columns.values[:len(df.columns)-1]) # list of split attribute
    target_attribute = df.columns.values[-1] # target attribute
    
    # Build a dictionay of the partition
    partitions = defaultdict(list)
    for key, value in partition.items():
        inputs = [input for input in instances if input.Index+1 in value]
        for input in inputs:
            partitions[key].append(input)

    f_results = []
    for key, inputs in partitions.items():
        # Calculate the entopy
        labels = [getattr(input, target_attribute) for input in inputs]
        entropy = cal_entropy(labels)
        print(f'E({key}) = {entropy}')

        # Calculate the conditional entropy and information gain
        gain_results = []
        for attribute in split_attributes:
            conditional_entropy = cal_partition_entropy(inputs, attribute, target_attribute)
            information_gain = entropy - conditional_entropy
            gain_results.append(Gain(key, attribute, information_gain))
            print(f'E({key}|{attribute}) = {conditional_entropy:.6}'.ljust(25),
                  f'G({key},{attribute}) = {information_gain:.6}')
        
        # Calculate F
        max_gain = max(gain_results, key=attrgetter('gain'))
        f =  (len(inputs) / len(df.index)) * max_gain.gain
        f_results.append(F(max_gain.id, max_gain.attribute, f))
        print(f'F_{key} = {f}')

    best_split = max(f_results, key=attrgetter('f'))

    if best_split.f !=0 :
        # Partition based on the attribute
        parts = partition_attr(partitions[best_split.id], best_split.attribute)
        p1 = {key: value for key, value in partition.items() if key is not best_split.id}
        p2 = {best_split.id+str(count): [value.Index+1 for value in values]
              for count, values in enumerate(parts.values(), start=1)}
        partition = p1 | p2
        p = ','.join(list(p2.keys()))
        print(partition)
        print(f'Partition {best_split.id} was replaced with partitions {p} using Feature {best_split.attribute}')
    else :
        # Partition is classified
        print(partition)
        print(f'Partition was not replaced')
    return partition

if __name__ == "__main__":
    # Read user input
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type = str, default = 'test.csv', help = "Input csv file name of dataset")
    parser.add_argument("--input", type = str, default = 'partition-2.txt', help = "Input partition")
    parser.add_argument("--output", type = str, default = 'partition-3.txt', help = "Output partition")
    args = parser.parse_args()
    dataset = args.dataset
    input_partition_filename = args.input
    output_partition_filename = args.output

    # Read input dataset and partition
    df = pd.read_csv(dataset)
    #print(df) # df.head(2)
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