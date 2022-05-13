from sklearn import svm
import numpy as np
import pandas as pd
import argparse
import time


if __name__ == "__main__":
    # Read user input
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type = str, default = 'wine_7_50.csv', help = "Training dataset")
    parser.add_argument("--test", type = str, default = 'wine.csv', help = "Test dataset")
    parser.add_argument("--target", type = str, default = 'Target', help = "Target column name")
    args = parser.parse_args()
    train_dataset = args.train
    test_dataset = args.test
    target_col_name = args.target
    # ====================================
    # STEP 1: read the training and testing data.
    
    # Do not change any code of this step.
    train_df = pd.read_csv(train_dataset)
    test_df = pd.read_csv(test_dataset)
    y_train = train_df[target_col_name]
    x_train = train_df.drop([target_col_name], axis = 1)
    y_test = test_df[target_col_name]
    x_test = test_df.drop([target_col_name], axis = 1)

    start_time = time.time()
    # ====================================
    # STEP 2: pre processing
    # Please modify the code in this step.
    print("Pre processing data")
    # you can skip this step, use your own pre processing ideas,
    # or use anything from sklearn.preprocessing

    # The same pre processing must be applied to both training and testing data
    # x_train = x_train / 1.0
    # x_test = x_test / 1.0

    # ====================================
    # STEP 3: train model and find the best parameters.
    # Please modify the code in this step.

    print("---train")
    model = svm.SVC() # this line should be changed with best parameters set up

    model.fit(x_train, y_train)

    # STEP 4: save the model
    # Please modify the code in this step

    # ====================================
    # STEP 5: evaluate model
    # Don't modify the code below.

    print("---evaluate")
    print(" number of support vectors: ", model.n_support_)
    acc = model.score(x_test, y_test)
    print("acc:", acc)
    end_time = time.time()
    print('Total time: ', end_time - start_time, ' seconds.')

   
