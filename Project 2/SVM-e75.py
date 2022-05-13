from sklearn import svm
import numpy as np
import pandas as pd
import argparse
import time
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import joblib


if __name__ == "__main__":
    # Read user input
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type = str, default = 'wine_1_75.csv', help = "Training dataset")
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

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # ====================================
    # STEP 3: train model and find the best parameters.
    # Please modify the code in this step.

    param_grid = [
        {'gamma': [0.0001, 0.001, 0.01, 0.1, 1], 'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}
      ]

    model = svm.SVC(kernel="rbf", random_state=1)

    grid_search = GridSearchCV(model, param_grid, cv=5)

    grid_search.fit(x_train, y_train)

    print(grid_search.best_params_)

    print("---train")
    model = svm.SVC(kernel="rbf", **grid_search.best_params_, random_state=1) # this line should be changed with best parameters set up

    model.fit(x_train, y_train)

    # STEP 4: save the model
    # Please modify the code in this step

    joblib.dump(model, "SVM-e75_model.sav")

    # ====================================
    # STEP 5: evaluate model
    # Don't modify the code below.

    print("---evaluate")
    print(" number of support vectors: ", model.n_support_)
    acc = model.score(x_test, y_test)
    print("acc:", acc)
    end_time = time.time()
    print('Total time: ', end_time - start_time, ' seconds.')

   
