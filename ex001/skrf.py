import argparse
import time
import configparser
import openml
import pandas as pd
import socket
from sys import platform
from math import floor, sqrt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from slacker import Slacker

SlackMSG = False

parser = argparse.ArgumentParser(description="Run on OpenML100-Friendly datasets.")
parser.add_argument("path", metavar="P", type=str, nargs=1, 
                    help="The output file path.")
parser.add_argument("task_id", metavar="T", type=int, nargs=1, 
                    help="An id into the OpenML dataset_ids.")
parser.add_argument("run_i", metavar="R", type=int, nargs=1, 
                    help="The run id.")
parser.add_argument("n_jobs", metavar="nj", type=int, nargs=1, 
                    help="The number of jobs for RF.")

args = parser.parse_args()
task_id = args.task_id[0]
run_i = args.run_i[0]
path = args.path[0]
n_jobs = args.n_jobs[0]

config_file = 'config.dat'


if socket.gethostname() == "Synapse.local": ## Check if I'm on my Mac.
    config_lev = "dev"
else:  ## Else assume we're on the cluster
    config_lev = "default"

### config file for parameters
config = configparser.ConfigParser()
config.read(config_file)

ntrees = int(config[config_lev]['ntrees'])
random_state = int(config[config_lev]['random_state'])
test_size = float(config[config_lev]['test_size'])
lam = float(config[config_lev]['lambda'])

if SlackMSG:
    ### Set up slacker for status updates
    with open(slack_config, 'r') as fp:
        slack_token = fp.readline().strip()
        slack_channel = fp.readline().strip()
    
    slack_config = 'slack.ini'
    slack = Slacker(slack_token)


alg = "skRF"

def getID(tid):

    dataset = openml.datasets.get_dataset(tid)

    # Print a summary
    print("This is dataset '%s', the target feature is '%s'" %
          (dataset.name, dataset.default_target_attribute))
    print("URL: %s" % dataset.url)
    print(dataset.description[:500])
    print("\n\n\n")

    X, Y, attribute_names,_ = dataset.get_data(target=dataset.default_target_attribute)
    
    if Y.dtype.name == "category":
        y = Y.cat.codes
    else:
        y = Y

    return(X,y)



if __name__ == "__main__":
    msg = "`MARCC`:\tRunning from Python:\n" + \
         f"{alg} on OpenML task_id{str(task_id).zfill(6)}"

    done = "Finished:\n" + \
         f"{alg} on OpenML: task_id{str(task_id).zfill(6)}"

    failed = "`Failed`:\t :warning:\n" + \
         f"{alg} on OpenML: task_id{str(task_id).zfill(6)}"

    if SlackMSG:
        slack.chat.post_message(slack_channel, msg)

    print(msg)

    out_file = f"{path}/OUTPUT_{alg}_openml_d_{task_id}_run_{str(run_i).zfill(3)}.csv"
    with open(out_file, "w") as f:
        f.write(f"alg,run_i,openmlDataID,error_rate,train_time,predict_time,"+\
                f"project_time,train_size,test_size,ntrees,"+\
                f"p,d,mtry,lambda\n")


    X,y = getID(task_id)

    RS = random_state + run_i
    try:
        clf = RandomForestClassifier(n_estimators = ntrees,
                max_features = 'sqrt', random_state = RS, n_jobs=n_jobs)

        # Split dataset into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state = RS
        )

        p = X_train.shape[1]
        d = floor(sqrt(X_train.shape[1]))

        project_time = "NA"

        test_size_int = X_test.shape[0]
        train_size_int = X_train.shape[0]

        train_time0 = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - train_time0

        predict_time0 = time.time()
        yhat = clf.predict(X_test)
        predict_time = time.time() - predict_time0

        accuracy = metrics.accuracy_score(y_test, yhat)
        error_rate = 1 - accuracy

        with open(out_file, "a") as f:
            f.write(f"{alg}, {run_i}, {task_id}, {error_rate}, {train_time}, {predict_time}," + \
                    f"{project_time}, {train_size_int}, {test_size_int}," + \
                    f"{ntrees}, {p}, {d}, {floor(sqrt(X_train.shape[1]))}, {lam}\n")

        stats_msg = f"alg:{alg}, err:{round(error_rate,4)}, train_time:{round(train_time,4)}\n"
        print(stats_msg)

        if SlackMSG:
            slack.chat.post_message(slack_channel, done + f"run_no {i}" + stats_msg)

    except:
        with open(out_file, 'a') as f:
            f.write(f"## Failed: alg:{alg}, task_id:{task_id}\n")

        if SlackMSG:
            slack.chat.post_message(slack_channel, failed + f"run_no {run_i}")

        print(failed + f"run_no {run_i}")


