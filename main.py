from multiprocessing import Pool
import subprocess


def run_process(process_desc):
    subprocess.run(["python", "train.py", "-p", process_desc[0], "-k", process_desc[1],
                    "-a", process_desc[2], "-t", process_desc[3], "-l", process_desc[4]],
                   stdout=subprocess.DEVNULL)


for k in [0, 25, 50, 60, 70, 80, 90, 95, 97, 99]:
    # Put these inside a directory
    train_acc_file_0 = "./train_acc_k_{}_strategy_0".format(k)
    train_acc_file_1 = "./train_acc_k_{}_strategy_1".format(k)
    test_acc_file_0 = "./test_acc_k_{}_strategy_0".format(k)
    test_acc_file_1 = "./test_acc_k_{}_strategy_1".format(k)
    log_file_0 = "./logs_{}_0.txt".format(k)
    log_file_1 = "./logs_{}_1.txt".format(k)

    print("Starting for K = {}".format(k))

    processes = (["0", str(k), train_acc_file_0, test_acc_file_0, log_file_0],
                 ["1", str(k), train_acc_file_1, test_acc_file_1, log_file_1])

    with Pool(processes=2) as p:
        p.map(run_process, processes)

