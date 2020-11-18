import os

datasets = ['LINUX']
run_command = 'python main_scripts/main_SimGNN.py '
args1 = '--config=configs/SimGNN_config.json '
args2 = '--dataset_name=%s'


def main():
    os.system('pwd')
    for i in range(len(datasets)):
        command = run_command + args1 + args2 % datasets[i]
        os.system(command)


if __name__ == '__main__':
    main()
