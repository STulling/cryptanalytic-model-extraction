import os


def run_all(model, seed):
    os.system(f'python3 train_models.py {model} {seed}')
    os.system(f'python3 extract.py {model} {seed}')
    os.system(f'python3 check_solution_svd.py {model} {seed}')


seed = 69

for i in range(10, 101, 10):
    for j in range(10, i + 1, 10):
        model = f'{i}-{j}-1'
        run_all(model, seed)
