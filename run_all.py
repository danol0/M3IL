import subprocess

app = 'python train.py'
# List of commands to run
modes = [
    '--mode omic',
    '--mode graph',
    '--mode graph --mil global',
    '--mode graph --mil global --attn_pool 1',
    '--mode graph --mil local',
    '--mode pathomic',
    '--mode pathomic --mil global',
    '--mode pathomic --mil global --attn_pool 1',
    '--mode pathomic --mil local',
    '--mode graphomic',
    '--mode graphomic --mil global',
    '--mode graphomic --mil global --attn_pool 1',
    '--mode graphomic --mil local',
    '--mode pathgraphomic',
    '--mode pathgraphomic --mil global',
    '--mode pathgraphomic --mil global --attn_pool 1',
    '--mode pathgraphomic --mil local',
]

# Run each command in order
for task in ['--task grad', '--task surv']:
    for mode in modes:
        process = subprocess.Popen(f'{app} {task} {mode}', shell=True)
        process.wait()  # Wait for the command to finish before running the next one
