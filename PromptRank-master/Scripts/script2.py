from utils.runner import ScriptRunner

threads = 8
log_dir = '../Logs/reg_rating'

data_path = '../Data/TripAdvisor/reviews.pickle'
index_dir = '../Data/TripAdvisor/1/'

regs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0]

models = [
    ('../RP', 'main2.py', ['--model', 'scor+', '--lp', f'{reg}', '--lq', f'{reg}'])
    for reg in regs
] + [
    ('../RP', 'main2.py', ['--model', 'libmf', '--lp', f'{reg}', '--lq', f'{reg}'])
    for reg in regs
]

cnt = len(models)
idx = 0

scripts = []
for work_dir, path, args in models:
    scripts.append({
        'path': path,
        'work_dir': work_dir,
        'args': args + ['--data_path', f'{data_path}', '--index_dir', f'{index_dir}']
    })

runner = ScriptRunner(max_workers=threads, log_dir=log_dir)
runner.run_scripts(scripts)

