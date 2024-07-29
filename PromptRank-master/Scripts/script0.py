from utils.runner import ScriptRunner

threads = 4  # about (memery: 19 GB, GPU: 6.6 GB)
log_dir = '../Logs/peter'

data_paths = [
    '../Data/TripAdvisor',
    '../Data/Amazon/ClothingShoesAndJewelry',
    '../Data/Amazon/MoviesAndTV',
    '../Data/Yelp',
]

models = [
    # work_dir, path, args
    ('../PETER', 'main.py', ['--peter_mask']),
]

data = []
for data_path in data_paths:
    for idx in range(1, 5 + 1):
        data.append((f'{data_path}/reviews.pickle', f'{data_path}/{idx}/'))

scripts = []
for work_dir, path, args in models:
    for data_path, index_dir in data:
        scripts.append({
            'path': path,
            'work_dir': work_dir,
            'args': args + ['--data_path', f'{data_path}', '--index_dir', f'{index_dir}']
        })

runner = ScriptRunner(max_workers=threads, log_dir=log_dir)
runner.run_scripts(scripts)
