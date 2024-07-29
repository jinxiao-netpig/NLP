from utils.runner import ScriptRunner

threads = 8
log_dir = '../Logs/rating'

data_paths = [
    '../Data/TripAdvisor',
    '../Data/Amazon/ClothingShoesAndJewelry',
    '../Data/Amazon/MoviesAndTV',
    '../Data/Yelp',
]

models = [
    # work_dir, path, args
    ('../RP', 'main.py', ['--rating_model', 'mf']),
    ('../RP', 'main.py', ['--rating_model', 'scor']),
    ('../RP', 'main.py', ['--rating_model', 'mf+scor']),
    ('../RP', 'main.py', ['--rating_model', 'libmf'])
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
