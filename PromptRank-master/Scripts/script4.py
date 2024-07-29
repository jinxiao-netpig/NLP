from utils.runner import ScriptRunner

threads = 5              # 最大线程数
log_dir = '../Logs/all'  # Log 的保存路径

# 使用的数据集 ( * 5)
data_paths = [
    '../Data/TripAdvisor',
    '../Data/Amazon/ClothingShoesAndJewelry',
    '../Data/Amazon/MoviesAndTV',
    '../Data/Yelp',
]

# 使用的模型
models = [
    # work_dir, path, args
    # add rating input
    ('../FVG', 'main3.py', ['--rating_model', 'mf']),
    ('../FVG', 'main3.py', ['--rating_model', 'scor']),
    ('../FVG', 'main3.py', ['--rating_model', 'mf', '--fixed_params']),
    ('../FVG', 'main3.py', ['--rating_model', 'scor', '--fixed_params']),
    # no rating input
    ('../FVG', 'main.py', ['--rating_model', 'mf']),
    ('../FVG', 'main.py', ['--rating_model', 'scor']),
    ('../FVG', 'main.py', ['--rating_model', 'mf', '--fixed_params']),
    ('../FVG', 'main.py', ['--rating_model', 'scor', '--fixed_params']),
    # PETER
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
