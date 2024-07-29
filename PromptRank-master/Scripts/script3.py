from utils.runner import ScriptRunner

threads = 5              # 最大线程数
log_dir = '../Logs/rating_reg'  # Log 的保存路径

data_path = '../Data/TripAdvisor/reviews.pickle'
index_dir = '../Data/TripAdvisor/1/'

rating_regs = [1e-3, 1e-2, 1e-1, 1e0, 1e1]

models = [
    # work_dir, path, args
    ('../FVG', 'main3.py', ['--rating_model', 'mf', '--rating_reg', f'{rating_reg}'])
    for rating_reg in rating_regs
]

scripts = []
for work_dir, path, args in models:
    scripts.append({
        'path': path,
        'work_dir': work_dir,
        'args': args + ['--data_path', f'{data_path}', '--index_dir', f'{index_dir}']
    })

runner = ScriptRunner(max_workers=threads, log_dir=log_dir)
runner.run_scripts(scripts)
