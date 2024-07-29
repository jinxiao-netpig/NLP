import concurrent.futures
import subprocess
import time
from threading import Lock
from datetime import datetime

class ScriptRunner:
    def __init__(self, max_workers=2, log_dir='../Logs/tmp'):
        self.max_workers = max_workers
        self.log_dir = log_dir
        self.lock = Lock()
        self.completed_scripts = 0
        self.total_scripts = 0

    def run_script(self, idx, script_info):
        path, work_dir, args = script_info["path"], script_info["work_dir"], script_info["args"]

        with self.lock:
            start_time = datetime.now()
            log_name = start_time.strftime("%Y-%m-%d_%H-%M-%S.log")
            start_time_str = start_time.strftime("%m-%d %H:%M:%S")
            print(f"[{start_time_str}] Completed {self.completed_scripts} / {self.total_scripts} - START - script {idx} where the log is saved in {self.log_dir + '/' + log_name}")
            time.sleep(2.0)  # 防止多线程下 log 命名重复

        args.extend([
            '--log_dir', self.log_dir,
            '--log_name', log_name,
            '--checkpoint', f'../Model/{idx}'  # output path for model and result
        ])
        process = subprocess.run(
            args=['python', path] + args, cwd=work_dir,
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        end_time = datetime.now()
        end_time_str = end_time.strftime("%m-%d %H:%M:%S")
        elapsed_time = end_time - start_time

        total_seconds = int(elapsed_time.total_seconds())  # 获取时间差的总秒数
        hours, remainder = divmod(total_seconds, 3600)     # 将总秒数转换为时分秒
        minutes, seconds = divmod(remainder, 60)

        # 打印执行时间和完成的脚本数量
        with self.lock:
            self.completed_scripts += 1
            print(f"[{end_time_str}] Completed {self.completed_scripts} / {self.total_scripts} -  END  - script {idx} executed in {hours}:{minutes}:{seconds}(H:M:S).")

        return process.returncode

    def run_scripts(self, scripts_info):
        self.completed_scripts = 0
        self.total_scripts = len(scripts_info)
        print(f"Running {len(scripts_info)} scripts in {self.max_workers} threads")
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.run_script, idx, script_info)
                for idx, script_info in enumerate(scripts_info)]

            # 等待所有脚本执行完毕
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                except Exception as exc:
                    print(f"Script generated an exception: {exc}")





