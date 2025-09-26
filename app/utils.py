# app/utils.py
import os, time

def ensure_dirs(*paths: str):
    for p in paths:
        os.makedirs(p, exist_ok=True)

class Stopwatch:
    def __enter__(self):
        self.t0 = time.time()
        self.t1 = None  # 初始化 t1
        return self
        
    def __exit__(self, *exc):
        self.t1 = time.time()
        
    @property
    def elapsed(self):
        if self.t1 is None:
            # 如果还没有调用 __exit__，返回当前时间与开始时间的差
            return time.time() - self.t0
        return self.t1 - self.t0