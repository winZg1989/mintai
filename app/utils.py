import os, time


def ensure_dirs(*paths: str):
for p in paths:
os.makedirs(p, exist_ok=True)


class Stopwatch:
def __enter__(self):
self.t0 = time.time(); return self
def __exit__(self, *exc):
self.t1 = time.time()
@property
def elapsed(self):
return self.t1 - self.t0