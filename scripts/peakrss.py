#!/usr/bin/env python3
"""Run a command and report wall time and peak resident memory (Linux/macOS)."""
import resource, subprocess, sys, time
t = time.time()
r = subprocess.run(sys.argv[1:])
ru = resource.getrusage(resource.RUSAGE_CHILDREN)
kb = ru.ru_maxrss if sys.platform != "darwin" else ru.ru_maxrss // 1024
print(f"[peakrss] exit {r.returncode}, wall {time.time()-t:.1f} s, peak RSS {kb/1024:.0f} MB", file=sys.stderr)
sys.exit(r.returncode)
