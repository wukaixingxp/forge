#!/usr/bin/env python3
import os
import torchtitan

exp_dir = os.path.join(os.path.dirname(torchtitan.__file__), 'experiments')
rl_dir = os.path.join(exp_dir, 'rl')
print(f'RL dir exists: {os.path.exists(rl_dir)}')
if os.path.exists(rl_dir):
    print(f'Contents: {os.listdir(rl_dir)}')
    unified_dir = os.path.join(rl_dir, 'unified')
    if os.path.exists(unified_dir):
        print(f'Unified dir exists: True')
        print(f'Unified contents: {os.listdir(unified_dir)}')
