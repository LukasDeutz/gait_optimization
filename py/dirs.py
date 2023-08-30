'''
Created on 29 Aug 2023

@author: amoghasiddhi
'''
from pathlib import Path

data_dir = Path('../data/')
fig_dir = Path('../figures/')
log_dir = data_dir / 'logs'
sweep_dir = data_dir / 'sweeps'
exp_dir = Path('../data/experiments')
video_dir = Path('../videos/')

data_dir.mkdir(parents = True, exist_ok = True)
fig_dir.mkdir(parents = True, exist_ok = True)
log_dir.mkdir(parents = True, exist_ok = True)
sweep_dir.mkdir(parents = True, exist_ok = True)
exp_dir.mkdir(parents = True, exist_ok = True)
video_dir.mkdir(parents = True, exist_ok = True)


