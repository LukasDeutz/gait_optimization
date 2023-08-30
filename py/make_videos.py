'''
Created on 29 Aug 2023

@author: amoghasiddhi
'''
# Built-in
from pathlib import Path

# Third-party import
import numpy as np
import h5py

from parameter_scan import ParameterGrid
from simple_worm.frame import FrameSequenceNumpy

# Local imports
from dirs import sweep_dir, video_dir, log_dir 
from worm_studio import WormStudio
from util import load_h5_file

#===============================================================================
# Helper functions
#===============================================================================

def raw_to_FS():
    '''
    Converts h5 raw data to FrameSequences
    '''

    pass

#===============================================================================
# Make videos 
#===============================================================================
    
def make_videos_for_figure_1(h5_a_b):
    '''
    Make videos for figure 1.    
    '''
    #===========================================================================
    # Load raw data
    #===========================================================================
    h5 = h5py.File(sweep_dir / h5_a_b, 'r')    
    PG_filepath = log_dir / h5.attrs['grid_filename']
    PG = ParameterGrid.init_pg_from_filepath(str(PG_filepath))

    a_arr = PG.v_from_key('a')
    b_arr = PG.v_from_key('b')
    
    #===========================================================================
    # Select simulations for which to create videos
    #===========================================================================
    
    b0 = -2
    b_idx = np.abs(b_arr - b0).argmin()  

    for i, a in enumerate(a_arr):
    
        r = h5['FS']['r'][i, b_idx, :]
        d1 = h5['FS']['d1'][i, b_idx, :]
        d2 = h5['FS']['d2'][i, b_idx, :]
        d3 = h5['FS']['d3'][i, b_idx, :]
        
        FS = FrameSequenceNumpy(x=r, e0=d3, e1=d1, e2=d2)
    
        filename = f'video_a={a}_b={b0}.mpg'
            
        WS = WormStudio(FS)
        WS.generate_clip(video_dir / filename, 
            add_trajectory = False, n_arrows = 0.2)
    
    return
            
def make_videos_for_figure_2(h5_f):
    '''
    
    '''
    #===========================================================================
    # Load raw data
    #===========================================================================        
    h5, PG = load_h5_file(h5_f)    
    f_arr = 1.0 / PG.v_from_key('T_c')

    fig_dir = video_dir / Path(h5_f).stem
    fig_dir.mkdir(parents=True, exist_ok=True)

    for i, f in zip(np.arange(len(f_arr), dtype = int)[::5], f_arr[::5]):
    
        r = h5['FS']['r'][i, :]
        d1 = h5['FS']['d1'][i, :]
        d2 = h5['FS']['d2'][i, :]
        d3 = h5['FS']['d3'][i, :]        
        t = h5['t'][:]
        
        FS = FrameSequenceNumpy(x=r, e0=d3, e1=d1, e2=d2)
        setattr(FS, 'times', t)
    
    
        file_path = fig_dir / f'sperm_video_{f}.mpg'
               
        WS = WormStudio(FS)
        WS.generate_clip(
            str(file_path), 
            add_trajectory = False, 
            n_arrows = 0.2)

        return
        
if __name__ == '__main__':
    
    h5_a_b  = ('raw_data_'
        'a_min=-2.0_a_max=3.0_a_step=0.2_'
        'b_min=-3.0_b_max=0.0_b_step=0.2_'
        'A=4.0_lam=1.0_T=5.0_N=250_dt=0.001.h5'
    )

    h5_f_sperm = ('raw_data_rikmenspoel_'
        'f_min=8.0_f_max=56.0_f_step=2.0_'
        'const_A=False_phi=None_T=5_N=250_dt=0.001.h5'
    )


    #make_videos_for_figure_1(h5_a_b)
    
    make_videos_for_figure_2(h5_f_sperm)

        
    
    
