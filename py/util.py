# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize

from parameter_scan import ParameterGrid

# Local imports
from dirs import sweep_dir, log_dir


def load_h5_file(filename):
    '''
    Loads hdf5 simulation file
    '''

    h5_filepath = sweep_dir / filename
    h5 = h5py.File(str(h5_filepath), 'r')    
    PG_filepath = log_dir / h5.attrs['grid_filename']
    PG = ParameterGrid.init_pg_from_filepath(str(PG_filepath))

    return h5, PG

def RFT(R, L0):
    '''
    Calculates dimensionless drag coefficients as predicted by 
    resistive-force theory from geometric parameters     
    '''            
    a = 2*R/L0 # slenderness parameter        
    
    # Linear drag coefficients
    c_t = 2 * np.pi / (np.log(2/a) - 0.5)
    c_n = 4 * np.pi / (np.log(2/a) + 0.5)

    # Angular drag coefficients
    y_t = 0.25 * np.pi * a**2
    y_n = np.pi * a**2
                      
    D = y_t / c_t 
    C = c_n / c_t 
    Y = y_n / y_t 
        
    return c_t, C, D, Y

def comp_optimal_c_and_wavelength(U, W, A, LAM, c_arr, lam0_arr,
        levels = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        optimize = 'W',
        ax = None):
        '''
        Finds the shape factor c and  wavelength lambda which 
        minimizes the mechanical muscle work on the contour lines 
        of equal swimming speed 
        
        To achieve a swimming speed U < U_max, which c and lambda 
        requires the least energy                                   
        '''
                
        # Interpolate swimming speed surface with Spline
        U_interp = RectBivariateSpline(lam0_arr, c_arr, U)    
        # Interpolate mechanical work surface with Spline        
        W_interp = RectBivariateSpline(lam0_arr, c_arr, W)
        # Interpolate curvature amplitude
        A_interp = RectBivariateSpline(lam0_arr, c_arr, A)
        # Interpolate real wavelength
        LAM_interp = RectBivariateSpline(lam0_arr, c_arr, LAM)

        # Define finer grid
        lam0_arr = np.linspace(lam0_arr.min(), lam0_arr.max(), 100)
        c_arr = np.linspace(c_arr.min(), c_arr.max(), 100)
        
        # Remesh interpolated surfaces to finer grid
        U = U_interp(lam0_arr, c_arr)
        W = W_interp(lam0_arr, c_arr)
        A = A_interp(lam0_arr, c_arr)
        LAM = LAM_interp(lam0_arr, c_arr)
                                                                                                                      
        # Use lam and c where U is maximal as initial guess 
        j_max, i_max = np.unravel_index(U.argmax(), U.shape)
        lam_max_0, c_max_0 = lam0_arr[i_max], c_arr[j_max]                    
        
        # Minimize -U_interp to find lam and c where U is maximal
        res = minimize(lambda x: -U_interp(x[0], x[1])[0], [lam_max_0, c_max_0], 
            bounds=[(lam0_arr.min(), lam0_arr.max()), (c_arr.min(), c_arr.max())])
                
        lam0_max, c_max = res.x[0], res.x[1]                  
        A0_max = 2* np.pi * c_max / lam0_max
        
        # Maximum swimming speed U_max 
        U_max = U_interp(lam0_max, c_max)
        # Real curvature amplitude A_max             
        A_max = A_interp(lam0_max, c_max)
        # Real wavelength 
        lam_max = LAM_interp(lam0_max, c_max)

        # Create contours for normalised swimming speed  
        U_over_U_max = U / U_max              
        
        LAM0, C = np.meshgrid(lam0_arr, c_arr)               
        
        if ax is None:       
            fig = plt.figure()     
            ax = plt.subplot(111)
        
        CS = ax.contour(LAM0, C, U_over_U_max.T, levels)            
                
        # Allocate arrays for optima on contours 
        lam0_opt_arr = np.zeros_like(levels)
        lam_opt_arr = np.zeros_like(levels)
        c_opt_arr = np.zeros_like(levels)
        W_c_min_arr = np.zeros_like(levels)
        A_opt_arr = np.zeros_like(levels)
        U_opt_arr = np.zeros_like(levels)
                                                                                
        # Iterate over contour lines    
        for i, contours in enumerate(CS.collections):
            
            # W_min on contour line
            W_c_min = W.max() 
                
            # Iterate over paths which make up the current
            # contour line. If contour lines is a closed curve
            # then it has only one path                 
            for path in contours.get_paths():
    
                if not contours.get_paths():
                    assert False, 'Contour line has not path'
                        
                # Get points which make up the current path                                
                lam0_on_path_arr = path.vertices[:, 0]
                c_on_path_arr = path.vertices[:, 1]
                                                
                if len(lam0_on_path_arr) <= 3:
                    k = len(lam0_on_path_arr)-1
                else:
                    k = 3
                                                                 
                # Compute B-spline representation of contour path
                tck, _ = splprep([lam0_on_path_arr, c_on_path_arr], s=0, k = k)                
                
                # Find minimum work along the contour path
                result = minimize_scalar(lambda u: W_interp.ev(*splev(u, tck)), 
                    bounds=(0, 1), method='bounded')
                
                lam0_min, c_min = splev(result.x, tck)                 
                W_p_min = W_interp.ev(lam0_min, c_min)
                
                # If minimum work on path is smaller than work on any 
                # other path which belongs to the contour then set 
                # it contour minimum
                if W_p_min < W_c_min:
                    
                    W_c_min = W_p_min
                    lam0_opt = lam0_min                    
                    lam_opt = LAM_interp(lam0_min, c_min)
                    c_opt = c_min
                    A_opt = A_interp.ev(lam0_min, c_min)
                    U_opt = U_interp.ev(lam0_min, c_min)
                
            lam0_opt_arr[i] = lam0_opt                            
            lam_opt_arr[i] = lam_opt
            c_opt_arr[i] = c_opt
            A_opt_arr[i] = A_opt
            W_c_min_arr[i] = W_c_min  
            U_opt_arr[i] = U_opt
                                    
        plt.close(fig)
        
        result = {}
        
        # refined mesh
        result['c_arr'] = c_arr
        result['lam0_arr'] = lam0_arr
        result['U'] = U
        result['W'] = W
        result['levels'] = levels
        
        # maximum speed kinematics
        result['lam0_max'] = lam0_max 
        result['lam_max'] = lam_max 
        result['c_max'] = c_max
        result['A0_max'] = A0_max                
        result['A_max']= A_max        
        result['U_max'] = U_max        
        
        # optimal kinematics on contours
        result['W_c_min_arr'] = W_c_min_arr
        result['lam0_opt_arr'] = lam0_opt_arr         
        result['lam_opt_arr'] = lam_opt_arr         
        result['c_opt_arr'] = c_opt_arr        
        result['A0_opt_arr'] = 2 * np.pi * c_opt_arr / lam_opt_arr         
        result['A_opt_arr'] = A_opt_arr
        result['U_opt_arr'] = U_opt_arr
                                            
        return result

def colors_from_contours(a_arr, b_arr, CS):
    '''
    Returns the colours of the contour bands the points (a, b) 
    in a_arr b_arr fall into
    '''
    
    colors = []
            
    for a, b in zip(a_arr, b_arr):
        
        in_path = False
        
        for i, collection in enumerate(CS.collections):
            c = collection.get_facecolor()[0]           
            paths = collection.get_paths()
            
            for path in paths:
                if path.contains_point((a, b)):                    
                    colors.append(rgb2hex(c))
                    in_path = True
                    break
            
            if in_path:
                break
        
        if not in_path:
            assert False
          

    return colors
