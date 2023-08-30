'''
Created on 14 Feb 2023

@author: lukas
'''
# Built-in
from pathlib import Path
import json
import dill

# Third-party imports
import numpy as np
import h5py
import pint
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, ScalarFormatter
from scipy.interpolate import UnivariateSpline

from parameter_scan.util import recover_hashed_dict
from parameter_scan import ParameterGrid

from util import *
from dirs import exp_dir, fig_dir, video_dir

ureg = pint.UnitRegistry()

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['text.usetex'] = True

# PNAS wants numbers, letters and symbols to be larger 
# than 6 points and smaller than 12 points.
# Main text used 11-point font
rcParams['axes.labelsize'] = 10
rcParams['xtick.labelsize'] = 8
rcParams['ytick.labelsize'] = 8
rcParams['font.size'] = 8
rcParams['legend.fontsize'] = 10
panel_label_fz = 12
textbox_fz = 12

# Figure width should match column width
cm_to_inch = 0.4

PNAS_single_column_width = 8.7 # cm 
PNAS_double_column_width_medium = 11.4 # cm
PNAS_double_column_width_large = 17.8 # cm
PNAS_page_height = 9 # inches

golden_ratio = 0.5*(1 + np.sqrt(5)) 
fig_single_column_width = cm_to_inch * PNAS_single_column_width  
fig_double_column_width_medium = cm_to_inch * PNAS_double_column_width_medium
fig_double_column_width_large  = cm_to_inch * PNAS_double_column_width_large


#===============================================================================
# General layout
#===============================================================================

figure_format = '.eps'
    
cm_dict = {
    'U': plt.cm.plasma,
    'D': plt.cm.hot,
    'W': plt.cm.hot,
    'k_norm': plt.cm.winter,
    'A_max': plt.cm.hot,
    'D_I_over_D': plt.cm.seismic,
    'W_over_U': plt.cm.cool
}
    
label_dict = {
    'mu': r'Viscosity $\mu$ [Pa ]',
    'lam': r'Wavelength $\lambda$ [Pa]',
    'f': r'Frequency $f$ [Hz]'
}

#===============================================================================
# Plot figures
#===============================================================================

def plot_figure_1(
        h5_filename: str,
        show = False):
    '''    
    Plots
    - A: parameter regions  
    - B: Filter cartoon
    - C: Normalized swimming speed U_star 
    - D: Noramlized curvature amplitude
    - E: Efficiency W/S 
    
    as a function of dimensionless time scale ratios a and b    
    '''
    
    #===========================================================================
    # Load simulation data
    #===========================================================================
    
    h5, PG = load_h5_file(h5_filename)
        
    U_star = h5['U'][:].T
    # k_norm = h5['k_norm'][:].T
    A0 = PG.base_parameter['A']    
    A_max = h5['A_max'][:].T    
    W_star = h5['energies']['W'][:].T
            
    #===============================================================================
    # Create a and b mesh 
    #===============================================================================    
    a_arr = PG.v_from_key('a')
    b_arr = PG.v_from_key('b')
    log_a_arr = np.log10(a_arr)
    log_b_arr = np.log10(b_arr)
    
    a_grid, b_grid = np.meshgrid(a_arr, b_arr)
    log_a_grid, log_b_grid = np.meshgrid(log_a_arr, log_b_arr)

    #===========================================================================
    # GridSpec Layout
    #===========================================================================
    width = fig_double_column_width_large
    height = 2 * width / 3
                
    plt.figure(figsize = (width, height))        

    gs = plt.GridSpec(3, 2, 
        hspace = 0.4,
        wspace = 0.4,
        left=0.1, bottom=0.1, right=0.9, top=0.9)
            
    ax00 = plt.subplot(gs[0,0])
    ax01 = plt.subplot(gs[0,1])
    ax10 = plt.subplot(gs[1,0])
    ax11 = plt.subplot(gs[1,1])    
    ax20 = plt.subplot(gs[2,0])
    ax21 = plt.subplot(gs[2,1])

    #===============================================================================
    # Panel A: Dynamical regions   
    #===============================================================================
    
    # contour lines
    levels1 = [0.1, 0.5, 0.9]    
    contour_ls = ['-', '--', '-']
    
    ax00.contour(log_a_grid, log_b_grid, U_star / U_star.max(), 
        levels = levels1, linestyles = contour_ls, colors = ('k',))        
        
    # Plot arrows to illustrate how dimless time scale ratios 
    # a and b change as a function of the phyical parameters
    a0, b0 = 0.5, -2.2
    a1, b1  = 2.8, -0.1
    a2, b2 = a0 + 1.8, b0 + 1.8
      
    arrow_size = 15
            
    ax00.plot(a0, b0, 's', c = 'k', ms = 7)
    ax00.annotate("", xy=[a1, b0], xytext=[a0, b0], 
        arrowprops=dict(arrowstyle='->', mutation_scale= arrow_size))
    ax00.annotate("", xy=[a0, b1], xytext=[a0, b0], 
        arrowprops=dict(arrowstyle='->', mutation_scale= arrow_size))
    ax00.annotate("", xy=[a2, b1], xytext=[a0, b0], 
        arrowprops=dict(arrowstyle='->', mutation_scale= arrow_size))
    
    # Arrow annotation
    ax00.text(a0 + 0.6, b0 - 0.25, r'$\mu \uparrow$', 
        weight = 'bold', fontsize = 12)
    ax00.text(a0 - 0.3, 0.6*(b0 + b1), r'$\eta \uparrow$', 
        weight = 'bold', fontsize = 12)
    ax00.text(a0 + 0.35 , b0 + 0.125, r'$ E \downarrow f \uparrow$', 
        rotation = 45, fontsize = 10)
    
    # Region labels
    ax00.text(a0 - 0.25, b0 - 0.1, 'I', weight = 'bold', 
        fontsize = panel_label_fz)
    ax00.text(a0 + 1.0, b0 + 0.7, 'II', weight = 'bold', 
        fontsize = panel_label_fz)    
    ax00.text(a2, b2 - 0.2, 'III', weight = 'bold', 
        fontsize = panel_label_fz)
        
    #===============================================================================
    # Panel B: Filter cartoon 
    #===============================================================================

    ax01.spines['top'].set_visible(False)
    ax01.spines['right'].set_visible(False)
    ax01.spines['left'].set_visible(False)
    ax01.spines['bottom'].set_visible(False)
    ax01.set_xticks([])
    ax01.set_yticks([])
    
    #===============================================================================
    # Panel C: Swimming Speed 
    #===============================================================================    
    levels0 = np.arange(0, 1.01, 0.2)
    
    CS = ax10.contourf(a_grid, b_grid, U_star / U_star.max(), 
        levels = levels0, cmap = cm_dict['U'])
    ax10.contour(a_grid, b_grid, U_star / U_star.max(), 
        levels = levels1, linestyles = contour_ls, colors = ('k',))        
    cbar = plt.colorbar(CS, ax = ax10, orientation = 'vertical')
    cbar.set_label(r'$U / \max(U)$')

    ax10.set_xscale('log')
    ax10.set_yscale('log')

    # Add invisible colorbar to first panel to ensure equale size
    cbar = plt.colorbar(CS, ax = ax00, orientation = 'vertical')
    cbar.outline.set_visible(False)
    cbar.ax.set_visible(False)
    
    #===============================================================================
    # Panel D: Speed along a, b trajectories
    #===============================================================================
         
    W_star = W_star
    W_over_S = W_star / U_star
        
    CS = ax11.contourf(a_grid, b_grid, np.log10(W_over_S / W_over_S.max()), 
        levels = len(levels0), cmap = cm_dict['W'])
    ax11.contour(a_grid, b_grid, U_star / U_star.max(), 
        levels = levels1, linestyles = contour_ls, colors = ('k',))        
    
    ax11.set_xscale('log')
    ax11.set_yscale('log')
    
    cbar = plt.colorbar(CS, ax = ax11, orientation = 'vertical')
    cbar.set_label(r'$W / S$')

    #===============================================================================
    # Panel E: Curvature Amplitude 
    #===============================================================================

    CS = ax20.contourf(a_grid, b_grid, A_max / A0, 
        levels = len(levels0), cmap = cm_dict['A_max'])
    ax20.contour(a_grid, b_grid, U_star / U_star.max(), 
        levels = levels1, linestyles = contour_ls, colors = ('k',))        
    ax20.set_xscale('log')
    ax20.set_yscale('log')

    cbar = plt.colorbar(CS, 
        ax = ax20, orientation = 'vertical',
        ticks = levels0)
    cbar.set_label(r'$A / A_0$')

    # Curvature error
    # CS = ax10.contourf(a_grid, b_grid, k_norm / A, 
    #     levels = len(levels0), cmap = cm_dict['k_norm'])
    # ax10.contour(a_grid, b_grid, U_star / U_star.max(), 
    #     levels = levels1, linestyles = contour_ls, colors = ('k',))        
    #
    # cbar = plt.colorbar(CS, ax = ax10, orientation = 'horizontal')
    # cbar.set_label(r'$L_2(\kappa - \kappa_0)  / A_0$')

    #===============================================================================
    # Panel Energy per unit distance
    #===============================================================================
                        
    P_star = 20.0          
    W_star = W_star + P_star
    W_over_U = W_star / U_star
        
    CS = ax21.contourf(a_grid, b_grid, np.log10(W_over_U / W_over_U.max()), 
        levels = len(levels0), cmap = cm_dict['W'])
    ax21.contour(a_grid, b_grid, U_star / U_star.max(), 
        levels = levels1, linestyles = contour_ls, colors = ('k',))        
    
    ax21.set_xscale('log')
    ax21.set_yscale('log')
    #
    cbar = plt.colorbar(CS, ax = ax21, orientation = 'vertical')
    cbar.set_label(r'$W / S$')
    
    #===========================================================================
    # Layout
    #===========================================================================
            
    ax00.set_title('A', loc = 'left', fontsize = panel_label_fz)   
    ax01.set_title('B', loc = 'left', fontsize = panel_label_fz)   
    ax10.set_title('C', loc = 'left', fontsize = panel_label_fz)   
    ax11.set_title('D', loc = 'left', fontsize = panel_label_fz)   
    ax20.set_title('E', loc = 'left', fontsize = panel_label_fz)   
    ax21.set_title('F', loc = 'left', fontsize = panel_label_fz)   

    yticks = 10.0**np.array([0, -1, -2, -3])
    xticks = 10.0**np.array([0, 1, 2, 3])

    for ax in [ax10, ax11, ax20, ax21]:                
        ax.set_yticks(yticks)
        ax.set_xticks(xticks)
        ax.set_xlim((1, 10**3))

    # Remove ticks panel A
    ax00.set_xticks([])
    ax00.set_yticks([])
    ax00.set_xlim((0,3))
                    
    for ax in [ax20, ax21]:        
        ax.set_xlabel(r'$a$')        
    
    for ax in [ax10, ax20]:        
        ax.set_ylabel(r'$b$')        
    
    plt.savefig(fig_dir / 'figure_1.png')
    plt.savefig(fig_dir / 'figure_1.svg')
    plt.savefig(fig_dir / 'figure_1.pdf')

    if show:
        plt.show()

    return

#------------------------------Paper figure 2------------------------------------------------ 

def plot_figure_2(h5_a_b, h5_f, h5_f_lam, show = False):    
    '''
    TODO
    '''
    #===========================================================================
    # Load data
    #===========================================================================
    
    # Simulation data
    h5_a_b, PG_a_b = load_h5_file(h5_a_b)
    h5_f, PG_f = load_h5_file(h5_f)
    h5_f_lam, PG_f_lam = load_h5_file(h5_f_lam)

    a_arr = PG_a_b.v_from_key('a')
    b_arr = PG_a_b.v_from_key('b')
    log_a_arr = np.log10(a_arr)
    log_b_arr = np.log10(b_arr)

    a_grid, b_grid = np.meshgrid(a_arr, b_arr)
    log_a_grid, log_b_grid = np.meshgrid(log_a_arr, log_b_arr)

    # Experimental data     
    with open(exp_dir / 'sperm.json', 'r') as file:    
        sperm_dict = json.load(file)
        sperm_dict = {k: v[0]*ureg(v[1]) for k, v in sperm_dict.items()}
        
    L0 = sperm_dict['L0']
    R = sperm_dict['R']
    B_max = sperm_dict['B_max']
    B_min = sperm_dict['B_min']
    mu = sperm_dict['mu']
    f_avg = sperm_dict['f_avg']
                
    f_arr = np.linspace(sperm_dict['f_min'].magnitude , sperm_dict['f_max'].magnitude, 
        int(1e2)) * ureg.hertz

    # operating point from experimental data
    c_t = RFT(R, L0)[0]
    
    tau_min =  c_t * mu * L0**4 / B_max 
    tau_max =  c_t * mu * L0**4 / B_min 

    xi_min = 1e-4 * ureg.second
    xi_max = 1e-3 * ureg.second
    
    #===============================================================================
    # Plotting
    #===============================================================================

    w = PNAS_single_column_width
    h = PNAS_page_height / 3

    fig = plt.figure(figsize=(w,h))
    gs = plt.GridSpec(3, 1)
    ax00 = plt.subplot(gs[0,0])
    ax10 = plt.subplot(gs[1,0])
    ax20 = plt.subplot(gs[2,0])

    #===============================================================================
    # Panel A
    #===============================================================================
    
    # Plot swimming speed contours    
    levels = [0.1, 0.5, 0.9]
    ax00.contour(a_grid, b_grid, h5_a_b['U'][:].T / h5_a_b['U'][:].max(), 
        levels = levels, linestyles = ['-', '--', '-'], colors = ('k',))        

    ax00.set_xscale('log')
    ax00.set_yscale('log')

    ax00.set_xlim(10, 1000)

    # TODO: Plot operational regime
    for f in f_arr:
    
        a_min = tau_min * f
        a_max = tau_max * f
        a_min.ito_base_units()
        a_max.ito_base_units()        
        b_min = xi_min * f
        b_max = xi_max * f
        b_min.ito_base_units()
        b_max.ito_base_units()
    
        ax00.fill_between([a_min, a_max], [b_min, b_min], [b_max, b_max], 
            color ='g', alpha = 0.3)

    #===============================================================================
    # Panel B
    #===============================================================================
    
    # Show regimes
    # Color dots    
    cmap = cm_dict['U']
    
    f_arr = 1.0 / PG_f.v_from_key('T_c')
    U_star_arr = h5_f['U'][:]/h5_f['U'][:].max()
    
    f_I_to_II = f_arr[np.abs(U_star_arr - 0.9).argmin()]
    # f_II_to_III = f_arr[np.abs(U_star_arr - 0.1).argmin()]
    
    colors = [cmap(U) for U in U_star_arr]

    ax10.plot(f_arr, U_star_arr, '-', c = 'k', zorder = 1)
    ax10.scatter(f_arr, U_star_arr, c=colors, marker='o', zorder = 2)

    # Plot vertical lines
    ax10.axvline(x=f_I_to_II, color='k', linestyle='--')

    # sm = mpl.cm.ScalarMappable(cmap=cmap)
    # sm.set_array([])  # Dummy array to define colormap range
    # # Add the colorbar to the axis
    # cbar = plt.colorbar(sm, ax=ax10)
    # cbar.set_label('$U/U_\mathrm{max}$')

    #===============================================================================
    # Panel C 
    # Plot lam for which speed is maximal
    # Plot lam for which energy efficiency is optimal assuming fixed speed    
    #===============================================================================

         
    with open(exp_dir / 'rikmenspoel_1978.json', 'r') as f:
        rikmenspoel__1978_data = json.load(f)
            
    with open(exp_dir / 'rikmenspoel_1978_fit.pkl', 'rb') as f:
        fit_rikmenspoel_1978 = dill.load(f)
    
    lam_fit = fit_rikmenspoel_1978['lam']

    lam_arr = PG_f_lam.v_from_key('lam')
    lam_refined_arr = np.linspace(lam_arr.min(), lam_arr.max(), int(1e3))   

    levels = np.arange(0.5, 0.91, 0.1)
        
    lam_max_arr = np.zeros(len(f_arr))    
    lam_opt_mat = np.zeros((len(levels), len(f_arr)))
                        
    for i, _ in enumerate(f_arr):
    
    
        U_star_norm_arr = h5_f_lam['U'][i, :] / h5_f_lam['U'][i, :].max() 
        W_star_arr = h5_f_lam['energies']['W'][i, :]
                
        U_fit = UnivariateSpline(lam_arr, U_star_norm_arr, s = 0.0)
        W_fit = UnivariateSpline(lam_arr, W_star_arr, s = 0.0)
        
        U_star_norm_refined_arr = U_fit(lam_refined_arr)
        W_star_norm_refined_arr = W_fit(lam_refined_arr) 
                
        for j, level in enumerate(levels):            
            zc_idx_arr = np.where(np.diff(np.sign(U_star_norm_refined_arr - level)))[0]
            
            W_star_min = np.inf
            lam_opt = 0

            for idx in zc_idx_arr:                                 
                
                W_star_level = W_fit(lam_refined_arr[idx]) 
                
                if W_star_level < W_star_min:
                    lam_opt = lam_refined_arr[idx]
                    W_star_min = W_star_level 
                    
            lam_opt_mat[j, i] = lam_opt
                        
        lam_max_arr[i] = lam_refined_arr[U_star_norm_refined_arr.argmax()]
    
    ax20 = plt.subplot(gs[0])
            
    f_fit_arr = np.linspace(
        rikmenspoel__1978_data['f'].magnitude.min(), rikmenspoel__1978_data['f'].magnitude.max(), int(1e3))
    ax20.plot(f_fit_arr, lam_fit(f_fit_arr), '-', c='k')
    ax20.plot(rikmenspoel__1978_data['f'].magnitude, rikmenspoel__1978_data['lam_star'].magnitude, 's', c='k')    
    ax20.plot(f_arr, lam_max_arr, 'x')
            
    #=======================================================================
    # Layout 
    #=====================================================================
    
    
    #===========================================================================
    # Save figure
    #===========================================================================
                                 
    plt.savefig(fig_dir / 'figure_2.png')
    plt.savefig(fig_dir / 'figure_2.svg')
    plt.savefig(fig_dir / 'figure_2.pdf')

    if show:
        plt.show()
    
    return

def plot_figure_3(
        h5_C_xi_mu,
        h5_lam_c,
        show = False):
    '''
    Plots     
    
    -A: Swimming speed mu = Water
    -B: Swimming speed mu = Peanut butter
    -C: Dimensionless mechanical muscle work = Water
    -D: Dimensionless mechanical muscle work = Peanut butter
    '''
    
    #===========================================================================
    # Load data
    #===========================================================================
    
    # Experimental data Fang Yen 2010    
    with open(exp_dir / 'fang_yen_2010.json', 'r') as data_file:    
        data_fang_yen = json.load(data_file)
        data_fang_yen = {k: v[0]*pint.Unit(v[1]) for k, v in data_fang_yen.items()}
    with open(exp_dir / 'fang_yen_2010_fit.pkl', 'rb') as pickle_file:
        fang_yen_fit = dill.load(pickle_file)
        
    log_mu_fit_arr = np.linspace(
        data_fang_yen['log_mu'].magnitude.min(), data_fang_yen['log_mu'].magnitude.max(), int(1e2))

    # Simulation data 1
    h5_C_eta_mu, PG_C_eta_mu = load_h5_file(h5_C_xi_mu)         
    C_arr_1 = PG_C_eta_mu.v_from_key('C')
    eta_arr_1 = PG_C_eta_mu.v_from_key('eta')    
    xi_arr_1 = eta_arr_1 / PG_C_eta_mu.base_parameter['E'].magnitude    
    mu_arr_1 = PG_C_eta_mu.v_from_key('mu')

    # Select one drag coefficient ratio to plot
    C0 = 1.5        
    C_idx_1 = np.abs(C_arr_1 - C0).argmin()
    
    # Simulation data 2
    h5_lam_c, PG_lam_c = load_h5_file(h5_lam_c)         

    C_arr_2 = PG_lam_c.v_from_key('C')
    eta_arr_2 = PG_lam_c.v_from_key('eta')
    log_xi_arr_2 = np.round(np.log10(eta_arr_2.magnitude 
        / PG_lam_c.base_parameter['E'].magnitude), 2)
    mu_arr_2 = PG_C_eta_mu.v_from_key('mu')
    log_mu_arr_2 = np.log10(mu_arr_2)
    c_arr_2 = PG_lam_c.v_from_key('c')
    lam0_arr_2 = PG_lam_c.v_from_key('lam')
    LAM0, _ = np.meshgrid(lam0_arr_2, c_arr_2)
    LAM0 = LAM0.T
        
    # Select C to plot
    idx_C_2 = np.abs(C_arr_2 - C0).argmin()
    # Select time scale xi to plot
    log_xi = -2    
    log_xi_idx = np.abs(log_xi_arr_2 - log_xi).argmin()
    # Select fluid viscosity to plot
    log_mu_min = -3.0
    log_mu_max = +1.0
    
    mu_idx_1 = np.abs(log_mu_arr_2 - log_mu_min).argmin()
    mu_idx_2 = np.abs(log_mu_arr_2 + log_mu_max).argmin()

    U_star = h5_lam_c['U'][idx_C_2, log_xi_idx, mu_idx_1, :, :].T
    W = h5_lam_c['energies']['W'][idx_C_2, log_xi_idx, mu_idx_1, :, :].T
    A = h5_lam_c['A_max'][idx_C_2, log_xi_idx, mu_idx_1, :, :].T
         
    #===============================================================================
    # Create GridSpec 
    #===============================================================================

    fig = plt.figure(figsize = [fig_double_column_width_large, 1.0*fig_double_column_width_large])
    
    # Create the outer GridSpec object with a single row and the specified width_ratios
    outer_gs = plt.GridSpec(3, 1, hspace = 0.3, 
        height_ratios = [1.2, 1.0, 1.0], 
        left = 0.1, right = 0.95, bottom = 0.1, top =  0.95)
            
    row0 = outer_gs[0].subgridspec(2, 2, wspace= 0.4, hspace = 0.1)                 
    row1_row_2 = outer_gs[1:].subgridspec(2, 2, wspace = 0.3, hspace = 0.3)
                
    ax00 = plt.subplot(row0[:, 0])
    ax00_twin1 = ax00.twinx()
    # ax00_twin2 = ax00.twinx()
    # ax00_twin2.spines.right.set_position(("axes", 1.4))
    ax01_upper = plt.subplot(row0[0, 1])
    ax01_upper_twin1 = ax01_upper.twinx()
    # ax01_upper_twin2 = ax01_upper.twinx()
    # ax01_upper_twin2 .spines.right.set_position(("axes", 1.2))

    ax01_lower = plt.subplot(row0[1, 1], sharex = ax01_upper)        
    ax10 = plt.subplot(row1_row_2[0, 0])
    ax11 = plt.subplot(row1_row_2[0, 1])
    ax20 = plt.subplot(row1_row_2[1, 0])
    ax21 = plt.subplot(row1_row_2[1, 1])

    #===============================================================================
    # Panel A Experimental data from Fang Yen (2010). 
    # Wavelength and curvature amplitude over frequency
    #===============================================================================                
    p1_00 = ax00.plot(data_fang_yen['f'].magnitude, data_fang_yen['lam'].magnitude, 's', ms = '3', c='r')[0]
    ax00.plot(fang_yen_fit['f'](log_mu_fit_arr), fang_yen_fit['lam'](log_mu_fit_arr), ls = '-', c='r')
    p2_00 = ax00_twin1.plot(fang_yen_fit['f'](data_fang_yen['log_mu'].magnitude), 
        fang_yen_fit['A'](data_fang_yen['log_mu'].magnitude), 's', ms='3', c = 'b')[0]
    ax00_twin1.plot(fang_yen_fit['f'](log_mu_fit_arr), fang_yen_fit['A'](log_mu_fit_arr), ls = '-', c='b')
        
    #===============================================================================
    # Panel B Experimental data from Fang Yen (2010) 
    # Wavelength and amplitude over fluid viscosity
    #===============================================================================

    p1_01 = ax01_upper.semilogx(10**data_fang_yen['log_mu'].magnitude, data_fang_yen['lam'].magnitude, 's', ms=3, c='r')[0]
    ax01_upper.semilogx(10**log_mu_fit_arr, fang_yen_fit['lam'](log_mu_fit_arr), ls = '-', c='r')
    
    p2_01 = ax01_upper_twin1.semilogx(10**data_fang_yen['log_mu'].magnitude, data_fang_yen['A'].magnitude, 's', ms=3, c='b')[0]
    ax01_upper_twin1.semilogx(10**log_mu_fit_arr, fang_yen_fit['A'](log_mu_fit_arr), ls = '-', c='b')
    
    # p3_01 = ax01_upper_twin2.semilogx(10**data_fang_yen['log_mu'].magnitude, data_fang_yen['f'].magnitude, 's', ms=3, c='k')[0]
    # ax01_upper_twin2.semilogx(10**log_mu_fit_arr, fang_yen_fit['f'](log_mu_fit_arr), ls = '-', c='k')
                        
    # Mark transition region
    
    # log_mu_start = -0.5
    # log_mu_end = 0.5
    #
    # ax00.set_ylim(0.0, 1.9)
    #
    # ax00.fill_between([10**log_mu_start, 10**log_mu_end], 
    #     *ax00.get_ylim(), color='gray', alpha=0.3)
        
    # Create a colormap     
    cmap = plt.get_cmap('jet')
    norm = mpl.colors.LogNorm(vmin=np.min(xi_arr_1), vmax=np.max(xi_arr_1))
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)     
    colors = cmap(norm(xi_arr_1))        
                
    for i, c in enumerate(colors):
        
        D_I = h5_C_eta_mu['energies']['D_I'][C_idx_1, i, :]
        D_F = h5_C_eta_mu['energies']['D_F'][C_idx_1, i, :]    
        D = D_I + D_F 
        D_I_over_D = D_I / D
        
        ax01_lower.semilogx(mu_arr_1, D_I_over_D, 
            label = fr'$\log(\xi)={log_xi}$', c = c)

    # Add colorbar    
    cbar = plt.colorbar(mappable, ax = ax01_lower, orientation = 'vertical')
    cbar.set_label(r'$\xi$')
    cbar.ax.set_yscale('log')
    #cbar.set_ticks([1e-1, 1e-2, 1e-3, 1e-4])
    ax01_lower.set_ylim(0.0, 1.05)

    cbar = plt.colorbar(mappable, ax = ax01_upper, orientation = 'vertical')
    cbar.outline.set_visible(False)
    cbar.ax.set_visible(False)

    # ax01.fill_between([10**log_mu_start, 10**log_mu_end], 
    #     *ax01.get_ylim(), color='gray', alpha=0.3)
           
    # Panel (C, D, E, F)
    axes = [[ax10, ax11],
            [ax20, ax21]]
        
    for i, mu_idx in enumerate([mu_idx_1, mu_idx_2]):
            
        U_star = h5_lam_c['U'][idx_C_2, log_xi_idx, mu_idx, :, :].T
        A = h5_lam_c['A_max'][idx_C_2, log_xi_idx, mu_idx, :, :].T
        W_star = h5_lam_c['energies']['W'][idx_C_2, log_xi_idx, mu_idx, :, :].T           

        axi0 = axes[i][0]
        axi1 = axes[i][1]

        results = comp_optimal_c_and_wavelength(U_star, W_star, A, LAM0, c_arr_2, lam0_arr_2, ax = axi0)
    
        U_star = results['U']
        W_star = results['W']
        lam_grid, c_grid = np.meshgrid(results['lam0_arr'], results['c_arr'])
        colors = results['levels']

        levels = np.arange(0.0, 1.01, 0.1)
        
        CS = axi0.contourf(lam_grid, c_grid, U_star.T / U_star.max(), 
            levels = levels, cmap = cm_dict['U'])
        axi0.contour(lam_grid, c_grid, U_star.T / U_star.max(), 
            levels = results['levels'], linestyles = '-', colors = ('k',))        

        axi0.scatter(results['lam_max'], results['c_max'], marker = 'x', c = 'k')
                
        plt.colorbar(CS, ax = axi0, orientation = 'vertical',
            label = r'$U / \max(U)$')

        CS = axi1.contourf(lam_grid, c_grid, np.log10(W_star.T / W_star.max()), 
            cmap = cm_dict['W'])        
        axi1.contour(lam_grid, c_grid, U_star.T / U_star.max(),   
            levels = levels, linestyles = '--', colors = ('k',))        
        cbar = plt.colorbar(CS, ax = axi1, orientation = 'vertical', 
            label = r'$W/\max(W)$')               
        cbar.locator = plt.MaxNLocator(nbins=4)
        cbar.update_ticks()
                
        axi1.scatter(results['lam_opt_arr'], results['c_opt_arr'], 
            marker ='o', c = colors, edgecolors = 'k', cmap = cm_dict['U'], zorder = 2)
        axi1.scatter(results['lam_max'], results['c_max'], marker = 'x', c = 'k')

                                          
    #------------------------------------------------------------------------------ 
    # Layout         

    ax00.set_title('A', loc = 'left', 
        fontsize = panel_label_fz)    
    # ax00.set_ylabel(r'Freq. $f$ [Hz]', color = p1.get_color())
    # ax00_twin1.set_ylabel(r'Wavelength $\lambda$', color = p2.get_color())
    # ax00_twin2.set_ylabel(r'Amplitude $A$', color = p3.get_color())
    # ax00.set_xlabel(r'Viscosity $\mu$ [Pa]')
    # ax00.set_xticks(10.0**np.array([-3, -2, -1, 0, 1]))
    
    ax00.set_ylabel(r'Wavelength $\lambda$', color = p1_00.get_color())
    ax00_twin1.set_ylabel(r'Amplitude $A$ ', color = p2_00.get_color())
    ax00.set_xlabel(r'Freq. $f$ [Hz]')
        
    ax00.tick_params(axis='y', colors=p1_00.get_color())   
    ax00_twin1.tick_params(axis='y', colors=p2_00.get_color())   
    ax00_twin1.set_yticks([3, 4, 5, 6])
                
    ax01_upper.tick_params(axis='y', colors=p1_01.get_color())        
    ax01_upper_twin1.tick_params(axis='y', colors=p2_01.get_color())
    ax01_upper_twin1.set_yticks([3, 4, 5, 6])
    # ax01_upper_twin2.tick_params(axis='y', colors=p3_01.get_color())
    #ax01_upper_twin2.set_yticks([3, 5, 7])

    ax01_upper.set_title('B', loc = 'left', 
        fontsize = panel_label_fz)    
    ax01_upper.set_ylabel('$\lambda$', color = p1_01.get_color())
    ax01_upper_twin1.set_ylabel('$A$', color = p2_01.get_color())
    # ax01_upper_twin2.set_ylabel('$f$ [Hz]', color = p3_01.get_color())
    
    ax01_lower.set_ylabel(r'$D_\mathrm{I}/ D_\mathrm{F}$')
    ax01_lower.set_xlabel(r'Viscosity $\mu$ [Pa]')
    
    ax01_upper.set_xticks([])
    ax01_lower.set_xticks(10.0**np.array([-3, -2, -1, 0, 1]))        
    ax01_lower.set_xlim(10**-3, 10**2)

    ax10.set_title('C', loc = 'left', fontsize = panel_label_fz)   
    ax11.set_title('D', loc = 'left', fontsize = panel_label_fz)   
    ax20.set_title('E', loc = 'left', fontsize = panel_label_fz)   
    ax21.set_title('F', loc = 'left', fontsize = panel_label_fz)   

    ax10.set_xticks([0.5, 1.0, 1.5, 2.0])
    ax11.set_xticks([0.5, 1.0, 1.5, 2.0])
    ax10.set_xticklabels([])
    ax11.set_xticklabels([])

    ax20.set_xticks([0.5, 1.0, 1.5, 2.0])
    ax21.set_xticks([0.5, 1.0, 1.5, 2.0])

    ax10.set_ylabel('Shape factor $c$')
    ax20.set_ylabel('Shape factor $c$')
    ax20.set_xlabel('Wavelength $\lambda$')
    ax21.set_xlabel('Wavelength $\lambda$')
          
    plt.savefig(fig_dir / 'figure_3.png')
    plt.savefig(fig_dir / 'figure_3.svg')
    plt.savefig(fig_dir / 'figure_3.pdf')
    
    if show:                                                      
        plt.show()
    
    plt.close(fig)
    
    return

def plot_figure_4():
    # TODO
    
    pass
    
if __name__ == '__main__':

    #===============================================================================
    # Simulation data filenames 
    #===============================================================================    
    
    h5_lam_a_b  = ('analysis_'
        'lam_min=0.6_lam_max=2.0_lam_step=0.2_'
        'a_min=0.0_a_max=4_a_step=0.2_'
        'b_min=-3_b_max=0_b_step=0.2_'
        'c=1.0_T=5.0_N=250_dt=0.001.h5'
    )

    h5_c_a_b = ('analysis_'
        'c_min=0.4_c_max=1.4_c_step=0.2_'
        'a_min=0.0_a_max=4_a_step=0.2_'
        'b_min=-3_b_max=0_b_step=0.2_'
        'lam=1.0_T=5.0_N=250_dt=0.001.h5')

    h5_mu_a_b = ('analysis_fang_yeng_'
        'mu_min=-3.0_mu_max=1.0_mu_step=1.0_'
        'c_min=0.4_c_max=1.4_c_step=0.1_'
        'lam_min=0.5_lam_max=2.0_lam_step=0.1'
        '_T=5.0_N=250_dt=0.001.h5')

    h5_C_xi_mu_fang_yen = ('analysis_fang_yeng_'
        'C_min=1.5_C_max=4.0_C_step=0.5_'
        'xi_min=-4.0_xi_max=-1.0_xi_step=0.5_'
        'mu_min=-4.0_mu_max=2.0_mu_step=0.5_'
        'T=5.0_N=250_dt=0.001.h5')
    
    h5_C_xi_mu_c_lam = ('analysis_fang_yeng_'
        'C_min=1.5_C_max=4.0_C_step=0.5_'
        'mu_min=-3.0_mu_max=2.0_mu_step=0.5_'
        'eta_min=-3.0_eta_max=-1.0_eta_step=0.5_'
        'c_min=0.4_c_max=1.6_c_step=0.1_'
        'lam_min=0.4_lam_max=2.0_lam_step=0.1_'
        'T=5.0_N=250_dt=0.001.h5')
    
    h5_C_xi_mu = ('analysis_fang_yeng_C_min=1.5_C_max=2.0_C_step=0.1_'
        'xi_min=-2.0_xi_max=-1.5_xi_step=0.1_'
        'mu_min=-4.0_mu_max=2.0_mu_step=0.5_'
        'T=5.0_N=250_dt=0.001.h5'
    )    

    h5_C_xi_mu = ('analysis_fang_yeng_C_min=1.5_C_max=4.0_C_step=0.5_'
        'xi_min=-4.0_xi_max=-1.0_xi_step=0.5_'
        'mu_min=-4.0_mu_max=2.0_mu_step=0.5_'
        'T=5.0_N=250_dt=0.001.h5'
    )

    h5_xi_mu_c_lam = ('analysis_fang_yeng_'
        'xi_min=-2.0_xi_max=-1.5_xi_step=0.1_'
        'mu_min=-4.0_mu_max=2.0_mu_step=1.0_'
        'c_min=0.4_c_max=1.6_c_step=0.1_'
        'lam_min=0.4_lam_max=2.0_lam_step=0.2_'
        'T=5_N=250_dt=0.001.h5')    

    h5_xi_mu_c_lam = ('analysis_fang_yeng_'
        'xi_min=-2.0_xi_max=-1.5_xi_step=0.1_'
        'mu_min=-4.0_mu_max=2.0_mu_step=1.0_'
        'c_min=0.4_c_max=1.6_c_step=0.1_'
        'lam_min=0.4_lam_max=2.0_lam_step=0.2_'
        'phi=c_elegans_T=5_N=250_dt=0.001.h5'
    )    

    h5_a_b_sperm = ('analysis_'
        'a_min=0.0_a_max=3.0_a_step=0.2_'
        'b_min=-3.0_b_max=-1.0_b_step=0.2_'
        'A=6.9_lam=0.7_T=5.0_N=250_dt=0.001.h5'
    )
    
    h5_f_sperm = ('analysis_rikmenspoel_'
        'f_min=8.0_f_max=56.0_f_step=2.0_'
        'const_A=False_phi=None_T=5_N=250_dt=0.001.h5')    
    

    h5_f_lam_sperm = ('analysis_rikmenspoel_'
        'f_min=5.0_f_max=55.0_f_step=5.0_'
        'lam_min=0.5_lam_max=2.0_lam_step=0.05_'
        'phi=None_T=5_N=250_dt=0.001.h5'
    )


    #===========================================================================
    # Plotting
    #===========================================================================
     
    #plot_figure_1(h5_a_b)
    
    plot_figure_2(h5_a_b_sperm, h5_f_sperm, h5_f_lam_sperm, show = True)
    # plot_figure_3(h5_C_xi_mu, h5_C_xi_mu_c_lam)
    #plot_figure_4()
    #plot_figure_5()
      
    print('Finished!')
