# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 11:17:42 2016

@author: Charlie

fluvial block model expanded to full long profile, for integration into Landlab.

v1: putting blocks into the working bare-bones version. 
v2: adding lake filling to combat instability
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from landlab import ModelParameterDictionary
from landlab import RasterModelGrid
from landlab.components import FlowRouter as Flow
from landlab.components import FastscapeEroder as Fsc
from landlab import BAD_INDEX_VALUE
from landlab.components import LinearDiffuser as Diff
from landlab.components.flow_routing import DepressionFinderAndRouter as LakeFill
from landlab.io.netcdf import write_netcdf

#import time
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
#from landlab.plot import channel_profile as prf
import sys

#get input file
plot_counter = 1 #plot every __ steps
input_file = './params_2.txt'
inputs = ModelParameterDictionary(input_file)
nrows = inputs.read_int('nrows')
ncols = inputs.read_int('ncols')
dx = inputs.read_float('dx')
leftmost_elev = inputs.read_float('leftmost_elevation')
initial_slope = inputs.read_float('initial_slope')
runtime = inputs.read_float('total_time')
dt = inputs.read_float('dt')
uplift_rate = inputs.read_float('uplift_rate')
nt = int(runtime // dt) # this is how many loops we'll need
uplift_per_step = uplift_rate * dt
k = inputs.read_float('K_sp')

gamma = 10. #10 for block topo

#build the grid:
mg = RasterModelGrid(nrows, ncols, dx)

#create elevation field for the grid
mg.add_zeros('node','topographic__elevation')
z = mg.zeros('node') + leftmost_elev
z += initial_slope*np.amax(mg.node_x) - initial_slope*mg.node_x

#put these values plus roughness into elevation field
#mg['node'][ 'topographic__elevation'] = z + np.random.rand(len(z))/100.
#mg['node']['elevation'] = np.load('initial_config.npy') #read in a pre-organized landscape
elev = np.load('initial_elevs_diffuse.npy')
mg['node']['topographic__elevation'] = elev
#mg['node']['topographic__elevation'][1020] -= 1 FOR LAKE FILL TESTING
elev_r = mg.node_vector_to_raster(elev)
plt.figure(1)
im = plt.imshow(elev_r, cmap=plt.cm.jet, extent=[0,5000,0,5000], origin='lower')  # display a colored image
plt.colorbar(im)
plt.title('Topography')
plt.show()

#set up its boundary conditions (right, bottom, left, top)
mg.set_closed_boundaries_at_grid_edges(True, True, False, True)

# Display initialization message
print('Running ...') 
dens_water = 1000 #kg/m3
g = 9.81 #m/s2
drag_cube = 0.8 #Carling
channel_width = 10 #m
threshold_drainage_area = 500 #unclear on what these units are
mg.add_zeros('node','water_depth')
mg.add_zeros('node','water_velocity')
mg.add_zeros('node','corrected_shear_stress')
mg.add_zeros('node', 'erodibility')
mg.add_zeros('node', 'incision_rate')
mg.add_zeros('node', 'num_blocks') #hopefully will contain array of sizes at a given node, one entry for each block
mg.add_zeros('node', 'f_covered')
mg['node']['erodibility'][:] = k
spatial_step = mg.empty(centering='node')

#instantiate components
fr = Flow(mg, input_file)
lf = LakeFill(mg, input_file)
sp = Fsc(mg, input_file)
hd = Diff(mg, input_file)

tracking_mat = np.zeros((100000, 4), dtype=np.float64) #colums: 0) which node, 1) side length 2) vol 3) sub/emergence
tracking_mat[:, :] = np.nan 
for_slicing = 0 #in case no blocks ever get put in

side_length = 4 #m, block side length
tau_c_br = 10 #Pa, for now
a1 = 6.5
a2 = 2.5
d = 0.1 #z0
tol = 0.01 #m water thickness error allowable

elapsed_time = 0.
keep_running = True
counter = 0 # simple incremented counter to let us see the model advance
while keep_running:
    if elapsed_time + dt > runtime:
        dt = runtime - elapsed_time
        keep_running = False
    #route flow to determine water volume flux in each cell
    _ = fr.route_flow() # route_flow isn't time sensitive, so it doesn't take dt as input
    
    #lake filling stuff here
    try:
        _ = lf.map_depressions()
        #_ = lf.route_flow() #route flow given that lakes may exist
    except AssertionError:
        print 'Be careful, depression filler took too many iterations'
    print counter
    print np.amin(mg['node']['topographic__elevation'])
    print '-----'
    #test = mg['node']['upstream_ID_order']
    #sys.exit('fuck you')
    #turn volume flux into specific discharge by dividing by channel width
    water_vol_flux = mg['node']['water__volume_flux'] #/ 365 / 24 / 3600 #drainage area times rain rate
    water_specific_q = water_vol_flux / channel_width
    
    fluvial_nodes = np.where(mg.at_node['drainage_area'] >= threshold_drainage_area)[0]
    for x in range(len(fluvial_nodes)):
        s = mg['node']['topographic__steepest_slope'][fluvial_nodes[x]]
        #print 'Node ' + str(x) + ' of ' + str(len(fluvial_nodes))
        #put in new blocks: [THIS SECTION SHOULD BE FUNCTIONAL 3/20/16]
        if s < 0.001:
            num_new_blocks = 0
        else:
            #actual code for subtracting block mass from the hillslopes:
            neighbor_nodes = mg.neighbors_at_node[fluvial_nodes[x]] #only finds 4 perpendicular neighbors (order: right, up, left, down)
            if np.any(neighbor_nodes == BAD_INDEX_VALUE):
                num_new_blocks = 0
            else:
                lam_rockfall = mg['node']['incision_rate'][fluvial_nodes[x]] * gamma #GAMMA CONTROLS RATE OF BLOCK INPUT
                num_new_blocks = int(np.random.poisson(lam_rockfall * dt, 1)) #number of new blocks
                block_vol_to_subtract = num_new_blocks * np.power(side_length, 3)
            
                #now need to find a place on the hillslopes to subtract this volume.
                #pseudocode:
                #1) find up-channel and down-channel nodes from current node
                #2) use this info to find channel-perpendicular nodes
                #3) find chain of channel-perpendicular nodes going from immediately channel-perp to some distance away
                #4) smear vol_to_subtract over those nodes (i.e., subtract vol_to_subtract/(cell_area*n_cells) from each cell)            
                
                #get the other four neighbors
                neighbor_supplements = np.array([neighbor_nodes[1] - 1, neighbor_nodes[1] + 1, neighbor_nodes[3] - 1, neighbor_nodes[3] + 1])            
                neighbor_nodes = np.append(neighbor_nodes, neighbor_supplements)            
                
                #find non-fluvial neighbor nodes
                hillslope_neighbors = np.array([])           
                for x1 in range(len(neighbor_nodes)):
                    if neighbor_nodes[x1] not in fluvial_nodes: #these are the hillslope nodes
                        hillslope_neighbors = np.append(hillslope_neighbors, neighbor_nodes[x1]) #add node to array of hillslope nodes
                        
                #now average out volume loss (assumes constant density) over hillslope cells
                if len(hillslope_neighbors) == 0:
                    print 'PROBLEM: NO HILLSLOPE NEIGHBORS'
                else:
                    mg['node']['topographic__elevation'][neighbor_nodes] -= (block_vol_to_subtract / len(hillslope_neighbors)) / np.power(dx, 2)
                    
                
        mg['node']['num_blocks'][fluvial_nodes[x]] += num_new_blocks
        for new in range(0, num_new_blocks):
            try:
                next_entry = max(max(np.where(np.isfinite(tracking_mat[:, 0])))) + 1 #adding one for next open entry
            except ValueError:
                next_entry = 0 #in case there aren't any blocks in the matrix yet
            for_slicing = next_entry + 1 #b/c when you slice it takes the one before the end                
            if for_slicing >= tracking_mat.shape[0]:
                addition = np.empty((1000, 4))
                tracking_mat = np.concatenate((tracking_mat, addition))
            else:
                pass
            tracking_mat[next_entry, 0] = fluvial_nodes[x] #ID of node where block fell in
            tracking_mat[next_entry, 1] = side_length #still a single value for now
            tracking_mat[next_entry, 2] = np.power(tracking_mat[next_entry, 2], 3) #calculate volume
        
        #roughness bisection scheme to get flow depth:
        if s < 0:
            sys.exit("NEGATIVE SLOPE-- KILLING MODEL")
        
        #make flow depth 0 in places under lakes
        if lf.lake_at_node[fluvial_nodes[x]] == True:
            water_specific_q[fluvial_nodes[x]] = 0
        
        rough_iter = 1
        error = 1
        h_up = 100 #upper q limit to test e.g. maximum possible flow height
        h_low = 0 #lowest possible flow height
        while error > tol:
            rough_iter += 1
            if s < 0.0001:
                if (s > 0) & (s < 0.0001):
                    s = 0.0001   #JANKY FIX FOR TALK ANIMATIONS TO BE TAKEN OUT IMMEDIATELY
                else:
                    h = 0.0
                    error = 0
            else:
                if rough_iter > 10000:
                    #print q_mid
                    #print water_specific_q[fluvial_nodes[x]]
                    #print error
                    #print '======'
                    sys.exit('fuck you')
                    print 'MAYDAY: ROUGHNESS CALCULATION MAY BE STUCK'
                else:
                    pass
                h_mid = (h_low + h_up)/2.
                q_mid = h_mid * np.sqrt(g * h_mid * s) * ((a1 * h_mid / d)) / np.sqrt(np.power(h_mid / d, 5/3) + np.power(a1 / a2, 2))
                if water_specific_q[fluvial_nodes[x]] == 0:
                    q_mid = 0
                    h_mid = 0
                error = abs(q_mid - water_specific_q[fluvial_nodes[x]])
                #print q_mid
                #print water_specific_q[fluvial_nodes[x]]
                #print error
                #print '======'
                #print 'water heights:'
                #print h_up
                #print h_mid
                #print h_low
                #print '======'
                if q_mid > water_specific_q[fluvial_nodes[x]] and error > tol:
                    h_up = h_mid
                elif q_mid < water_specific_q[fluvial_nodes[x]] and error > tol:
                    h_low = h_mid
                else:
                    pass
                h = h_mid
                #time.sleep(.1)
                
        #print 'Out of bisection function'    
        if h > 0:
            v = water_specific_q[fluvial_nodes[x]] / h
        else:
            v = 0
        tau_initial = dens_water * g * h * s #shear stress at each node(Pa)
        
        #figure out which blocks are in the cell around the current node:
        is_block_in_cell = tracking_mat[0:for_slicing, 0] == fluvial_nodes[x] #how pythonic is this syntax?
        
        #self.uncorrected_tau_array[x] = tau_initial   
        
        #DETRACT SHEAR STRESS WITH KEAN/SMITH APPROACH
        blocks_above_flow = (is_block_in_cell) #& (tracking_mat[0:slicing_index, 2] >= flow_depth)             
        if np.count_nonzero(blocks_above_flow) == 0:
            sigma_d_blocks = 0
        else:
            beta = (a1 * (h / d)) / np.power(np.power(h / d, 5 / 3) + np.power(a1 / a2, 2), 1/2)
            avg_diam_blocks = np.average(tracking_mat[blocks_above_flow, 1])
            submerged_block = (is_block_in_cell) & (tracking_mat[0:for_slicing, 1] < h)
            emergent_block = (is_block_in_cell) & (tracking_mat[0:for_slicing, 1] >= h)
            tracking_mat[submerged_block, 3] = tracking_mat[submerged_block, 1]
            tracking_mat[emergent_block, 3] = h            
            avg_submerged_height_blocks = np.average(tracking_mat[is_block_in_cell, 3])
            avg_spacing_blocks =  dx / np.count_nonzero(blocks_above_flow)
            sigma_d_blocks = (1 / 2) * drag_cube * np.power(beta, 2) *(avg_submerged_height_blocks * avg_diam_blocks / np.power(avg_spacing_blocks, 2))
        adjusted_shear_stress = tau_initial / (1 + sigma_d_blocks)            
        #tau = self.calc_shear_stress_with_roughness(tau_initial, self.tracking_mat, is_block_in_cell, self.drag_cube, h, self.dx, self.z0, self.for_slicing)
        #self.corrected_tau_array[x] = tau    
        #h, v, tau = ut.calc_flow_depth_and_velocity(x, np.array([False])) #False array is standing in for is_block_in_cell
        mg['node']['water_depth'][fluvial_nodes[x]] = h
        mg['node']['water_velocity'][fluvial_nodes[x]] = v
        mg['node']['corrected_shear_stress'][fluvial_nodes[x]] = adjusted_shear_stress
        
        f_covered = sum(np.power(tracking_mat[is_block_in_cell, 1], 2)) / np.power(dx, 2) #(np.count_nonzero(is_block_in_cell) * np.power(side_length, 2)) / (dx * dx)
        if f_covered > 1:
            f_covered = 1
        mg['node']['f_covered'][fluvial_nodes[x]] = f_covered
        
    #print 'ready to erode with fsc'
    #erode with fastscape: ordering code poached from the Fsc module by DEJH
    ###upstream_order_IDs = mg['node']['upstream_node_order']
    ###defined_flow_receivers = np.not_equal(mg['node']['links_to_flow_receiver'],BAD_INDEX_VALUE)
    ###flow_receivers = mg['node']['flow_receiver']
    ###flow_link_lengths = mg.link_length[mg['node']['links_to_flow_receiver'][defined_flow_receivers]]
    ###spatial_step[defined_flow_receivers] = flow_link_lengths            
    
    #skipping cython implementation, only pure python here
    #print 'done with hydraulics, eroding with fastscape'
    #f_open = 1 - f_covered #for now, in the case of no blocks
    old_node_elevs = np.zeros((nrows*ncols))
    old_node_elevs[:] = mg['node']['topographic__elevation']
    #print old_node_elevs[0:100]
    ###for i in upstream_order_IDs:
    ###    j = flow_receivers[i]
    ###    if i != j:
    ###        if lf.lake_at_node[i] == False:
    ###            mg['node']['topographic__elevation'][i] = (mg['node']['topographic__elevation'][i] + (mg['node']['topographic__elevation'][j] * (1 - mg['node']['f_covered'][i]) * mg['node']['erodibility'][i] * dens_water * g * (mg['node']['corrected_shear_stress'][i] / (g * dens_water * mg['node']['topographic__steepest_slope'][i])) * (dt / spatial_step[i])) + (dt * (1 - mg['node']['f_covered'][i]) * mg['node']['erodibility'][i] * tau_c_br)) / (1 + ((1 - mg['node']['f_covered'][i]) * mg['node']['erodibility'][i] * dens_water * g * mg['node']['water_depth'][i] * (dt / spatial_step[i])))
    ###        else:
    ###            pass
        
    #Forward Euler solver b/c fastscape ones turns out to be incorrect
    mg['node']['topographic__elevation'] -= mg['node']['erodibility'] * (mg['node']['corrected_shear_stress'] - tau_c_br) * (1 - mg['node']['f_covered']) * dt     
    
    mg['node']['incision_rate'] = abs(old_node_elevs - mg['node']['topographic__elevation'])
    #print mg['node']['topographic__elevation'][0:100]    #sets up ability to deliver blocks based on incision rate
    #print '==='
    #print max(mg['node']['incision_rate'])    
    #print '==='
        #self.surface_elev_array[-1] -= (baselevel_drop * self.timestep) #adjust baselevel node
        #for cell in range(n_cells - 2, -1, -1):
        #    h_star = self.corrected_tau_array[cell] / (self.dens_water * self.g * ((self.surface_elev_array[cell] - self.surface_elev_array[cell + 1]) / self.dx))
        #    f_open = 1 - self.cover_frac_array[cell]            
        #    self.surface_elev_array[cell] = (self.surface_elev_array[cell] + (self.surface_elev_array[cell + 1] * f_open * self.ke_br * self.dens_water * self.g * h_star * (self.timestep / self.dx)) + (self.timestep * f_open * self.ke_br * self.tau_c_br)) / (1 + (f_open * self.ke_br * self.dens_water * self.g * h_star * (self.timestep / self.dx)))
        
        
    #_ = sp.erode(mg, dt=dt)
    # this component is of an older style,
    # so it still needs a copy of the grid to be passed
    _ = hd.diffuse(dt)
    mg.at_node['topographic__elevation'][mg.core_nodes] += uplift_per_step # add the uplift
    #print max(mg['node']['topographic__elevation'])
    elapsed_time += dt
    if counter % plot_counter == 0:
        print ('Completed loop %d' % counter)
        elev = mg['node']['topographic__elevation']
        cover = mg['node']['f_covered']
        elev_r = mg.node_vector_to_raster(elev)
        x_blocks = mg.node_x[np.where(mg['node']['num_blocks'] > 0)]
        y_blocks = mg.node_y[np.where(mg['node']['num_blocks'] > 0)]
        # Plot topography
        plt.figure()
        im = plt.imshow(elev_r, cmap=plt.cm.terrain, extent=[0,5000,0,5000], origin='lower', vmin = 0, vmax = 9)  # display a colored image
        plt.scatter(x_blocks * 10, y_blocks * 10, color='k')
        plt.xlim(0, 5000)
        plt.ylim(0, 5000)
        plt.colorbar(im)
        plt.title('Elevation [m]')
        plt.savefig('broken_block_topo_' + str(counter) + '.png')
        #plt.show()
        
        #save out as .npy...
        np.save('broken_block_elev_' + str(counter) + '.npy', elev)
        np.save('broken_block_cover_' + str(counter) + '.npy', cover)
        
        #and netCDF
        write_netcdf('broken_block_elev_' + str(counter) + '.nc', mg, format='NETCDF3_64BIT', names='topographic__elevation', at='node')
        write_netcdf('broken_block_cover_' + str(counter) + '.nc', mg, format='NETCDF3_64BIT', names='f_covered', at='node')
    counter += 1

    
#Get resulting topography, turn into a raster
elev = mg['node']['topographic__elevation']
elev_r = mg.node_vector_to_raster(elev)
da = mg['node']['drainage_area']
da_r = mg.node_vector_to_raster(da)
wvf = mg['node']['water__volume_flux']
wvf_r = mg.node_vector_to_raster(wvf)

np.save('broken_block_final_elevs.npy', elev)

#nodes with blocks in them, to plot as dots
x_blocks = mg.node_x[np.where(mg['node']['num_blocks'] > 0)]
y_blocks = mg.node_y[np.where(mg['node']['num_blocks'] > 0)]
# Plot topography
plt.figure(1)
im = plt.imshow(elev_r, cmap=plt.cm.terrain, extent=[0,5000,0,5000], origin='lower')  # display a colored image
plt.scatter(x_blocks * 10, y_blocks * 10, color='k')
plt.colorbar(im)
plt.title('Elevation [m]')

plt.figure(2)
im = plt.imshow(da_r, cmap=plt.cm.jet, extent=[0,5000,0,5000], origin='lower')  # display a colored image
plt.scatter(x_blocks * 10, y_blocks * 10, color='k')
plt.colorbar(im)
plt.title('Drainage Area')

plt.figure(3)
im = plt.imshow(wvf_r, cmap=plt.cm.jet, extent=[0,5000,0,5000], origin='lower')  # display a colored image
plt.scatter(x_blocks * 10, y_blocks * 10, color='k')
plt.colorbar(im)
plt.title('Water Volume Flux')
plt.show()