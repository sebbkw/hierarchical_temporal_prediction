import pickle, math, os

import numpy as np
import torch

import scipy
import scipy.stats
from scipy import ndimage
import scipy.optimize as opt
from scipy import signal
import scipy.fft as fft

class VirtualPhysiology:
    # model               trained model
    # hyperparameters     model hyperparameters
    # hidden_unit_range   range object of hidden units for analysis
    # device              device for tensors (cpu or cuda)
    def __init__ (self, model, hyperparameters, frame_shape, hidden_units, device):
        self.data = []

        self.model = model
        self.model.eval()

        self.hyperparameters = hyperparameters
        self.warmup = hyperparameters["warmup"]
        self.frame_size = hyperparameters["frame_size"]
        self.t_steps = 50

        self.frame_shape = frame_shape
        self.hidden_units = hidden_units
        self.device = device

        self.data = []
        for group in self.hidden_units:
            self.data.append([])

        self.osi_thresh = 0.4
        self.dsi_thresh = 0.3
        self.mean_response_offset = 5

        min_n, max_n = min(frame_shape), max(frame_shape)
        self.spatial_frequencies = [i/max_n for i in range(1, min_n//2+1)] # Cycles / pixel
        self.orientations = np.arange(0, 360, 5) # Degrees
        self.temporal_frequencies = np.linspace(1/self.t_steps, 1/4, self.t_steps//4) # Cycles / frame

    @classmethod
    def load (cls, data_path, model, hyperparameters, frame_shape, hidden_units, device):
        virtual_physiology = cls(
            model=model,
            hyperparameters=hyperparameters,
            frame_shape=frame_shape,
            hidden_units=hidden_units,
            device=device
        )

        with open(data_path, 'rb') as handler:
            virtual_physiology.data = pickle.load(handler)

        return virtual_physiology

    # relative_path = use file name as part of data_path relative to model dir
    def save (self, data_path):            
        with open(data_path, 'wb') as p:
            pickle.dump(self.data, p, protocol=4)

        with open(data_path + '.params', 'wb') as p:
            params = {
                "t_steps": self.t_steps,
                "spatial_frequencies": self.spatial_frequencies,
                "orientations": self.orientations,
                "temporal_frequencies": self.temporal_frequencies
            }
            pickle.dump(params, p, protocol=4)


    def get_unit_data (self, unit_idx):
        for group in self.data:
            for unit_data in group:
                if unit_data["hidden_unit_index"] == unit_idx:
                    return unit_data
        return False


    def get_group_from_unit_idx (self, unit_idx):
        for group_idx in range(len(self.hidden_units)):
            if unit_idx <  np.sum(self.hidden_units[:group_idx+1]):
                return group_idx
        return False

    def get_response_weighted_average (self, n_rand_stimuli=100):
        # Pre-allocate lists of lists to hold gaussian noise stimuli and associated unit activity
        stimuli = []
        unit_activity = []
        for _ in range (np.sum(self.hidden_units)):
            stimuli.append([])
            unit_activity.append([])

        noise_shape = (n_rand_stimuli, self.warmup+self.t_steps, self.frame_size)
        noise = np.random.normal(loc=0, scale=1, size=noise_shape)
        noise = torch.Tensor(noise).to(self.device)

        with torch.no_grad():
            _, hidden_state = self.model(noise)


        response = hidden_state.detach().numpy().reshape(n_rand_stimuli*(self.warmup+self.t_steps), -1)
        noise = noise.detach().numpy().reshape(-1, self.frame_size)

        rwa_arr = []
        for unit_idx, unit_responses in enumerate(response.T):
            group_idx = self.get_group_from_unit_idx(unit_idx)

            if unit_idx % 100 == 0:
                print('Processing RWA for unit', unit_idx)

            if np.sum(unit_responses):
                if group_idx == 0:
                    rwa = np.average(noise, axis=0, weights=unit_responses)
                else:
                    rwa = np.average(noise[:-group_idx], axis=0, weights=unit_responses[group_idx:])

                self.data[group_idx].append({
                    "hidden_unit_index": unit_idx,
                    "response_weighted_average": rwa
                })

        print('Finished averaging stimuli')

        return self

    # sf = cycles per pixel
    # tf = cycles per second
    # speed = tf/sf = pixels per second
    def get_grating_stimuli(self, spatial_frequency, orientation, temporal_frequency, grating_amplitude, frames):
        y_size, x_size = self.frame_shape

        theta = (orientation-90) * np.pi/180
        x, y = np.meshgrid(np.arange(0, x_size), np.arange(0, y_size))
        x_theta = x * np.cos(theta) + y * np.sin(theta)

        phase_shift = 2*np.pi*temporal_frequency
        phases = np.arange(frames)*phase_shift

        grating_frames = []
        for phase in phases:
            grating_frames.append( grating_amplitude * np.sin(2*spatial_frequency*np.pi*x_theta - phase) )


        gratings = np.array(grating_frames).reshape(1, frames, y_size*x_size)
        gratings = (gratings-np.mean(gratings))/np.std(gratings) ##
        gratings = torch.Tensor(gratings).to(self.device)

        return gratings

    def get_grating_responses (self):
        # Add array to data dictionary structures containing
        # complete response (for each grating phase) and mean response (averaged across phases)
        # for each spatial frequency/orientation/tf combination
        for group_data in self.data:
            for unit_data in group_data:
                unit_data["grating_responses"] = np.zeros((
                    len(self.spatial_frequencies),
                    len(self.orientations),
                    len(self.temporal_frequencies),
                    self.t_steps
                ))
                unit_data["mean_grating_responses"] = np.zeros((
                    len(self.spatial_frequencies),
                    len(self.orientations),
                    len(self.temporal_frequencies)
                ))

        # Keep track of progress for display purposes
        param_count = 0
        try:
            total_params = self.data[0][0]["mean_grating_responses"].size
        except:
            total_params = self.data[1][0]["mean_grating_responses"].size            

        # Loop through each parameter combination for each unit
        for sf_idx, sf in enumerate(self.spatial_frequencies):
            for ori_idx, ori in enumerate(self.orientations):
                for tf_idx, tf in enumerate(self.temporal_frequencies):

                    # Generate gratings for particular param combination
                    gratings = self.get_grating_stimuli(sf, ori, tf, grating_amplitude=1, frames=self.warmup+self.t_steps)

                    # Feedforward pass through network
                    with torch.no_grad():
                        _, hidden_state = self.model(gratings)

                    # Loop through unit responses at each time step
                    for group_data in self.data:
                        for unit_data in group_data:
                            unit_idx = unit_data["hidden_unit_index"]

                            # Discard warm up period of network's response to gratings
                            unit_responses = hidden_state[0, self.warmup:, unit_idx].cpu().numpy()

                            unit_data["grating_responses"][sf_idx, ori_idx, tf_idx] = unit_responses
                            unit_data["mean_grating_responses"][sf_idx, ori_idx, tf_idx] = np.mean(unit_responses)

                    if param_count % 100 == 99:
                        print("Finished param combination {}/{}".format(param_count+1, total_params)) 
                    param_count += 1

        print("Finished tuning curve")

        return self

    # Takes orientation tuning curve at max tf and sf
    # Returns direction selectivity (DSI)
    def get_DSI (self, tuning_curve):    
        orient_pref_idx = np.where(tuning_curve == np.max(tuning_curve))[0][0]
        orient_pref = self.orientations[orient_pref_idx]
        orient_pref_resp = tuning_curve[orient_pref_idx]

        orient_opp = (orient_pref + 180) % 360
        orient_opp_idx = np.where(self.orientations == orient_opp)[0][0]
        orient_opp_resp = tuning_curve[orient_opp_idx]

        DSI = (orient_pref_resp - orient_opp_resp) / (orient_pref_resp + orient_opp_resp)
        #DSI = 1 - (orient_opp_resp/orient_pref_resp)

        return DSI


    # Takes orientation tuning curve at max tf and sf
    # Returns orientation selectivity (OSI)
    def get_OSI (self, tuning_curve):    
        orient_pref_idx = np.where(tuning_curve == np.max(tuning_curve))[0][0]
        orient_pref = self.orientations[orient_pref_idx]
        orient_pref_resp = tuning_curve[orient_pref_idx]

        orient_orth1 = (orient_pref + 90) % 360
        orient_orth1_idx = np.where(self.orientations == orient_orth1)[0][0]
        orient_orth2 = (orient_pref - 90) % 360
        orient_orth2_idx = np.where(self.orientations == orient_orth2)[0][0]
        orient_orth_resp = (tuning_curve[orient_orth1_idx]+tuning_curve[orient_orth2_idx]) / 2

        OSI = (orient_pref_resp - orient_orth_resp) / (orient_pref_resp + orient_orth_resp)

        return OSI

    def get_orientation_tuning_curve(self, unit_data):
        mean_grating_responses = unit_data["mean_grating_responses"]

        # Get indices of max grating response (mean across time steps)
        max_mean_grating_response = np.max(mean_grating_responses)
        unit_data["max_mean_grating_response"] = max_mean_grating_response
        sf_idx, ori_idx, tf_idx = [idx[0] for idx in np.where(mean_grating_responses == max_mean_grating_response)]

        orientation_tuning_curve = mean_grating_responses[sf_idx, :, tf_idx]
        return orientation_tuning_curve


    # Gets OSI, DSI, CV and modulation ratio for each unit
    def get_grating_responses_parameters (self):
        for group_idx, group_data in enumerate(self.data):
            for unit_i, unit_data in enumerate(group_data):
                grating_responses = unit_data["grating_responses"]
                mean_grating_responses = unit_data["mean_grating_responses"]

                # Get indices of max grating response (mean across time steps)
                max_mean_grating_response = np.max(mean_grating_responses)
                unit_data["max_mean_grating_response"] = max_mean_grating_response
                sf_idx, ori_idx, tf_idx = [idx[0] for idx in np.where(mean_grating_responses == max_mean_grating_response)]

                # Convert these indices into the underlying parameters
                max_sf = unit_data["preferred_sf"] = self.spatial_frequencies[sf_idx]
                max_ori = unit_data["preferred_orientation"] =  self.orientations[ori_idx]
                max_tf = unit_data["preferred_tf"] = self.temporal_frequencies[tf_idx]

                # Response to moving grating for parameters that give maximum response
                optimum_grating_response = grating_responses[sf_idx, ori_idx, tf_idx]
                unit_data["optimum_grating_response"] = optimum_grating_response

                # Get CV and DSI measures from curve
                # given max spatial frequency and temporal frequency parameters
                orientation_curve = mean_grating_responses[sf_idx, :, tf_idx]
                unit_data["OSI"] = self.get_OSI(orientation_curve)
                unit_data["DSI"] = self.get_DSI(orientation_curve)

                if unit_i % 50 == 49:
                    print("Finished unit {} / {}, group {} / {}".format(
                        unit_i+1, len(group_data), group_idx+1, len(self.data)
                    ))

        return self

    # Filter unresponsive units and units where curve fitting failed
    def filter_unit_data (self, group_idx):
        group_data = self.data[group_idx]

        # Reject low mean response units (< 1% of mean, max mean response)
        all_max_mean = []
        for group in self.data:
            for u in group:
                all_max_mean.append(u["max_mean_grating_response"])

        mean_responses = [u["max_mean_grating_response"] for u in group_data]

        response_threshold = 0.01 * np.mean(all_max_mean)
        n_filtered = len(np.where(mean_responses < response_threshold)[0])
        print("{} / {} units below response threshold".format(n_filtered, len(group_data)))

        # Reject units where curve fitting failed (modulation_ratio set as False)
        n_filtered = len([u for u in group_data if not u["modulation_ratio"]])
        print("{} / {} units failed to fit curve for modulation ratio estimate".format(n_filtered, len(group_data)))

        # Now actually filter those units out
        return [u for u in group_data if u["modulation_ratio"] and u["max_mean_grating_response"] >= response_threshold]

    # Filter unresponsive units (in place method)
    def filter_nonresponding_units (self):
        for group_idx, group in enumerate(self.data):
            filtered = []
            for u in group:
                max_mean = np.max(u["mean_grating_responses"])
                if max_mean > 0:
                    filtered.append(u)

            self.data[group_idx] = filtered

            original_n = self.hidden_units[group_idx]
            filtered_n = len(self.data[group_idx])
            print(f"{filtered_n} / {original_n} units kept after filtering non-responsive units")

        return self


    def get_moving_bar_stimuli (self, direction, x, y, bar_amplitude=1, bar_size=5, frames_len = 20):
        # Include warmup period for network in stimulus
        # Where first n frames will be gray (all 0's)
        total_frames = self.warmup + frames_len

        stimuli = np.zeros((total_frames, self.frame_shape[0], self.frame_shape[1]))

        for i in range(frames_len):
            bar_position = 0
            square = np.ones((bar_size, bar_size))*bar_amplitude

            if direction == 0 or direction == 270:
                bar_position = (bar_size - i) % bar_size
            else:
                bar_position = i%(bar_size) 

            if direction == 0 or direction == 180:
                square[bar_position, :] = -bar_amplitude
            else:
                square[:, bar_position] = -bar_amplitude

            stimulus = np.zeros((self.frame_shape[0], self.frame_shape[1]))
            stimulus[y:y+bar_size, x:x+bar_size] = square

            stimuli[self.warmup+i, :, :] = stimulus

        # Reshape frame into flat array and convert into Tensor object
        stimuli = stimuli.reshape(total_frames, self.frame_size)
        stimuli = torch.Tensor(stimuli).unsqueeze(0).to(self.device)

        return stimuli

    def fit_sine (self, x, y):
        # Fit to sine
        def func(x, a, b, c, d):
            return a*np.sin(b*x + c) + d

        def get_mse_loss (y, y_est):
            return np.sum((y-y_est)**2)/ len(y)

        # Get r_squared from https://stackoverflow.com/a/37899817
        def get_rsq (y, y_est):
            residuals = y - y_est
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y-np.mean(y))**2)
            r_squared = 1 - (ss_res / ss_tot)

            return r_squared


        best_params = []

        for iteration in range(5):
            scale = 1-0.2*iteration
            n_random_guesses = 10000 if iteration == 0 else 1000

            params = []
            loss_list = []

            for _ in range(n_random_guesses):
                if iteration == 0:
                    rand_params = [
                        np.random.uniform(low=np.mean(y)-2.5, high=np.mean(y)+2.5),
                        np.random.uniform(low=0, high=10),
                        np.random.uniform(low=0, high=len(y)),
                        np.random.uniform(low=np.min(y)-2.5, high=np.max(y)+2.5)
                    ]
                else:
                    prev_best_params = best_params[-1]
                    rand_params_ = [
                        np.random.uniform(low=-2*scale, high=2*scale),
                        np.random.uniform(low=-1*scale, high=1*scale),
                        np.random.uniform(low=-2*scale, high=2*scale),
                        np.random.uniform(low=-2*scale, high=2*scale)
                    ]
                    rand_params = [p+prev_best_params[idx] for idx, p in enumerate(rand_params_)]

                # Get the estimated curve based on fitted parameters
                y_est = func(x, *rand_params)
                loss = get_mse_loss(y, y_est) #get_rsq(y, y_est)

                params.append(rand_params)
                loss_list.append(loss)

            # Get the index of the lowest RSQ, use this to find the
            # corresponding parameters used
            best_params.append(params[np.argmin(loss_list)])

        final_params = best_params[-1]
        final_y_est = func(x, *final_params)
        final_loss = get_mse_loss(y, final_y_est)
        final_rsq = get_rsq(y, final_y_est)

        return final_params, final_y_est, final_rsq, final_loss


    # Takes list of response as well as a start and
    # end offset for where curve fitting should occur
    # Returns modulation ratio, estimated curve and RSQ of curve fit
    def get_modulation_ratio (self, activity, start_offset, end_offset):
        x = np.arange(start_offset, end_offset)
        y = activity[start_offset:end_offset]

        final_params, final_y_est, final_rsq, final_loss = self.fit_sine (x, y)
        # Try one more time if it fails
        if final_loss < 0.05 or final_rsq > 0.5:
            final_params, final_y_est, final_rsq, final_loss = self.fit_sine (x, y)

        # Average unit activity
        f0 = np.mean(activity[self.warmup:])
        # Absolute of the amplitude of the fitted sine
        f1 = (abs(final_params[0]))
        mod_ratio = f1/f0

        # Reject f values for those units with poor sine fits
        #if (final_loss < 0.05 or final_rsq > 0.5) and f0 != 0:
        cc = scipy.stats.pearsonr(final_y_est, y)[0]
        if cc>0.9 and f0 != 0:
            return {
                'modulation_ratio'        : mod_ratio,
                'modulation_ratio_all'    : mod_ratio,
                'modulation_ratio_cc'     : cc,
                'modulation_ratio_y'      : final_y_est,
                'modulation_ratio_rsq'    : final_rsq,
                'modulation_ratio_loss'   : final_loss, 
                'modulation_ratio_params' : final_params
            }
        else:
            return {
                'modulation_ratio'        : False,
                'modulation_ratio_all'    : mod_ratio,
                'modulation_ratio_cc'     : cc,
                'modulation_ratio_y'      : final_y_est,
                'modulation_ratio_y_true' : final_y_est,
                'modulation_ratio_rsq'    : final_rsq,
                'modulation_ratio_loss'   : final_loss, 
                'modulation_ratio_params' : final_params
            }

    def get_modulation_ratio_all (self):
        for g_idx, g in enumerate(self.data):
            for unit_idx, unit_data in enumerate(g):    
                if unit_idx % 10 == 0:
                    print(f'Starting group {g_idx}, unit {unit_idx}')

                modulation_data = self.get_modulation_ratio(
                    unit_data['optimum_grating_response'], self.warmup, self.t_steps
                )
                
                unit_data["modulation_ratio"]        = modulation_data["modulation_ratio"]
                unit_data["modulation_ratio_cc"]     = modulation_data["modulation_ratio_cc"]
                unit_data["modulation_ratio_all"]    = modulation_data["modulation_ratio_all"]
                unit_data["modulation_ratio_y"]      = modulation_data["modulation_ratio_y"]
                unit_data["modulation_ratio_rsq"]    = modulation_data["modulation_ratio_rsq"]
                unit_data["modulation_ratio_loss"]   = modulation_data["modulation_ratio_loss"]
                unit_data["modulation_ratio_params"] = modulation_data["modulation_ratio_params"]
                

    # https://www.nature.com/articles/nn1786#Sec11
    # plaid_angle = spread of plaid components
    def get_plaid_pattern_index (self, grating_amplitude=1):
        # Test if true response is more similar to pattern or component predictions
        # Partial correlation of x and y controlling for z
        def partial_correlation (x, y, z):
            r_xy, _ = scipy.stats.pearsonr(x, y)
            r_xz, _ = scipy.stats.pearsonr(x, z)
            r_yz, _ = scipy.stats.pearsonr(y, z)

            return (r_xy - r_xz*r_yz) / ((1-r_xz**2)*(1-r_yz**2))**0.5

        # Converts from distribution of r values to normal distribution
        # (to compare across units)
        def fisher_r_to_z (a, df):
            return 0.5*np.log((1+a)/(1-a)) / (1/df)**0.5
        
        orientation_step = 6
        orientations     = self.orientations[::6]
        
        for group_idx, group_data in enumerate(self.data):
            for unit_i, unit_data in enumerate(group_data):
                max_sf_idx = np.where(np.array(self.spatial_frequencies) == unit_data['preferred_sf'])[0][0]
                max_tf_idx = np.where(np.array(self.temporal_frequencies) == unit_data['preferred_tf'])[0][0]

                # Response if unit responds to plaid the same as a grating of same angle
                pattern_prediction = unit_data['mean_grating_responses'][max_sf_idx, ::orientation_step, max_tf_idx]
                
                z_p_arr = []
                z_c_arr = []
                
                for plaid_angle in [60, 90, 120, 150]:
                    # Response if unit responds to individual components of the plaid
                    # Shift is the amount to rotate the tuning curve (+/- half the plaid angle)
                    shift = int( (plaid_angle/2) / (orientations[1]-orientations[0]) )
                    component_prediction_a = np.roll(pattern_prediction,  shift)
                    component_prediction_b = np.roll(pattern_prediction, -shift)
                    component_prediction = (component_prediction_a + component_prediction_b) / 2

                    # Now get 'true' response to the plaid (as a function of orientation)
                    plaid_response = []
                    with torch.no_grad():
                        for ori_idx, ori in enumerate(orientations): 
                            # Plaid at unit's preferred tf and sf
                            grating_a = self.get_grating_stimuli(
                                unit_data['preferred_sf'],
                                ori+plaid_angle/2,
                                unit_data['preferred_tf'],
                                grating_amplitude/2,
                                50
                            )
                            grating_b = self.get_grating_stimuli(
                                unit_data['preferred_sf'],
                                ori-plaid_angle/2,
                                unit_data['preferred_tf'],
                                grating_amplitude/2,
                                50
                            )
                            plaid = grating_a + grating_b

                            # Feed into model
                            _, hidden_state = self.model(plaid)

                            # Get mean response for current unit
                            mean_plaid_response = hidden_state[0, self.warmup:, unit_data['hidden_unit_index']] \
                                .mean() \
                                .detach() \
                                .numpy()
                            plaid_response.append(mean_plaid_response)

                    r_p = partial_correlation(plaid_response, pattern_prediction, component_prediction)
                    r_c = partial_correlation(plaid_response, component_prediction, pattern_prediction)

                    z_p = fisher_r_to_z(r_p, len(pattern_prediction)-3)
                    z_c = fisher_r_to_z(r_c, len(component_prediction)-3)
                    
                    z_p_arr.append(z_p)
                    z_c_arr.append(z_c)

                z_p_mean = np.mean(z_p_arr)
                z_c_mean = np.mean(z_c_arr)
                pattern_idx = z_p_mean - z_c_mean
                                
                # Save all these results
                unit_data['plaid_rp'] = r_p
                unit_data['plaid_rc'] = r_c
                unit_data['plaid_zp'] = z_p_mean
                unit_data['plaid_zc'] = z_c_mean
                unit_data['plaid_pattern_index'] = pattern_idx
                unit_data['plaid_response'] = plaid_response
                
                print("Finished unit {} / {}, group {} / {}".format(
                    unit_i+1, len(group_data), group_idx+1, len(self.data)
                 ))
                
        return self
    
    
    def get_moving_bar_responses (self):
        bar_size = 20
        # -5 because we are effectively convolving with moving bar "kernel" without padding
        convolved_size_r = self.frame_shape[0] - bar_size
        convolved_size_pixels_r = np.arange(convolved_size_r)
        convolved_size_c = self.frame_shape[1] - bar_size
        convolved_size_pixels_c = np.arange(convolved_size_c)

        # Pre-allocate array to hold mean response to stimulus at each location/direction
        for group_data in self.data:
            for unit_data in group_data:
                unit_data["moving_bar_responses"] = np.zeros((4, convolved_size_r, convolved_size_c))

        # Loop through each orientation and spatial location
        for orient_idx, orient in enumerate([0, 90, 180, 270]):
            for row_idx, row in enumerate(convolved_size_pixels_r):
                for col_idx, col in enumerate(convolved_size_pixels_c):

                    stimuli = self.get_moving_bar_stimuli(orient, col, row, bar_size=bar_size)

                    with torch.no_grad():
                        _, hidden_state = self.model(stimuli)

                    for group_data in self.data:
                        for unit_data in group_data:
                            # Get response at each time step of stimulus
                            unit_activity = hidden_state[0, :, unit_data["hidden_unit_index"]].cpu().numpy()
                            # Take baseline response as mean of warmup period
                            #baseline_response = np.mean(unit_activity[:self.warmup])
                            # Take overlal mean response as non-warmup period subtracted from baseline
                            mean_response = np.mean(unit_activity[:]) # - baseline_response)

                            unit_data["moving_bar_responses"][orient_idx, row_idx, col_idx] = mean_response
                    
                print("Got responses for {} degrees, row {}".format(orient, row+1))

        # Also compute average response across orientations to produce 'heatmap'
        for group_data in self.data:
            for unit_data in group_data:
                unit_data["mean_moving_bar_responses"] = np.mean(unit_data["moving_bar_responses"], axis=0)

        return self

        
    def get_receptive_field_centres (self):
        def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):    
            xy = (16, 16)
            x = np.linspace(0, int(xy[0]-1), int(xy[0]))
            y = np.linspace(0, int(xy[1]-1), int(xy[1]))
            x,y = np.meshgrid(x, y)

            xo = float(xo)
            yo = float(yo)    
            a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
            b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
            c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
            g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                                    + c*((y-yo)**2)))

            return g.ravel()

        def weighted_centre (resp):
            (X,Y) = np.meshgrid(np.arange(0, resp.shape[0]), np.arange(0, resp.shape[1]))
            x_coord = (X*resp).sum() / resp.sum().astype("float")
            y_coord = (Y*resp).sum() / resp.sum().astype("float")
            return (y_coord, x_coord)


        for group_idx, group_data in enumerate(self.data):
            poor_fit_count = 0

            for unit_i, unit_data in enumerate(group_data):
                resp = unit_data["mean_moving_bar_responses"]

                max_val       = np.max(resp)
                mean_centre   = weighted_centre(resp)
                initial_guess = (max_val, mean_centre[1], mean_centre[0], 3, 3, 0, 0)

                try:
                    popt, pcov = opt.curve_fit(
                        twoD_Gaussian,
                        resp.shape,
                        resp.reshape(-1),
                        p0=initial_guess,
                        bounds=(
                            [0, -np.inf, -np.inf, 0, 0, 0, 0],
                            [np.inf, np.inf, np.inf, np.inf, np.inf, np.pi*2, np.inf]
                        )
                    )

                    resp_gaussian_fit = twoD_Gaussian(resp.shape, *popt)

                    centre = popt[2], popt[1]
                    cc = scipy.stats.pearsonr(resp.reshape(-1), resp_gaussian_fit)[0]
                except Exception as e:
                    centre = (np.nan, np.nan)

                # 'Convolution' with moving bar means this center needs to be scaled back up to 
                # full size of frame
                scale_factor = self.frame_shape[0]/resp.shape[0]
                centre = (centre[0] * scale_factor, centre[1] * scale_factor)

                if np.isnan(centre[0]) or np.isnan(centre[1]) or cc < 0.6:
                    unit_data["receptive_field_centre"]       = (-1, -1)
                    unit_data["receptive_field_gaussian_fit"] = np.nan
                    unit_data["receptive_field_response"] = resp

                    poor_fit_count += 1
                else:
                    unit_data["receptive_field_centre"]       = centre
                    unit_data["receptive_field_gaussian_fit"] = resp_gaussian_fit.reshape(16, 16)

                if unit_i % 100 == 99:
                    print("Finished unit {} / {}, group {} / {}".format(
                        unit_i+1, len(group_data), group_idx+1, len(self.data)
                    ))

            print("{}/{} units were rejected as Gaussian could not be fit to receptive fields in group {}".format(
                poor_fit_count, len(group_data), group_idx+1
            ))

        return self
