from __future__ import division, print_function

import os, pickle, traceback
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.stats import scoreatpercentile

from .data import TSeries, PatchClampRecording
from .filter import bessel_filter
from .baseline import mode_filter, adaptive_detrend
from .event_detection import threshold_events
from .util.data_test import DataTestCase


def detect_evoked_spikes(data, pulse_edges, **kwds):
    """Return a list of dicts describing spikes in a patch clamp recording that were evoked by a single stimulus pulse.

    This function simply wraps either detect_ic_evoked_spikes or detect_vc_evoked_spikes, depending on the clamp mode
    used in *data*.

    Parameters
    ==========
    data : PatchClampRecording
        The recorded patch clamp data. The recording should be made with a brief pulse intended
        to evoke a single spike with short latency.
    pulse_edges : (float, float)
        The start and end times of the stimulation pulse, relative to the timebase in *data*. 

    Returns
    =======
    spikes : list
        Each item in this list is a dictionary containing keys 'onset_time', 'max_slope_time', and 'peak_time',
        indicating three possible time points during the spike that could be detected. Any of these values may
        be None to indicate that the timepoint could not be reliably determined. Additional keys may be present,
        such as 'peak' and 'max_slope'.
    """
    trace = data['primary']
    if data.clamp_mode == 'vc':
        return detect_vc_evoked_spikes(trace, pulse_edges, **kwds)
    elif data.clamp_mode == 'ic':
        return detect_ic_evoked_spikes(trace, pulse_edges, **kwds)
    else:
        raise ValueError("Unsupported clamp mode %s" % trace.clamp_mode)


def rc_decay(t, tau, Vo): 
    """function describing the deriviative of the voltage.  If there
    is no spike one would expect this to fall off as the RC of the cell. """
    return -(Vo/tau)*np.exp(-t/tau)


def detect_ic_evoked_spikes(trace, 
                            pulse_edges, 
                            mse_threshold=30., 
                            dv2_event_area=10e-6,
                            pulse_bounds_move=[.15e-3, 0.02e-3],  #0.03e-3 sometimes gets some artifact
                            double_spike=1.e-3,
                            ui=None,
                            artifact_width=.1e-3, #this number seems fairly acurate
                            dv_thresholds = [125., 15.] #130 is too high 
                            ):
    """Return a dict describing an evoked spike in a patch clamp recording. Or Return None
    if no spike is detected.

    This function assumes that a square voltage pulse is used to evoke a spike
    in a current clamp recording, and that the spike initiation occurs *during* or within a
    short region after the stimulation pulse.  Note that finding 


    Parameters
    ==========
    trace: *Trace* instance
        The recorded patch clamp data. The recording should be made with a brief pulse intended
        to evoke a single spike with short latency.
    pulse_edges: (float, float)
        The start and end times of the stimulation pulse, relative to the timebase in *trace*.
     mse_threshold: float
        Value used to determine if there was a spike close to the end of the stimulation pulse.  If the 
        mean square error value of a fit to a RC voltage decay is larger than *mse_threshold* a spike
        has happened.
    event_area: float
        The integral of the 'bump' in dv2 must have at least the following *dv2_event_area*. 
    pulse_bounds_move: np.array; [float, float]
        There are large fluxuations in v, dv, and dv2 around the time of pulse initiation
        and termination.  *pulse_bounds_move* specifies how much time after the edges of the 
        stimulation pulse should be considered in the search window. *pulse_bounds_move[0]* is added to
        stimulation pulse initiation, *pulse_bounds_move[1]* is added to stimulation pulse termination.
        Spikes after the termination of the search window are identified by attempting to fit the dv decay 
        to the trace that would be seen from standard RC decay (see *mse_threshold*). 
    double_spike: float
        time between 
    ui: 
        user interface for viewing spike detection
    artifact_width:
        amount of time that should be not considered for spike metrics after the pulse search window,
        i.e. pulse_edges[1] + pulse_bounds_move[1]
    dv_thresholds: [float, float]
        *dv_thresholds[0]* (*dv_thresholds[1]*) threshold for potential spike at beginning (end) of pulse window.

    Returns 
    =======
    spikes: dictionary
        contains following information about spikes:
        onset_time: float
            Onset time of region where searching for spike. Defined as a crossing 
            of a *threshold* or *baseline* in dv2/dt.
        peak_time: float
            time of spike peak
        max_slope_time:
            time where the voltage is changing most rapidly (i.e. dvdt = 0) 
        peak_value: float
            None if peak_time is None else trace.value_at(peak_time),
        max_slope: float

    """
    if not isinstance(trace, TSeries):
        raise TypeError("data must be Trace instance.")

    if ui is not None:
        ui.clear()
        ui.console.setStack()
        ui.plt1.plot(trace.time_values, trace.data)

    assert trace.data.ndim == 1

    pulse_edges = np.array(tuple(map(float, pulse_edges)))  # confirms pulse_edges is [float, float]
    window_edges = pulse_edges + pulse_bounds_move

    #---------------------------------------------------------
    #----this is were vc and ic code diverge------------------
    #---------------------------------------------------------

    # calculate derivatives within pulse window
    diff1 = trace.time_slice(window_edges[0], window_edges[1]).diff()

    # note that this isnt actually used for anything other than plotting
    diff2 = diff1.diff() 
    # mask out pulse artifacts in diff2 before lowpass filtering
    for edge in pulse_edges:
        apply_cos_mask(diff2, center=edge + 100e-6, radius=400e-6, power=2)

    # low pass filter the second derivative
    diff2 = bessel_filter(diff2, 10e3, order=4, bidir=True)
    # low pass filter the second derivative
    
    diff1 = bessel_filter(diff1, 10e3, order=4, bidir=True)

    # create a linear threshold
    dv_threshold = dv_thresholds[0] - (dv_thresholds[0] - dv_thresholds[1])/len(diff1.data)  * np.arange(len(diff1.data))
    events1 = list(threshold_events(diff1, 
                                    threshold=dv_threshold, 
                                    adjust_times=False, 
                                    omit_ends=False,
                                    debug=False))

    if ui is not None:
        ui.plt2.plot(diff1.time_values, diff1.data)
        ui.plt2.plot(diff1.time_values, dv_threshold, pen='y')
        ui.plt3.plot(diff2.time_values, diff2.data)
    
    # for each bump in dvdt, either discard the event or generate 
    # spike metrics from v and dvdt  
    spikes = []

    slope_hit_boundry = False
    for ev in events1:
        if ev['sum'] < 0:
            continue

        onset_time = ev['time'] #this number is arbitrary 

        # don't double-count spikes 
        if len(spikes) > 0 and onset_time < spikes[-1]['onset_time'] + double_spike:
            continue

        #max slope must be within the event.
        max_slope_time, is_edge = max_time(diff1.time_slice(onset_time, min(onset_time + ev['duration'] + diff1.dt, window_edges[1]))) #window edges will not be relevant if the duration is used
        max_slope = diff1.value_at(max_slope_time)

        if is_edge ==-1:
            #if it is an edge at the beginning of the pulse  V is most likely just increasing from pulse initiation  
            continue
        #this should only happen at the end of pulse window since we are looking at dvdt in events
        if is_edge == 1:
            slope_hit_boundry = True
            peak_time = None # if slope is hitting the pulse window boundry don't trust the peak
        else: 
            slope_hit_boundry = False
            peak_time, is_edge = max_time(trace.time_slice(onset_time, min(max_slope_time + 1.e-3, window_edges[1]))) 
            if is_edge != 0 or peak_time > window_edges[1]:
                # peak is obscured by pulse edge
                peak_time = None
        
        spikes.append({
            'onset_time': onset_time,
            'peak_time': peak_time,
            'max_slope_time': max_slope_time,
            'peak_value': None if peak_time is None else trace.value_at(peak_time),
            'max_slope': max_slope,
        })


    """check for evidence of spike in the decay after the pulse if
    1. there are no previous spikes in the trace. Note this is specifically here so don't get an error from spikes[-1] if it is empty
    2. there are no previous spikes within 1 ms of the boundry
    3. a potential spike was found but a clear max slope was not found because it is hitting window boundry (set where 
       artifact begins). However, if the max_slope is > 80 it is likely that the spike is too fast to observe in the decay so the values
       in the spike dictionary remain.  If the max_slope at the boundry is less than 80, the algorigthm checks to see if a spike is
       detected in the decay.  If the slope is not > 80 it checks to see if the max slope after the artifact is also a boundry, if it is 
       it takes the largest value of the two, if not it replaces the last spike with new values
    """    
    if (len(spikes) == 0) or (slope_hit_boundry and (spikes[-1]['max_slope'] < 80.)) or (spikes[-1]['max_slope_time'] < (window_edges[1] - double_spike)): #last spike is more than 1 ms before end

        #TODO: resolve different trace types and filtering
        dv_after_pulse = trace.time_slice(window_edges[1] + artifact_width, None).diff() #taking non filtered data
        dv_after_pulse = bessel_filter(dv_after_pulse, 15e3, bidir=True) #does this really need a different filtering

        # create a vector to fit
        ttofit = dv_after_pulse.time_values  # setting time to start at zero, note: +1 because time trace of derivative needs to be one shorter
        ttofit = ttofit - ttofit[0]

        # do fit and see if it matches
        popt, pcov = curve_fit(rc_decay, ttofit, dv_after_pulse.data, maxfev=10000)
        fit = rc_decay(ttofit, *popt)
        diff = dv_after_pulse - fit
        mse = (diff.data**2).mean()  # mean squared error
        if ui is not None:
            ui.plt2.plot(dv_after_pulse.time_values, dv_after_pulse.data, pen='b')
            if mse > mse_threshold:        
                ui.plt2.plot(dv_after_pulse.time_values, fit, pen='g')
            else: 
                ui.plt2.plot(dv_after_pulse.time_values, fit, pen='r')

        # there is a spike, so identify paramters
        if mse > mse_threshold:
            onset_time = window_edges[1] + artifact_width
            max_slope_time, slope_is_edge = max_time(dv_after_pulse.time_slice(onset_time, onset_time + 0.5e-3)) #can't search far out because slope might have dropped and then peak will be found far away.
            max_slope = dv_after_pulse.value_at(max_slope_time)
            peak_time, peak_is_edge = max_time(trace.time_slice(max_slope_time,  max_slope_time + 1e-3))
            peak_value = dv_after_pulse.value_at(peak_time)
#            import pdb; pdb.set_trace()
            if peak_is_edge != 0:
                peak_time = None
                peak_value = None

            if len(spikes) == 0: #if there are no previous values in the spike array
                # append the newly found values
                pass
            else:  #there was a value in spike array
                #if max slopes are on the boundry on each side of the artifact
                if slope_is_edge and slope_hit_boundry: 
                    # check if the slope before the artifact was larger than after the artifact
                    if spikes[-1]['max_slope'] > max_slope:  
                        # set values to the values before artifact
                        onset_time = spikes[-1]['onset_time'] # set the onset to be before the artifact 
                        peak_time = spikes[-1]['peak_time']
                        peak_value = spikes[-1]['peak_value']
                        max_slope = spikes[-1]['max_slope']
                        max_slope_time = spikes[-1]['max_slope_time']
                    spikes.pop(-1) # get rid of the last values for spike because they will be updated

                # if there is an obvous minimum after the artifact and there is a previous value in the spike array that hit a boundry
                elif slope_hit_boundry and not slope_is_edge: 
                    # get rid of the last recorded spike so it can be replaced
                    spikes.pop(-1) #so get rid of the last value in spike set off before artifact but not completed because it hit the boundry looking for slope time

                #there was not a spike close to the boundry so no need to get rid of last spike
                elif not slope_hit_boundry and slope_is_edge: 
                    pass
                elif not slope_hit_boundry and not slope_is_edge:
                    #there is no spike close to the boundry and there is an obvious min after the artifact
                    pass
                else:
                    raise Exception('this has not been accounted for')

            # add the found spike to the end
            spikes.append({
                'onset_time': onset_time,
                'max_slope_time': max_slope_time,
                'peak_time': peak_time,
                'peak_value':peak_value,
                'max_slope': max_slope
                })
        
        else:
            # there is no spike found from the decay 
            if slope_hit_boundry:  
                # if last spike recorded was against the bound delete it
                spikes.pop(-1)


    for spike in spikes:    
        # max_slope_time is how we define spike initiation, so, make sure it is defined.
        assert 'max_slope_time' in spike
    return spikes


def detect_vc_evoked_spikes(trace,
                            pulse_edges,
                            pulse_bounds_move=[.1e-3, 0.02e-3],  #0.03e-3 sometimes gets some artifact
                            ui=None):
    """Return a dict describing an evoked spike in a patch clamp recording, or None if no spike is detected.

    This function assumes that a square voltage pulse is used to evoke an unclamped spike
    in a voltage clamp recording, and that the peak of the unclamped spike occurs *during*
    the stimulation pulse.

    Parameters
    ==========
    trace : TSeries instance
        The recorded patch clamp data. The recording should be made with a brief pulse intended
        to evoke a single spike with short latency.
    pulse_edges : (float, float)
        The start and end times of the stimulation pulse, relative to the timebase in *trace*.
    ui:                                                                                                                            
        user interface for viewing spike detection 
    
    Returns 
    =======
    spikes: dictionary
        contains following information about spikes:
        onset_time: float
            Onset time of region where searching for a spike (identified in event_detection 
            module). Defined as a crossing of a *threshold* or *baseline* in dv2/dt.
        peak_time: float
            time of spike peak
        max_slope_time:
            time where the voltage is changing most rapidly (i.e. max or min of dvdt) 
        peak_value: float
            None if *peak_time* is None, else *trace.value_at(peak_time)*
        max_slope: float
    """
    if not isinstance(trace, TSeries):
        raise TypeError("data must be TSeries instance.")

    if ui is not None:
        ui.clear()
        ui.console.setStack()
        ui.plt1.plot(trace.time_values, trace.data)

    assert trace.ndim == 1
    pulse_edges = np.array(tuple(map(float, pulse_edges)))  # make sure pulse_edges is (float, float)
    window_edges = pulse_edges + pulse_bounds_move
    
    # calculate derivatives within pulse window
    diff1 = trace.time_slice(window_edges[0], window_edges[1]).diff()
    diff2 = diff1.diff()

    # crop and filter diff1
    diff1 = bessel_filter(diff1, cutoff=20e3, order=4, btype='low', bidir=True)

    # crop and low pass filter the second derivative
    diff2 = diff2.time_slice(pulse_edges[0] + 150e-6, pulse_edges[1])
    # mask out pulse artifacts in diff2 before lowpass filtering
    #for edge in pulse_edges:
    #   apply_cos_mask(diff2, center=edge + 100e-6, radius=400e-6, power=2)
    diff2 = bessel_filter(diff2, 10e3, order=4, bidir=True)
    # chop off ending transient
    diff2 = diff2.time_slice(None, diff2.t_end)

    # look for negative bumps in second derivative
    # dv1_threshold = 1e-6
    dv2_threshold = 0.02

    #=========================================================================
    debug = False
    #=========================================================================

    events = list(threshold_events(diff2 / dv2_threshold, 
        threshold=1.0, adjust_times=False, omit_ends=True, debug=debug))



    if ui is not None:
        ui.plt2.plot(diff1.time_values, diff1.data)
        # ui.plt2.plot(diff1_hp.time_values, diff1.data)
        # ui.plt2.addLine(y=-dv1_threshold)
        ui.plt3.plot(diff2.time_values, diff2.data)
        ui.plt3.addLine(y=dv2_threshold)
        # ui.plt3.plot(diff2.time_values, diff2.data/dv2_threshold)
        # ui.plt3.addLine(y=1)

    if len(events) == 0:
        return []

    # for each bump in d2vdt, either discard the event or generate 
    # spike metrics from v and dvdt

    spikes = []
    for ev in events:
        if np.abs(ev['sum']) < 2.:
            continue
        if ev['sum'] > 0 and ev['peak'] < 5. and ev['time'] < diff2.t0 + 60e-6:
            # ignore positive bumps very close to the beginning of the trace
            continue
        if len(spikes) > 0 and ev['peak_time'] < spikes[-1]['max_slope_time'] + 1e-3:
            # ignore events that follow too soon after a detected spike
            continue

        #TODO: What is this doing?
        # if ev['sum'] < 0:
        #     onset_time = ev['peak_time']
        #     search_time = onset_time
        # else:
        #     search_time = ev['time'] - 200e-6
        #     onset_time = ev['peak_time']  

        if ev['sum'] < 0:
            # only accept positive bumps
            continue
        else:
            search_time = ev['time'] - 100e-6
            onset_time = search_time


        max_slope_rgn = diff1.time_slice(search_time, search_time + 0.5e-3)
        max_slope_time, is_edge = min_time(max_slope_rgn) #note this is looking for min because event must be negative in VC
        max_slope = diff1.value_at(max_slope_time)

        if max_slope > 0:
            # actual slope must be negative at this point
            # (above we only tested the sign of the high-passed event)
            continue
        
        peak_search_rgn = trace.time_slice(max_slope_time, min(pulse_edges[1], search_time + 1e-3))

        if len(peak_search_rgn) == 0:
            peak = None
            peak_time = None
        else:
            peak_time, is_edge = min_time(peak_search_rgn)
            if is_edge:
                peak = None
                peak_time = None
            else:
                peak = trace.time_at(peak_time)

        spikes.append({
            'onset_time': onset_time,
            'max_slope_time': max_slope_time,
            'max_slope': max_slope,
            'peak_time': peak_time,
            'peak_value': peak,
        })

    # remove spikes where the same values were found from two different events
    # it is probable that this would no happen in negative bumps in dv2 where 
    # ignored.
    # if len(spikes) > 1:
    #     slopes=[]

    #     # find unique values of slope
    #     for spike in spikes:   
    #         slopes.append(spike['max_slope_time'])
    #     uq = np.unique(slopes)

    #     # if there are less unique values than spikes there are duplicates
    #     if len(uq)!=len(spikes):
    #         #find indicies to remove
    #         remove_indicies=[]
    #         for ii in range(1,len(slopes)):
    #             for jj in range(ii): 
    #                 if slopes[ii] == slopes[jj]:
    #                     remove_indicies.append(ii)
    #         #remove the duplicate spikes
    #         for ii in remove_indicies:
    #             del spikes[ii]  

    for spike in spikes:
        assert 'max_slope_time' in spike
    return spikes


def apply_cos_mask(trace, center, radius, power):
    """Multiply a region of a trace by a cosine mask to dampen artifacts without generating 
    sharp edges.
    
    The input *trace* is modified in-place.
    """
    chunk = trace.time_slice(center - radius, center + radius)
    w1 = np.pi * (chunk.t0 - center) / radius
    w2 = np.pi * (chunk.t_end - center) / radius
    mask_t = np.pi + np.linspace(w1, w2, len(chunk))
    mask = ((np.cos(mask_t) + 1) * 0.5) ** power
    chunk.data[:] = chunk.data * mask


def max_time(trace):
    """Return the time of the maximum value in the trace, and a value indicating whether the
    time returned coincides with the first value in the trace (-1), the last value in the
    trace (1) or neither (0).
    """
    ind = np.argmax(trace.data)
    if ind == 0:
        is_edge = -1
    elif ind == len(trace) - 1:
        is_edge = 1
    else:
        is_edge = 0
    return trace.time_at(ind), is_edge


def min_time(trace):
    """Return the time of the minimum value in the trace, and a value indicating whether the
    time returned coincides with the first value in the trace (-1), the last value in the
    trace (1) or neither (0).
    """
    ind = np.argmin(trace.data)
    if ind == 0:
        is_edge = -1
    elif ind == len(trace) - 1:
        is_edge = 1
    else:
        is_edge = 0
    return trace.time_at(ind), is_edge


class SpikeDetectTestCase(DataTestCase):
    def __init__(self):
        DataTestCase.__init__(self, detect_evoked_spikes)

    def check_result(self, result):
        for spike in result:
            assert 'max_slope_time' in spike
            assert 'onset_time' in spike
            assert 'peak_time' in spike
        DataTestCase.check_result(self, result)

    @property
    def name(self):
        meta = self.meta
        return "%s_%s_%s_%0.3f" % (meta['expt_id'], meta['sweep_id'], meta['device_id'], self.input_args['pulse_edges'][0])
