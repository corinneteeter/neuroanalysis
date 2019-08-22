from __future__ import division
import numpy as np
from .data import TSeries


def zero_crossing_events(data, min_length=3, min_peak=0.0, min_sum=0.0, noise_threshold=None):
    """Locate events of any shape in a signal. Works by finding regions of the signal
    that deviate from noise, and filtering with multiple criteria such as peak value,
    area under the event, and duration of the event.
    
    This algorithm relies on having data where the signal is lower frequency than the noise.
    Any low frequency noise must be removed by high-pass filtering to produce a flat baseline.
    In addition, the event amplitude must be sufficiently large that the signal does not 
    cross 0 within a single event (low-pass filtering may be necessary to achive this).
    
    Returns an array of events where each row is (start, length, sum, peak)
    """
    
    if isinstance(data, TSeries):
        xvals = data.time_values
        data1 = data.data
    else:
        xvals = None
        data1 = data.view(np.ndarray)
    
    ## find all 0 crossings
    mask = data1 > 0
    diff = mask[1:] - mask[:-1]  ## mask is True every time the trace crosses 0 between i and i+1
    times1 = np.argwhere(diff)[:, 0]  ## index of each point immediately before crossing.
    
    times = np.empty(len(times1)+2, dtype=times1.dtype)  ## add first/last indexes to list of crossing times
    times[0] = 0                                         ## this is a bit suspicious, but we'd rather know
    times[-1] = len(data1)                               ## about large events at the beginning/end
    times[1:-1] = times1                                 ## rather than ignore them.
    #p.mark('find crossings')
    
    ## select only events longer than min_length.
    ## We do this check early for performance--it eliminates the vast majority of events
    long_events = np.argwhere(times[1:] - times[:-1] > min_length)
    if len(long_events) < 1:
        n_events = 0
    else:
        long_events = long_events[:, 0]
        n_events = len(long_events)
    
    ## Measure sum of values within each region between crossings, combine into single array
    if xvals is None:
        events = np.empty(n_events, dtype=[('index',int),('len', int),('sum', float),('peak', float)])  ### rows are [start, length, sum]
    else:
        events = np.empty(n_events, dtype=[('index',int),('time',float),('len', int),('sum', float),('peak', float)])  ### rows are [start, length, sum]
    #p.mark('empty %d -> %d'% (len(times), n_events))
    #n = 0
    for i in range(n_events):
        t1 = times[long_events[i]]+1
        t2 = times[long_events[i]+1]+1
        events[i]['index'] = t1
        events[i]['len'] = t2-t1
        evData = data1[t1:t2]
        events[i]['sum'] = evData.sum()
        if events[i]['sum'] > 0:
            peak = evData.max()
        else:
            peak = evData.min()
        events[i]['peak'] = peak
    #p.mark('generate event array')
    
    if xvals is not None:
        events['time'] = xvals[events['index']]
    
    if noise_threshold > 0:
        ## Fit gaussian to peak in size histogram, use fit sigma as criteria for noise rejection
        stdev = measureNoise(data1)
        #p.mark('measureNoise')
        hist = histogram(events['sum'], bins=100)
        #p.mark('histogram')
        histx = 0.5*(hist[1][1:] + hist[1][:-1]) ## get x values from middle of histogram bins
        #p.mark('histx')
        fit = fitGaussian(histx, hist[0], [hist[0].max(), 0, stdev*3, 0])
        #p.mark('fit')
        sigma = fit[0][2]
        minSize = sigma * noise_threshold
        
        ## Generate new set of events, ignoring those with sum < minSize
        #mask = abs(events['sum'] / events['len']) >= minSize
        mask = abs(events['sum']) >= minSize
        #p.mark('mask')
        events = events[mask]
        #p.mark('select')

    if min_peak > 0:
        events = events[abs(events['peak']) > min_peak]
    
    if min_sum > 0:
        events = events[abs(events['sum']) > min_sum]
    
    return events

def _deal_unbalanced_initial_off(omit_ends, on_inds, off_inds):
    """Deals with situation where there is an "off" crossing from above to below threshold
    at the beginning of a trace without there first being an "on" crossing from below to above 
    threshold.  Note that the usage of this function is looking for extreme regions
    where a trace is below a negative threshold or above a positive threshold, thus, the 
    sign of the trace value at *on_inds* and *off_inds* can be positive or negative
    """
    if not omit_ends:
        on_inds = [0] + on_inds #prepend the edge as on on ind
    else:
        off_inds = off_inds[1:] #remove the off ind
    return on_inds, off_inds

def _deal_unbalanced_termination_on(omit_ends, on_inds, off_inds, off_to_add):
    """Deals with situation where there is an "on" crossing from below to above threshold
    toward the end of a trace without an "off" crossing happening thereafter.  Note that 
    the usage of this function is looking for extreme regions
    where a trace is below a negative threshold or above a positive threshold, thus, the 
    sign of the trace value at *on_inds* and *off_inds* can be positive or negative
    """
    if not omit_ends:
        off_inds = off_inds + [off_to_add] #append the index of the last data point
    else:
        on_inds = on_inds[:-1] #remove the last on indicie
    return on_inds, off_inds


def threshold_events(trace, threshold, adjust_times=True, baseline=0.0, omit_ends=True):
    """
    Finds regions in a trace that cross a threshold value (as measured by distance from baseline) and then 
    recross threshold ('bumps').  If a threshold is crossed at the end of the trace, an event may be excluded
    or the beginning/end may be used as the the start/end of the event (depending on the value of *omit_ends*). 


    Parameters                                                                                                                                     
    ==========
    trace: *Tseries* instance                            
    threshold: float or np.array with dimensions of *trace.data*
        Algorithm checks if waveform crosses both positive and negative *threshold* symetrically 
        around from the y-axis.  i.e. if -5. is provided, the algorithm looks for places where 
        the waveform crosses +/-5. If an array is provided, each index of the *threshold* will 
        be compared with the data pointwise.
    adjust_times: boolean
        If True, move the start and end times of the event outward, estimating the zero-crossing point for the event
    baseline: float
        Value subtracted from the data.
    omit_ends: boolean
        If true, add the trace endpoint indices to incomplete events, i.e., events that started above threhold at the 
        beginning of trace, or crossed threshold but did not return below threshold at the end of a trace.  If false, 
        remove the imcomplete events.

    
    Returns
    =======
    events: numpy structured array.  
        An event ('bump') is a region of the *trace.data* waveform that crosses above *threshold* and then falls below 
        threshold again. Each index contains information about an event.  Fields as follows:
        index: int
           index of the initial crossing of the *threshold*
        len: int
            index length of the event 
        sum: float
           sum of the values in the array between the start and end of the event
        peak: float
           peak value of event
        peak_index: int
           index value of the peak
        time: float, or np.nan if timing not available
           time of the onset of an event
        duration: float, or np.nan if timing not available
           duration of time of the event
        area: float, or np.nan if timing not available
           area under the curve of the event
        peak_time: float, or np.nan if timing not available
           time of peak
    """


    data = trace.data
    data1 = data - baseline
    
    # convert threshold array
    if isinstance(threshold, float):
        threshold = np.ones(len(data)) * abs(threshold)
    
    ## find all threshold crossings in both positive and negative directions
    ## deal with imcomplete events, and store events
    
    # -1 (or +1) when crosses from above to below threshold (or visa versa if threshold is negative). Note above threshold refers to value furthest from zero, i.e. it can be positive or negative
    masks = [(data1 > threshold).astype(np.byte), (data1 < -threshold).astype(np.byte)] 
    
    hits = []
    for mask in masks:
        diff = mask[1:] - mask[:-1]
        # indices where crosses from below to above threshold ('on')
        on_inds = list(np.argwhere(diff==1)[:,0] + 1)  
        # indices where crosses from above to below threshold ('off') 
        off_inds = list(np.argwhere(diff==-1)[:,0] + 1) 
        
        # deal with cases when there are unmatched on and off indicies 
        if len(off_inds) > 0:  #if there are some off indicies
            if len(on_inds) > 0: #and there are also on indicies
                if on_inds[0] > off_inds[0]: #check if off happens before on
                    on_inds, off_inds = _deal_unbalanced_initial_off(omit_ends, on_inds, off_inds)
            else: #there are no on indicies
                on_inds, off_inds = _deal_unbalanced_initial_off(omit_ends, on_inds, off_inds)

        if len(on_inds) > 0:  #if there are some on indicies
            if len(off_inds) > 0: #and there are also off indicies
                if on_inds[-1] > off_inds[-1]: #check if off happens before on
                    on_inds, off_inds = _deal_unbalanced_termination_on(omit_ends, on_inds, off_inds, len(data1))
            else: #there are no off indicies
                on_inds, off_inds = _deal_unbalanced_termination_on(omit_ends, on_inds, off_inds, len(data1))


        # at this point every 'on' should have and 'off'
        assert len(on_inds) == len(off_inds)

        # put corresponding on and off indeces in a list
        for i in range(len(on_inds)):
            if on_inds[i] == off_inds[i]:
                #something wierd happened
                continue
            hits.append((on_inds[i], off_inds[i]))
    
    ## sort hits  ## NOTE: this can be sped up since we already know how to interleave the events..
    hits.sort(key=lambda a: a[0])
    
    n_events = len(hits)
    events = np.empty(n_events, dtype=[
        ('index', int),
        ('len', int),
        ('sum', float),
        ('peak', float),
        ('peak_index', int),
        # only used if timing is available:
        ('time', float),
        ('duration', float),
        ('area', float),
        ('peak_time', float),
    ])
    
    ## Lots of work ahead:
    ## 1) compute length, peak, sum for each event
    ## 2) adjust event times if requested, then recompute parameters
    for i in range(n_events):
        ind1, ind2 = hits[i]
        ln = ind2 - ind1
        ev_data = data1[ind1:ind2]
        sum = ev_data.sum()
        if sum > 0:
            peak_ind = np.argmax(ev_data)
        else:
            peak_ind = np.argmin(ev_data)
        peak = ev_data[peak_ind]
        peak_ind += ind1
            
        #print "event %f: %d" % (xvals[ind1], ind1) 
        if adjust_times:  ## Move start and end times outward, estimating the zero-crossing point for the event
        
            ## adjust ind1 first
            mind = np.argmax(ev_data)
            pdiff = abs(peak - ev_data[0])
            if pdiff == 0:
                adj1 = 0
            else:
                adj1 = int(threshold * mind / pdiff)
                adj1 = min(ln, adj1)
            ind1 -= adj1
            
            ## check for collisions with previous events
            if i > 0:
                lind2 = hits[i-1][1]
                if ind1 < lind2:
                    diff = lind2-ind1   ## if events have collided, force them to compromise
                    tot = adj1 + last_adj
                    if tot != 0:
                        d1 = diff * float(last_adj) / tot
                        d2 = diff * float(adj1) / tot
                        hits[i-1] = (int(hits[i-1][0]), int(hits[i-1][1]-(d1+1)))
                        ind1 += d2
            
            ## adjust ind2
            mind = ln - mind
            pdiff = abs(peak - ev_data[-1])
            if pdiff == 0:
                adj2 = 0
            else:
                adj2 = int(threshold * mind / pdiff)
                adj2 = min(ln, adj2)
            ind2 += adj2
            last_adj = adj2
            
        hits[i] = (int(ind1), int(ind2))
        events[i]['peak'] = peak
        events[i]['index'] = ind1
        events[i]['peak_index'] = peak_ind
        events[i]['len'] = ln
        events[i]['sum'] = sum
        
    if adjust_times:  ## go back and re-compute event parameters.
        mask = np.ones(n_events, dtype=bool)
        for i in range(n_events):
            ind1, ind2 = hits[i]
            
            ln = ind2 - ind1
            ev_data = data1[ind1:ind2]
            sum = ev_data.sum()
            if len(ev_data) == 0:
                mask[i] = False
                continue
            if sum > 0:
                peak_ind = np.argmax(ev_data)
            else:
                peak_ind = np.argmin(ev_data)
            peak = ev_data[peak_ind]
            peak_ind += ind1
                
            events[i]['peak'] = peak
            events[i]['index'] = ind1
            events[i]['peak_index'] = peak_ind
            events[i]['len'] = ln
            events[i]['sum'] = sum
    
        ## remove masked events
        events = events[mask]

    # add in timing information if available:
    if trace.has_timing:
        for ev in events:
            i1 = ev['index']
            i2 = i1 + ev['len']
            ev['time'] = trace.time_at(i1)
            ev['duration'] = trace.time_at(i2) - ev['time']
            ev['area'] = np.trapz(y=data1[i1:i2], x=trace.time_values[i1:i2])
            ev['peak_time'] = trace.time_at(ev['peak_index'])
    else:
        ev['time'] = np.nan
        ev['duration'] = np.nan
        ev['area'] = np.nan
        ev['peak_time'] = np.nan

    return events


def rolling_sum(data, n):
    """A sliding-window filter that returns the sum of n source values for each
    value in the output array::
    
        output[i] = sum(input[i:i+n])
    
    """
    d1 = np.cumsum(data)
    d2 = np.empty(len(d1) - n + 1, dtype=data.dtype)
    d2[0] = d1[n-1]  # copy first point
    d2[1:] = d1[n:] - d1[:-n]  # subtract
    return d2
    

def clements_bekkers(data, template):
    """Sliding, scale-invariant template matching algorithm.

    Slides a template along a signal, measuring their similarity and scale difference
    at each sample. Similar to deconvolution.
    
    Parameters
    ----------
    data : array
        The signal data to process
    template : array
        A template 
    
    Returns
    -------
    detection_criterion : array
        Indicates how well the template matches the signal at each sample.
    scale : array
        The scale factor providing the best fit between template and signal at each sample.
    offset : array
        The y-offset of the best template fit at each sample.
    
    Notes
    -----
    Fast and sensitive for well-isolated events but performs poorly on overlapping events.
    See Clements & Bekkers, Biophysical Journal, 73: 220-229, 1997.
    """
    # Strip out meta-data for faster computation
    D = data.view(ndarray)
    T = template.view(ndarray)
    
    # Prepare a bunch of arrays we'll need later
    N = len(T)
    sumT = T.sum()
    sumT2 = (T**2).sum()
    sumD = rolling_sum(D, N)
    sumD2 = rolling_sum(D**2, N)
    sumTD = correlate(D, T, mode='valid')
    
    # compute scale factor, offset at each location:
    scale = (sumTD - sumT * sumD / N) / (sumT2 - sumT**2 / N)
    offset = (sumD - scale * sumT) / N
    
    # compute SSE at every location
    SSE = sumD2 + scale**2 * sumT2 + N * offset**2 - 2 * (scale*sumTD + offset*sumD - scale*offset*sumT)
    
    # finally, compute error and detection criterion
    error = sqrt(SSE / (N-1))
    DC = scale / error
    return DC, scale, offset


def exp_deconvolve(trace, tau):
    """Exponential deconvolution used to isolate overlapping events; works nicely on PSPs, calcium transients, etc.

    See: Richardson & Silberberg 2008, "Measurement and Analysis of Postsynaptic Potentials Using 
         a Novel Voltage-Deconvolution Method"
    """
    dt = trace.dt
    arr = trace.data
    deconv = arr[:-1] + (tau / dt) * (arr[1:] - arr[:-1])
    if trace.has_time_values:
        # data is one sample shorter; clip time values to match.
        return trace.copy(data=deconv, time_values=trace.time_values[:-1])
    else:
        return trace.copy(data=deconv)
    
def exp_reconvolve(trace, tau):
    # equivalent to subtracting an exponential decay from the original unconvolved signal
    dt = trace.dt
    d = np.zeros(trace.data.shape, trace.data.dtype)
    dtt = dt / tau
    dtti = 1. - dtt
    for i in range(1, len(d)):
        d[i] = dtti * d[i-1] + dtt * trace.data[i-1]
    return trace.copy(data=d)
