import numpy as np
import scipy as sp
import os, sys
import matplotlib.pyplot as plt

from scipy.optimize import leastsq
import astropy.io.fits as fits
from astropy.table import Table

try:
    from bottleneck import median
    from bottleneck import nansum
except:
    from numpy import median
    from numpy import nansum
    print("Not Using bottleneck: Speed will be improved if you install bottleneck")



#  Number of cuts of the data we look at.  More takes more time.
NumSpectralCuts = 101

# Start and end range of area used to look for a valid cut.  As a
#  proportion.
PixelCutStart=0.25
PixelCutEnd=0.75

# Probe Radius, used for drawing. Microns.
ProbeRadius=52.5

def count_enabled(binaryTable, quiet):
    if not quiet:
        print("---> Counting enabled fibres")

    a1=np.asarray(binaryTable.field('SELECTED')==1).nonzero()
    return np.shape(a1)[1]
    
def get_probe_list(binaryTable, quiet):
    if not quiet:
        print("---> Getting probe list")


    a1=np.asarray(binaryTable.field('SELECTED')==1).nonzero()
    a2=np.asarray(binaryTable.field('TYPE')    =='P').nonzero()

    return np.unique(binaryTable.field('PROBENAME')[a1 and a2])

def comxyz(x,y,z):
    """Centre of mass given x, y and z vectors (all same size). x,y give position which has value z."""

    Mx=0
    My=0
    mass=0

    for i in range(len(x)):
        Mx=Mx+x[i]*z[i]
        My=My+y[i]*z[i]
        mass=mass+z[i]

    com=(Mx/mass, My/mass)
    return com


def process_object_file(object_file, sigma_clip, cut_loc, peaks, expectedPeaks, pix_waveband, quiet):
    if not quiet:
        print("---> Processing object file:", object_file)
    # Location of fibre peaks for linear tramline
    tram_loc=[]
    for i in np.arange(np.shape(peaks[0])[0]):
        tram_loc.append(peaks[0][i][0])
    
    # Import object frame
    object = pf.open(object_file)
    object_data = object['Primary'].data
    object_fibtab = object['MORE.FIBRES_IFU'].data
    try:
        object_guidetab = object['MORE.FIBRES_GUIDE'].data
        object_guidetab = object_guidetab[object_guidetab['TYPE']=='G']
    except:
        object_guidetab = None

    enabledFibres = count_enabled(object_fibtab, quiet)
    if enabledFibres != expectedPeaks:
        raise SystemExit("The flat field had "+str(expectedPeaks)+" enabled fibres, but the object has "+str(enabledFibres)+ ". Look like they are different configurations!")

    # Perform cut along spatial direction at same position as cut_loc
    s = np.shape(object_data)
    object_cut = object_data[:,int((s[1]*cut_loc)-pix_waveband/2):int((s[1]*cut_loc)+pix_waveband/2)]
    
    # "Sigma clip" to get set bad pixels as row median value
    if sigma_clip == True:
        if not quiet:
            print("---> Performing 'Sigma-clip'... (~20 seconds)")
        n=np.arange(np.shape(object_cut)[0])[-1]
        #print("n=")
        for i in np.arange(np.shape(object_cut)[0]):
            #if i % 100 == 0 and not quiet:
            #    print("---> Sigma progress:", i, "/", n)
            for j in np.arange(np.shape(object_cut)[1]):
                med = median(object_cut[i,:])
                err = np.absolute((object_cut[i,j]-med)/med)
                if err > 0.25:
                    object_cut[i,j] = med
        if not quiet:
            print("---> Sigma-clip complete")
     
    # Collapse spectral dimension
    object_cut_sum = nansum(object_cut,axis=1)
    
    # Extract intensities at fibre location.  Will have 819 entries in object_spec
    object_spec = object_cut_sum[tram_loc]

    return object_spec,object_fibtab,object_guidetab

def process_flat(flat_data, pix_waveband, pix_start, expectedPeaks, NumSpectralPixels, quiet):

    # Range to find spatial cut
    if (pix_start != "unknown") and (pix_start != 0) :
        cut_loc_start = np.float(pix_start+5)/np.float(NumSpectralPixels)
        cut_locs = np.linspace(cut_loc_start,PixelCutEnd,NumSpectralCuts)
    else:
        cut_locs = np.linspace(PixelCutStart,PixelCutEnd,NumSpectralCuts)

    if quiet:
        print("Determining spatial cut")
    else:
        print("---> Finding suitable cut along spatial dimension...")

    # Check each spatial slice until "expectedPeaks" fibres (peaks) have been found
    for cut_loc in cut_locs:
        # perform cut along spatial direction
        flat_cut = flat_data[:,int(np.shape(flat_data)[1]*cut_loc)]
        flat_cut_leveled = flat_cut - 0.1*np.max(flat_cut)
        flat_cut_leveled[flat_cut_leveled < 0] = 0.
        # find peaks (fibres)
        peaks = peakdetect(flat_cut_leveled, lookahead = 3)
        Npeaks = np.shape(peaks[0])[0]
        if Npeaks == expectedPeaks:
            break
        else:
            continue
    
        if not quiet:
            print("--->")
    

    if Npeaks != expectedPeaks:
        raise SystemExit("---> Can't find "+str(expectedPeaks)+", found "+str(Npeaks)+". Check [1] Flat Field is correct [2] Flat Field is supplied as the first variable in the function. If 1+2 are ok then use the 'pix_start' variable and set it at least 10 pix beyond the previous value (see terminal for value)");
        
    if quiet:
        print("Spatial cut at:",int(cut_loc*NumSpectralPixels),", Waveband:",pix_waveband, ", Num fibres:", np.shape(peaks[0])[0])
    else:
        print("---> Spatial cut at pixel number: ",int(cut_loc*NumSpectralPixels))
        print("---> Number of waveband pixels: ",pix_waveband)
        print("---> Number of fibres found: ",np.shape(peaks[0])[0])
        print("--->")


    return cut_loc,peaks



def process_object_file(object_file, sigma_clip, cut_loc, peaks, expectedPeaks, pix_waveband, quiet):

    if not quiet:
        print("---> Processing object file:", object_file)
    # Location of fibre peaks for linear tramline
    tram_loc=[]
    for i in np.arange(np.shape(peaks[0])[0]):
        tram_loc.append(peaks[0][i][0])
    
    # Import object frame
    object = fits.open(object_file)
    object_data = object['Primary'].data
    object_fibtab = object['MORE.FIBRES_IFU'].data
    try:
        object_guidetab = object['MORE.FIBRES_GUIDE'].data
        object_guidetab = object_guidetab[object_guidetab['TYPE']=='G']
    except:
        object_guidetab = None

    enabledFibres = count_enabled(object_fibtab, quiet)
    if enabledFibres != expectedPeaks:
        raise SystemExit("The flat field had "+str(expectedPeaks)+" enabled fibres, but the object has "+str(enabledFibres)+ ". Look like they are different configurations!")

    # Perform cut along spatial direction at same position as cut_loc
    s = np.shape(object_data)
    object_cut = object_data[:,int((s[1]*cut_loc)-pix_waveband/2):int((s[1]*cut_loc)+pix_waveband/2)]
    
    # "Sigma clip" to get set bad pixels as row median value
    if sigma_clip == True:
        if not quiet:
            print("---> Performing 'Sigma-clip'... (~20 seconds)")
        n=np.arange(np.shape(object_cut)[0])[-1]
        #print("n=")
        for i in np.arange(np.shape(object_cut)[0]):
            #if i % 100 == 0 and not quiet:
            #    print("---> Sigma progress:", i, "/", n)
            for j in np.arange(np.shape(object_cut)[1]):
                med = median(object_cut[i,:])
                err = np.absolute((object_cut[i,j]-med)/med)
                if err > 0.25:
                    object_cut[i,j] = med
        if not quiet:
            print("---> Sigma-clip complete")
     
    # Collapse spectral dimension
    object_cut_sum = nansum(object_cut,axis=1)
    
    # Extract intensities at fibre location. 
    object_spec = object_cut_sum[tram_loc]

    return object_spec,object_fibtab,object_guidetab

def peakdetect(y_axis, x_axis = None, lookahead = 300, delta=0):
    
    """
    #
    # "peakdetect"
    #
    #   Determines peaks from data. Translation of the MATLAB code "peakdet.m"
    #   and taken from https://gist.github.com/sixtenbe/1178136
    #
    #   Called by "raw"
    #
    """
    
    i = 10000
    x = np.linspace(0, 3.5 * np.pi, i)
    y = (0.3*np.sin(x) + np.sin(1.3 * x) + 0.9 * np.sin(4.2 * x) + 0.06 * np.random.randn(i))
    
    # Converted from/based on a MATLAB script at:
    # http://billauer.co.il/peakdet.html
    
    # function for detecting local maximas and minmias in a signal.
    # Discovers peaks by searching for values which are surrounded by lower
    # or larger values for maximas and minimas respectively
    
    # keyword arguments:
    # y_axis........A list containg the signal over which to find peaks
    # x_axis........(optional) A x-axis whose values correspond to the y_axis list
    #               and is used in the return to specify the postion of the peaks.
    #               If omitted an index of the y_axis is used. (default: None)
    # lookahead.....(optional) distance to look ahead from a peak candidate to
    #               determine if it is the actual peak (default: 200)
    #               '(sample / period) / f' where '4 >= f >= 1.25' might be a good
    #               value
    # delta.........(optional) this specifies a minimum difference between a peak
    #               and the following points, before a peak may be considered a
    #               peak. Useful to hinder the function from picking up false
    #               peaks towards to end of the signal. To work well delta should
    #               be set to delta >= RMSnoise * 5. (default: 0)
    # delta.........function causes a 20% decrease in speed, when omitted.
    #               Correctly used it can double the speed of the function
    # return........two lists [max_peaks, min_peaks] containing the positive and
    #               negative peaks respectively. Each cell of the lists contains
    #               a tupple of: (position, peak_value) to get the average peak
    #               value do: np.mean(max_peaks, 0)[1] on the results to unpack
    #               one of the lists into x, y coordinates do: x, y = zip(*tab)
    #
    
    max_peaks = []
    min_peaks = []
    dump = []   #Used to pop the first hit which almost always is false
    
    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    # store data length for later use
    length = len(y_axis)
    
    #perform some checks
    if lookahead < 1:
        raise ValueError("Lookahead must be '1' or above in value")
    if not (np.isscalar(delta) and delta >= 0):
        raise ValueError("delta must be a positive number")
    
    #maxima and minima candidates are temporarily stored in
    #mx and mn respectively
    mn, mx = np.Inf, -np.Inf
    
    #Only detect peak if there is 'lookahead' amount of points after it
    for index, (x, y) in enumerate(zip(x_axis[:-lookahead], y_axis[:-lookahead])):
        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x
        
        ####look for max####
        if y < mx-delta and mx != np.Inf:
            #Maxima peak candidate found
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].max() < mx:
                max_peaks.append([mxpos, mx])
                dump.append(True)
                #set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
                continue
        
        ####look for min####
        if y > mn+delta and mn != -np.Inf:
            #Minima peak candidate found
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].min() > mn:
                min_peaks.append([mnpos, mn])
                dump.append(False)
                #set algorithm to only find maxima now
                mn = -np.Inf
                mx = -np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
    
    #Remove the false hit on the first value of the y_axis
    try:
        if dump[0]:
            max_peaks.pop(0)
        else:
            min_peaks.pop(0)
        del dump
    except IndexError:
        #no peaks were found, should the function return empty lists?
        pass
    
    return [max_peaks, min_peaks]

def _datacheck_peakdetect(x_axis, y_axis):
    """Used as part of "peakdetect" """
    if x_axis is None:
        x_axis = range(len(y_axis))
    
    if len(y_axis) != len(x_axis):
        raise ValueError("Input vectors y_axis and x_axis must have same length")
    
    #needs to be a numpy array
    y_axis = np.array(y_axis)
    x_axis = np.array(x_axis)
    return x_axis, y_axis

class BundleFitter:
    """ Fits a 2d Gaussian with PA and ellipticity. Params in form (amplitude, mean_x, mean_y, sigma_x, sigma_y,
    rotation, offset). Offset is optional. To fit a circular Gaussian use (amplitude, mean_x, mean_y, sigma, offset),
    again offset is optional. To fit a Moffat profile use (amplitude, mean_x, mean_y, alpha, beta, offset), with
    offset optional. """

    def __init__(self, p, x, y, z, model='',weights=None):
        self.p_start = p
        self.p = p
        self.x = x
        self.y = y
        self.z = z
        self.model = model
        
        if weights == None:
            self.weights = sp.ones(len(self.z))
        else:
            self.weights = weights

        self.perr = 0.
        self.var_fit = 0.

        if model == 'gaussian_eps':
            # 2d elliptical Gaussian with offset.
            self.p[0] = abs(self.p[0]) # amplitude should be positive.
            self.fitfunc = self.f1
            
        elif model == 'gaussian_eps_simple':
            # 2d elliptical Gaussian witout offset.
            self.p[0] = abs(self.p[0]) # amplitude should be positive.
            self.fitfunc = self.f2
            
        elif model == 'gaussian_circ':
            # 2d circular Gaussian with offset.
            self.p[0] = abs(self.p[0]) # amplitude should be positive.
            self.fitfunc = self.f3
            
        elif model == 'gaussian_circ_simple':
            # 2d circular Gaussian without offset
            self.p[0] = abs(self.p[0])
            self.fitfunc = self.f4
            
        elif model == 'moffat':
            # 2d Moffat profile
            self.p[0] = abs(self.p[0])
            self.fitfunc = self.f5

        elif model == 'moffat_simple':
            # 2d Moffat profile without offset
            self.p[0] = abs(self.p[0])
            self.fitfun = self.f6

        else:
            raise Exception

    def f1(self, p, x, y):
        # f1 is an elliptical Gaussian with PA and a bias level.

        rot_rad=p[5]*sp.pi/180 # convert rotation into radians.

        rc_x=p[1]*sp.cos(rot_rad)-p[2]*sp.sin(rot_rad)
        rc_y=p[1]*sp.sin(rot_rad)+p[2]*sp.cos(rot_rad)
    
        return p[0]*sp.exp(-(((rc_x-(x*sp.cos(rot_rad)-y*sp.sin(rot_rad)))/p[3])**2\
                                    +((rc_y-(x*sp.sin(rot_rad)+y*sp.cos(rot_rad)))/p[4])**2)/2)+p[6]

    def f2(self, p, x, y):
        # f2 is an elliptical Gaussian with PA and no bias level.

        rot_rad=p[5]*sp.pi/180 # convert rotation into radians.

        rc_x=p[1]*sp.cos(rot_rad)-p[2]*sp.sin(rot_rad)
        rc_y=p[1]*sp.sin(rot_rad)+p[2]*sp.cos(rot_rad)
    
        return p[0]*sp.exp(-(((rc_x-(x*sp.cos(rot_rad)-y*sp.sin(rot_rad)))/p[3])**2\
                                    +((rc_y-(x*sp.sin(rot_rad)+y*sp.cos(rot_rad)))/p[4])**2)/2)

    def f3(self, p, x, y):
        # f3 is a circular Gaussian, p in form (amplitude, mean_x, mean_y, sigma, offset).
        return p[0]*sp.exp(-(((p[1]-x)/p[3])**2+((p[2]-y)/p[3])**2)/2)+p[4]

    def f4(self, p, x, y):
        # f4 is a circular Gaussian as f3 but without an offset
        return p[0]*sp.exp(-(((p[1]-x)/p[3])**2+((p[2]-y)/p[3])**2)/2)
        
    def f5(self,p,x,y):
        # f5 is a circular Moffat profile
        return p[0]*((p[4] - 1.0)/np.pi/p[3]/p[3])*(1+(((x-p[1])**2+(y-p[2])**2)/p[3]/p[3]))**(-1*p[4])+p[5]

    def f6(self,p,x,y):
        # f6 is a circular Moffat profile but without an offset
        return p[0]*((p[4] - 1.0)/np.pi/p[3]/p[3])*(1+(((x-p[1])**2+(y-p[2])**2)/p[3]/p[3]))**(-1*p[4])

    def errfunc(self, p, x, y, z, weights):
        # If Moffat alpha of beta become unphysical return very large residual
        if (self.model == 'moffat') or (self.model == 'moffat_simple'):
            if (p[4] <= 0) or (p[3] <= 0):
                return np.ones(len(weights))*1e99
        return weights*(self.fitfunc(p, x, y) - z)

    def fit(self):

        self.p, self.cov_x, self.infodict, self.mesg, self.success = \
        leastsq(self.errfunc, self.p, \
                args=(self.x, self.y, self.z, self.weights), full_output=1)

        var_fit = (self.errfunc(self.p, self.x, \
            self.y, self.z, self.weights)**2).sum()/(len(self.z)-len(self.p))

        self.var_fit = var_fit

        if self.cov_x is not None:
            self.perr = sp.sqrt(self.cov_x.diagonal())*self.var_fit

        if not self.success in [1,2,3,4]:
            print("Fit Failed...")
            #raise ExpFittingException("Fit failed")

    def fwhm(self):
        if (self.model == 'moffat') or (self.model == 'moffat_simple'):
            psf = 2*self.p[3]*np.sqrt(2**(1/self.p[4])-1)
        elif (self.model == 'gaussian_circ') or (self.model == 'gaussian_circ_simple'):
            psf = self.p[3]*2*np.sqrt(2*np.log(2))
        else:
            print('Unknown model, no PSF measured')
            psf = 0.0
            
        return psf

    def __call__(self, x, y):
        return self.fitfunc(self.p, x, y)
    
def fibre_integrator(fitter, diameter, pixel=False):
    """Edits a fitter's fitfunc so that it integrates over each SAMI fibre."""

    # Save the diameter; not used here but could be useful later
    fitter.diameter = diameter

    # Define the subsampling points to use
    n_pix = 5       # Number of sampling points across the fibre
    # First make a 1d array of subsample points
    delta_x = np.linspace(-0.5 * (diameter * (1 - 1.0/n_pix)),
                          0.5 * (diameter * (1 - 1.0/n_pix)),
                          num=n_pix)
    delta_y = delta_x
    # Then turn that into a 2d grid of (delta_x, delta_y) centred on (0, 0)
    delta_x = np.ravel(np.outer(delta_x, np.ones(n_pix)))
    delta_y = np.ravel(np.outer(np.ones(n_pix), delta_y))
    if pixel:
        # Square pixels; keep everything
        n_keep = n_pix**2
    else:
        # Round fibres; only keep the points within one radius
        keep = np.where(delta_x**2 + delta_y**2 < (0.5 * diameter)**2)[0]
        n_keep = np.size(keep)
        delta_x = delta_x[keep]
        delta_y = delta_y[keep]

    old_fitfunc = fitter.fitfunc

    def integrated_fitfunc(p, x, y):
        # The fitter's fitfunc will be replaced by this one
        n_fib = np.size(x)
        x_sub = (np.outer(delta_x, np.ones(n_fib)) +
                 np.outer(np.ones(n_keep), x))
        y_sub = (np.outer(delta_y, np.ones(n_fib)) +
                 np.outer(np.ones(n_keep), y))
        return np.mean(old_fitfunc(p, x_sub, y_sub), 0)

    # Replace the fitter's fitfunc
    fitter.fitfunc = integrated_fitfunc

    return


def plot_ifu(x,y,f,p=None,probe=None):
    
    import matplotlib.colors as colors
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    
    fig = plt.figure()
    plt.clf()
    ax = plt.subplot(111)
    ax.set_aspect('equal')
    
    # Set up color scaling for fibre fluxes
    jet = plt.get_cmap('jet')
    cnorm = colors.Normalize(vmin=min(f),vmax=max(f))
    scalarmap = cm.ScalarMappable(norm=cnorm,cmap=jet)
    
    # Plot fibre locations and fluxes
    for xi,yi,fi in zip(x,y,f):
        plt.plot(xi,yi,'o',color=scalarmap.to_rgba(fi),ms=15)
 
    # Plot fitted centroid location       
    if p is not None:  
        plt.plot(p[1],p[2],'kx',ms=30)
        
    # Label each image with the probe name
    if probe is not None:
        plt.title(probe,fontsize=15)
        
    plt.draw()

    
def centroid_fit(x,y,flux,microns=False,premask=False,do_moffat=True):

    com = comxyz(x,y,flux) #**use good data within masking

    # Peak height guess could be closest fibre to com position.
    dist = (x-com[0])**2+(y-com[1])**2 # distance between com and all fibres.
 
    # First guess at width of Gaussian - diameter of a core in degrees/microns.
    if microns == True:
        sigx = 105.0
        core_diam = 105.0
    else:
        sigx = 4.44e-4*3600
        core_diam = 4.44e-4*3600

    # Fit circular 2D Gaussians.
    p0 = [np.mean(flux[np.where(dist==np.min(dist))]), com[0], com[1], sigx, 0.0]
    gf = BundleFitter(p0,x,y,flux,model='gaussian_circ')
    fibre_integrator(gf, core_diam)
    gf.fit()    
    
    # Refit using initial values from Gaussian fit
    if do_moffat:
        p0 = [gf.p[0], gf.p[1], gf.p[2], gf.p[3]*np.sqrt(2*np.log(2)), 1.0, gf.p[4]]
        gf = BundleFitter(p0,x,y,flux,model='moffat')
        fibre_integrator(gf, core_diam)
        gf.fit()     
 

    # Make a linear grid to reconstruct the fitted Gaussian over.
    x_0 = np.min(x) 
    y_0 = np.min(y)

    # dx should be 1/10th the fibre diameter (in whatever units)
    dx = sigx/10.0
    
    xlin = x_0+np.arange(100)*dx # x axis
    ylin = y_0+np.arange(100)*dx # y axis

    # Reconstruct the model
    model = np.zeros((len(xlin), len(ylin)))
    # Reconstructing the Gaussian over the proper grid.
    for ii in range(len(xlin)):
        xval=xlin[ii]
        for jj in range(len(ylin)):
            yval=ylin[jj]
            model[ii,jj]=gf.fitfunc(gf.p, xval, yval)
    
    return gf, flux, xlin, ylin, model

def hector_circle(x,xc,yc,radius):
    return yc + np.sqrt(radius**2 - (x - xc)**2)

def rotation_fit(file_list,plot_fit=False):

    """
    Fit for bundle rotation, returning rotation centroid and radius for all bundles. Take
    a list of >3 fitted centroid positions for a set of bundles, then determine a rotation
    centre by fitting a simple circle to the input data. Loop over all bundles in input file
    then print results to screen.
    
    Required inputs: file_list - list of strings with paths to centroid input files as
                        output by hector_utils.main. NB all files should have data for
                        the same set of probes
    """
    
    # Check if required input has been provided
    if len(file_list) < 3:
        print('Please provide a minimum of three input files, otherwise rotation cannot be constrained')
        return
        
    # Open the first input file to get list of probes
    tab = Table.read(file_list[0],format='ascii.commented_header')
    Probe_list = tab['Probe'].data
    
    def calc_R(x,y,xc,yc):
        return np.sqrt((x-xc)**2 + (y-yc)**2)
        
    def f(c,x,y):
        Ri = calc_R(x,y,*c)
        return Ri - Ri.mean()
    
    # Loop over probes in Probe_list
    for Probe in Probe_list:
    
        # Read in x,y coordinates of centroid for 1 probe from all input files
        xdat,ydat = [],[]
        for file in file_list:
            tab = Table.read(file,format='ascii.commented_header')
            index = np.where(tab['Probe'] == Probe)
            xdat.append(tab['X_mic'][index].data[0])
            ydat.append(tab['Y_mic'][index].data[0])
            
        # Fit a circle
        
        #p0 = [np.mean(xdat),np.mean(ydat),500.]
        #popt, pcov = curve_fit(hector_circle,xdat,ydat)#,p0=p0)
        
        p0 = np.mean(xdat),np.mean(ydat)
        center_2, ier = leastsq(f,p0,args=(xdat,ydat))
        xc_2,yc_2 = center_2
        Ri_2 = calc_R(xdat,ydat,*center_2)
        R_2 = Ri_2.mean()
        popt = [xc_2,yc_2,R_2]
        
        print(p0)
        print(popt)
        
        
        print('Probe: {}, Xrot: {}, Yrot: {}, Radrot: {}'.format(Probe,popt[0],popt[1],popt[2]))
        if plot_fit:
            fig = plt.figure()
            plt.clf()
            ax = plt.subplot(111)
            ax.set_aspect('equal')
            theta = np.arange(1000)/1000*2*np.pi
            x = popt[0] + popt[2]*np.cos(theta)
            y = popt[1] + popt[2]*np.sin(theta)
            ax.plot(x,y,'k-',lw=3)
            ax.plot(xdat,ydat,'rx',ms=5)


def main(flat_file,object_file,pix_waveband=100,pix_start=500,sigma_clip=True,
        quiet=False,do_moffat=True,to_file=None,display_plot=False):

    """ 
    Fit the centroids of all probes using either a 2D Gaussian or Moffat profile. First
    identifies tram line locations in a flat, then extracts those same locations in an
    object framme, collapses each bundle to an image, then fits a model to each bundle image.

    Required inputs:    flat_file - string containing full path to a SAMI/Hector flat
                        object_file - string containing full path to a SAMI/Hector object

    Settings:           pix_waveband - number of spectral pixels to use for tram line identification
                        pix_start - spectral pixel to start looking for tram lines
                        sigma_clip - remove outlier pixels when creating bundle images (highly
                        recommended to turn this on)
                        quiet - print additional commmand line output
                        do_moffat - if True fit a Moffat profile, if false fit a 2D Gaussian only
                        to_file - if not None, write centroids and FWHM to file path given here
                        display_plot - plot bundle images and fitted centroids
    """

    # Import flat field frame
    flat = fits.open(flat_file)
    flat_fibtab = flat['MORE.FIBRES_IFU'].data
    # This returns a numpy ndarray
    flat_data = flat['Primary'].data

    # Determine number of tram lines from fibre table
    expectedPeaks = count_enabled(flat_fibtab, quiet)
    if not quiet:
        print("---> Flat field data array size is:", flat_data.shape, ". There are", expectedPeaks, "enabled fibres.")
    NumSpectralPixels=flat_data.shape[1]
    
    # Identify tram line locations from flat field
    cut_loc, peaks = process_flat(flat_data, pix_waveband, pix_start, expectedPeaks, NumSpectralPixels,quiet)
    
    # Extract fibre spectra from object frame then collapse to determine fibre fluxes
    object_spec, obs_ftab, obs_gtab = process_object_file(object_file, sigma_clip, cut_loc, peaks, expectedPeaks, pix_waveband, quiet)
    
    # Determine probes from fibre table
    Probe_list = get_probe_list(flat_fibtab, quiet)
    if not quiet:
        print("----> Probe List:", Probe_list)
        
    centroids_deg = []
    centroids_micron = []
    fwhms = []
        
    # Loop over probes, fitting centroids
    for Probe in Probe_list:
        mask = obs_ftab['PROBENAME'] == Probe
        flux = object_spec[mask]
        idx0 = np.where(obs_ftab['FIBNUM'][mask] == 1)
        x_deg = obs_ftab['XPOS'][mask]#-obs_ftab['XPOS'][mask][idx0]
        y_deg = obs_ftab['YPOS'][mask]#-obs_ftab['YPOS'][mask][idx0]
        x_micron = obs_ftab['FIBPOS_X'][mask]# - obs_ftab['FIBPOS_X'][mask][idx0]
        y_micron = obs_ftab['FIBPOS_Y'][mask]# - obs_ftab['FIBPOS_Y'][mask][idx0]
        
        gf_mic, flux_mic, xlin_mic, ylin_mic, model_mic=centroid_fit(x_micron, y_micron, 
                                                        flux, microns=True,do_moffat=do_moffat)
        gf_deg, flux_deg, xlin_deg, ylin_deg, model_deg=centroid_fit(x_deg, y_deg, 
                                                        flux, microns=False,do_moffat=do_moffat)
        
        
        if do_moffat:
            amp_deg, xout_deg, yout_deg, alpha_deg, beta_deg, bias_sky=gf_deg.p
            amp_mic, xout_mic, yout_mic, alpha_mic, beta_mic, bias_mic=gf_mic.p
        else:
            amp_deg, xout_deg, yout_deg, sig_deg, bias_sky=p_deg
            amp_mic, xout_mic, yout_mic, sig_mic, bias_mic=p_mic
        
        centroids_deg.append((xout_deg,yout_deg))
        centroids_micron.append((xout_mic,yout_mic))
        fwhms.append(gf_deg.fwhm())
        
        print("Probe:",str(Probe).rstrip(), "Centroid is:", xout_mic, yout_mic, "FWHM (arcsec) is:", gf_deg.fwhm())
        if display_plot:
            plot_ifu(x_micron,y_micron,flux,gf_mic.p,Probe)
         
    # If required, write output to file - note file cannot already exist   
    if to_file is not None:
        if os.path.exists(to_file):
            print('Output file already exists, NOT overwriting.')
        else:
            with open(to_file,'w') as f:
                print('#','Probe','X_mic','Y_mic','X_deg','Y_deg','FWHM',file=f)
                for i,Probe in enumerate(Probe_list):
                    print(Probe,centroids_micron[i][0],centroids_micron[i][1],
                        centroids_deg[i][0],centroids_deg[i][1],fwhms[i],file=f)
               
    return Probe_list, centroids_deg, centroids_micron
        
# If run from the command line
if __name__ == '__main__':

    main()