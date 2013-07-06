#! /usr/bin/env python
'''
This file contains a set of utilities for reading Zemax glass (*.agf) files, analyzing glass
properties, and displaying glass data.

Note that the "library" is considered
'''

from numpy import *
import os, glob, sys
import matplotlib.pyplot as plt
import matplotlib.transforms
from matplotlib.transforms import offset_copy
import DataCursor
import pdb

class ZemaxGlassLibrary(object):
    '''
    ZemaxGlassLibrary is a class to hold all of the information contained in a Zemax-format library of glass catalogs.

    Glass catalogs are in the form of *.agf files, typically given with a vendor name as the filename. The class
    initializer, if given the directory where the catalogs are located, will read them all into a single dictionary
    data structure. The ZemaxLibrary class also gathers together associated methods for manipulating the data, such
    as methods to cull the number of glasses down to a restricted subset, the ability to plot some glass properties
    versus others, the ability to fit different paramatrized dispersion curves to the refractive index data, etc.

    Attributes
    ----------
    ...

    Methods
    -------
    ...
    '''

    def __init__(self, dir, wavemin=400.0, wavemax=700.0, nwaves=300, catalog=None, sampling_domain='wavelength',
                 degree=3, debug=False):
        self.debug = debug
        self.dir = dir
        self.wavemin = wavemin                  ## the minimum wavelength for performing fits to spectral property curves
        self.wavemax = wavemax                  ## the maximum wavelength for performing fits to spectral property curves
        self.nwaves = nwaves                    ## the number of wavelength/wavenumber samples to take for refractive index data
        self.degree = degree                    ## the degree of polynomial to use when fitting dispersion data
        #self.basis = basis                     ## the type of basis to use for polynomial fitting ('Taylor','Legendre')
        self.sampling_domain = sampling_domain  ## the domain ('wavelength' or 'wavenumber') in which to evenly sample the data
        self.library = read_library(dir, catalog=catalog)

        if (sampling_domain == 'wavelength'):
            self.waves = linspace(wavemin, wavemax, nwaves)      ## wavelength in nm
            self.wavenumbers = 1000.0 / self.waves               ## wavenumber in um^-1
        elif (sampling_domain == 'wavenumber'):
            sigma_min = 1000.0 / wavemax
            sigma_max = 1000.0 / wavemin
            self.wavenumbers = linspace(sigma_min, sigma_max, nwaves) ## wavenumber in um^-1
            self.waves = 1000.0 / self.wavenumbers                    ## wavelength in nm

        return

    ## =========================
    def __getattr__(self, name):
        '''
        Redirect the default __getattr__() function so that any attempt to generate a currently nonexisting attribute
        will trigger a method to generate that attribute from existing attributes.

        Parameters
        ----------
        name : str
            The name of the attribute being accessed.
        '''

        if (name == 'nglasses'):
            nglasses = 0
            for catalog in self.library:
                for glass in self.library[catalog]:
                    nglasses += 1
            return(nglasses)
        elif (name == 'catalogs'):
            catalogs = self.library.keys()
            return(catalogs)
        elif (name == 'glasses'):
            glasses = []
            for catalog in self.library:
                glasses.extend(self.library[catalog].keys())
            return(glasses)

        return

    ## =========================
    def pprint(self, catalog=None, glass=None):
        '''
        Pretty-print the glass library, or a chosen catalog in it.

        Parameters
        ----------
        catalog : str
            The name of the catalog within the library to print.
        glass : str
            The name of the glass within the library to print.
        '''

        for glasscat in self.library:
            if (catalog != None) and (glasscat != catalog.lower()): continue
            print(glasscat + ':')
            for glassname in self.library[glasscat]:
                if (glass != None) and (glassname != glass.upper()): continue
                glassdict = self.library[glasscat][glassname]
                print('  ' + glassname + ':')
                print('    nd       = ' + str(glassdict['nd']))
                print('    vd       = ' + str(glassdict['vd']))
                print('    dispform = ' + str(glassdict['dispform']))
                print('    tce      = ' + str(glassdict['tce']))
                print('    density  = ' + str(glassdict['density']))
                print('    dpgf     = ' + str(glassdict['dpgf']))
                print('    cd       = ' + str(glassdict['cd']))
                print('    td       = ' + str(glassdict['td']))
                print('    od       = ' + str(glassdict['od']))
                print('    ld       = ' + str(glassdict['ld']))
                if ('interp_coeffs' in glassdict):
                    print('    coeffs   = ' + repr(glassdict['interp_coeffs']))
        return

    ## =============================================================================
    def simplify_schott_catalog(self, zealous=False):
        '''
        Remove redundant, little-used, and unusual glasses from the Schott glass catalog.

        Parameters
        ----------
        zealous : bool, optional
            Whether to remove the "high transmission" and close-to-redundant glasses.
        '''

        schott_glasses = []
        for glass in self.library['schott']:
            schott_glasses.append(glass)

        ## Remove the "inquiry glasses".
        I_glasses = ['FK3', 'N-SK10', 'N-SK15', 'BAFN6', 'N-BAF3', 'N-LAF3', 'SFL57', 'SFL6', 'SF11', 'N-SF19', 'N-PSK53', 'N-SF64', 'N-SF56', 'LASF35']
        num_i = alen(I_glasses)

        ## Remove the "high-transmission" duplications of regular glasses.
        H_glasses = ['LF5HT', 'BK7HT', 'LLF1HT', 'N-SF57HT', 'SF57HT', 'LF6HT', 'N-SF6HT', 'F14HT', 'LLF6HT', 'SF57HHT', 'F2HT', 'K5HT', 'SF6HT', 'F8HT', 'K7HT']
        num_h = alen(H_glasses)

        ## Remove the "soon-to-be-inquiry" glasses from the Schott catalog.
        N_glasses = ['KZFSN5', 'P-PK53', 'N-LAF36', 'UBK7', 'N-BK7']
        num_n = alen(N_glasses)

        ## Remove the Zinc-sulfide and zinc selenide glasses.
        ZN_glasses = ['CLEARTRAN_OLD', 'ZNS_VIS']
        num_zn = alen(ZN_glasses)

        ## "zealous": remove the "P" glasses specifically designed for hot press molding, and several glasses that are nearly identical to others in the catalog.
        Z_glasses = ['N-F2', 'N-LAF7', 'N-SF1', 'N-SF10', 'N-SF2', 'N-SF4', 'N-SF5', 'N-SF57', 'N-SF6', 'N-ZK7', 'P-LASF50', 'P-LASF51', 'P-SF8', 'P-SK58A', 'P-SK60']
        num_z = alen(Z_glasses)

        for glass in schott_glasses:
            remove = (glass in I_glasses) or (glass in H_glasses) or (glass in N_glasses) or \
                     (glass in ZN_glasses)
            if zealous:
                remove = remove or (glass in Z_glasses)
            if remove:
                del self.library['schott'][glass]

        ## Refresh any existing information in the library.
        if hasattr(self, 'nglasses'):
            nglasses = 0
            for catalog in self.library:
                for glass in self.library[catalog]:
                    nglasses += 1
            self.nglasses = nglasses
        elif (name == 'glasses'):
            glasses = []
            for catalog in self.library:
                glasses.extend(self.library[catalog].keys())
            self.glasses = glasses

        return

    ## =========================
    def get_dispersion(self, glass):
        '''
        For a given glass, calculate the dispersion curve (refractive index as a function of wavelength in nm).

        If sampling_domain=='wavenumber' then the curve is still returned in wavelength units, but the sampling
        will be uniform in wavenumber and not uniform in wavelength.

        Parameters
        ----------
        glass : str
            The name of the glass we want to know about.

        Returns
        -------
        indices : ndarray
            A numpy array giving the sampled refractive index curve.
        '''

        catalog = find_catalog_for_glassname(self.library, glass)
        if ('indices' in self.library[catalog][glass]):
            return(self.waves, self.library[catalog][glass]['indices'])

        if (catalog == None):
            print('Warning: cannot find glass "' + glass + '" in the library! Aborting ...')
            return(None, None)
        if ('waves' in self.library[catalog][glass]) and ('indices' in self.library[catalog][glass]):
            return(self.waves, self.library[catalog][glass]['indices'])

        cd = self.library[catalog][glass]['cd']
        dispform = self.library[catalog][glass]['dispform']
        ld = self.library[catalog][glass]['ld']

        ## Zemax's dispersion formulas all use wavelengths in um. So, to compare "ld"
        ## and wavemin,wavemax we first convert the former to nm and then, when done
        ## we convert to um.
        if (amax(self.waves) < ld[0]*1000.0) or (amin(self.waves) > ld[1]*1000.0):
            print('wavemin,wavemax=(%f,%f), but ld=(%f,%f)' % (amin(self.waves), amax(self.waves), ld[0], ld[1]))
            print('Cannot calculate an index in the required spectral range. Aborting ...')
            return(None,None)

        ## Choose which domain is the one in which we sample uniformly. Regardless
        ## of choice, the returned vector "w" gives wavelength in um.
        if (self.sampling_domain == 'wavelength'):
            w = self.waves / 1000.0     ## convert from nm to um
        elif (self.sampling_domain == 'wavenumber'):
            w = self.wavenumbers

        if (dispform == 1):
            formula_rhs = cd[0] + (cd[1] * w**2) + (cd[2] * w**-2) + (cd[3] * w**-4) + (cd[4] * w**-6) + (cd[5] * w**-8)
            indices = sqrt(formula_rhs)
        elif (dispform == 2): ## Sellmeier1
            formula_rhs = (cd[0] * w**2 / (w**2 - cd[1])) + (cd[2] * w**2 / (w**2 - cd[3])) + (cd[4] * w**2 / (w**2 - cd[5]))
            indices = sqrt(formula_rhs + 1.0)
        elif (dispform == 3): ## Herzberger
            L = 1.0 / (w**2 - 0.028)
            indices = cd[0] + (cd[1] * L) + (cd[2] * L**2) + (cd[3] * w**2) + (cd[4] * w**4) + (cd[5] * w**6)
        elif (dispform == 4): ## Sellmeier2
            formula_rhs = cd[0] + (cd[1] * w**2 / (w**2 - (cd[2])**2)) + (cd[3] * w**2 / (w**2 - (cd[4])**2))
            indices = sqrt(formula_rhs + 1.0)
        elif (dispform == 5): ## Conrady
            indices = cd[0] + (cd[1] / w) + (cd[2] / w**3.5)
        elif (dispform == 6): ## Sellmeier3
            formula_rhs = (cd[0] * w**2 / (w**2 - cd[1])) + (cd[2] * w**2 / (w**2 - cd[3])) + \
                          (cd[4] * w**2 / (w**2 - cd[5])) + (cd[6] * w**2 / (w**2 - cd[7]))
            indices = sqrt(formula_rhs + 1.0)
        elif (dispform == 7): ## HandbookOfOptics1
            formula_rhs = cd[0] + (cd[1] / (w**2 - cd[2])) - (cd[3] * w**2)
            indices = sqrt(formula_rhs)
        elif (dispform == 8): ## HandbookOfOptics2
            formula_rhs = cd[0] + (cd[1] * w**2 / (w**2 - cd[2])) - (cd[3] * w**2)
            indices = sqrt(formula_rhs)
        elif (dispform == 9): ## Sellmeier4
            formula_rhs = cd[0] + (cd[1] * w**2 / (w**2 - cd[2])) + (cd[3] * w**2 / (w**2 - cd[4]))
            indices = sqrt(formula_rhs)
        elif (dispform == 10): ## Extended1
            formula_rhs = cd[0] + (cd[1] * w**2) + (cd[2] * w**-2) + (cd[3] * w**-4) + (cd[4] * w**-6) + \
                          (cd[5] * w**-8) + (cd[6] * w**-10) + (cd[7] * w**-12)
            indices = sqrt(formula_rhs)
        elif (dispform == 11): ## Sellmeier5
            formula_rhs = (cd[0] * w**2 / (w**2 - cd[1])) + (cd[2] * w**2 / (w**2 - cd[3])) + \
                          (cd[4] * w**2 / (w**2 - cd[5])) + (cd[6] * w**2 / (w**2 - cd[7])) + \
                          (cd[8] * w**2 / (w**2 - cd[9]))
            indices = sqrt(formula_rhs + 1.0)
        elif (dispform == 12): ## Extended2
            formula_rhs = cd[0] + (cd[1] * w**2) + (cd[2] * w**-2) + (cd[3] * w**-4) + (cd[4] * w**-6) + \
                          (cd[5] * w**-8) + (cd[6] * w**4) + (cd[7] * w**6)
            indices = sqrt(formula_rhs)

        ## Zemax's dispersion formulas all use wavelengths in um. So, to compare "ld"
        ## and wavemin,wavemax we first convert the former to nm and then, when done
        ## we convert to um.
        if (amin(self.waves) < ld[0]*1000.0):
            print('Truncating fitting range since wavemin=%f, but ld[0]=%f ...' % (self.wavemin, ld[0]))
            indices[self.waves < ld[0]*1000.0] = 0.0
        if (amax(self.waves) > ld[1]*1000.0):
            print('Truncating fitting range since wavemax=%f, but ld[1]=%f ...' % (self.wavemax, ld[1]))
            indices[self.waves > ld[1]*1000.0] = 0.0

        ## Convert waves in um back to waves in nm for output.
        self.library[catalog][glass]['indices'] = indices
        return(self.waves, indices)

    ## =========================
    def get_interp_dispersion(self, glass):
        catalog = find_catalog_for_glassname(self.library, glass)
        if ('interp_indices' in self.library[catalog][glass]):
            return(self.waves, self.library[catalog][glass]['interp_indices'])

        ## Generate a vector of wavelengths in nm, with samples every 1 nm.
        (waves, indices) = self.get_dispersion(glass)

        okay = (indices > 0.0)
        coeffs = polyfit(waves[okay], indices[okay], self.degree)
        coeffs = coeffs[::-1]       ## reverse the vector so that the zeroth degree coeff goes first
        self.library[catalog][glass]['interp_coeffs'] = coeffs
        #yyy
        print('coeffs=', coeffs)

        interp_indices = polyeval_Horner(waves, coeffs)
        self.library[catalog][glass]['interp_indices'] = interp_indices

        return(waves, interp_indices)

    ## =============================================================================
    def get_glassdata(self, glass):
        '''
        Get the dictionary describing a glass' optical properties.

        Parameters
        ----------
        glass : str
            The name of the glass (in upper case).

        Returns
        -------
        glassdict : dict
            The glass property data. (Returns `None` if the glass is not found.
        '''

        catalog = find_catalog_for_glassname(self.library, glass)
        if (catalog != None):
            glassdict = self.library[catalog][glass]
            return(glassdict)
        return(None)

#    ## =============================================================================
#    def insert_dispersion_coeffs(self, catalog=None):
#        '''
#        Insert the dispersion coefficients into every glass in the library.
#
#        Parameters
#        ----------
#        catalog : str, optional
#            The catalog to use if you want to insert dispersion coefficients in only one catalog and not the entire library.
#        '''
#
#        for catalog in self.library:
#            for glass in self.library[catalog]:
#                self.get_interp_dispersion(glass)
#
#        return

    ## =============================================================================
    def cull_library(self, key1, tol1, key2=None, tol2=None):
        '''
        Reduce all catalogs in the library such that no two glasses are simultaneously
        within (+/- tol1) of key1 and (+/- tol2) of key2.
        '''

        keydict1 = {}
        keydict2 = {}
        names = []
        keyval1 = []
        keyval2 = []

        for catalog in self.library:
            for glass in self.library[catalog]:
                names.append(glass)

                if (key1 in self.library[catalog][glass]):
                    keyval1.append(self.library[catalog][glass][key1])
                else:
                    keyval1.append(self.library[catalog][glass][None])

                if (key2 != None):
                    if (key2 in self.library[catalog][glass]):
                        keyval2.append(self.library[catalog][glass][key2])
                    else:
                        keyval2.append(self.library[catalog][glass][None])

        glasses_to_remove = []
        keyval1 = array(keyval1)
        keyval2 = array(keyval2)

        for i,glass in enumerate(names):
            if (key2 == None):
                idx = where(abs(keyval1[i] - keyval1) < tol1)
                glasses_to_remove.append([name for name in names[idx] if name != names[i]])
            else:
                idx = where((abs(keyval1[i] - keyval1) < tol1) and (abs(keyval2 - keyval2[i]) < tol2))
                #print('%3i %3i %5.3f %5.3f %6.3f %6.3f %12s %12s --> REMOVE %3i %12s' % (i, j, keyval1[i], keyval1[j], keyval2[i], keyval2[j], names_all[i], names_all[j], j, names_all[j]))
                glasses_to_remove.append([name for name in names[idx] if name != names[i]])

        ## Remove the duplicates from the "remove" list, and then delete those glasses
        ## from the glass catalog.
        glasses_to_remove = unique(glasses_to_remove)
        for glass in glasses_to_remove:
            catalog = find_catalog_for_glassname(glass)
            #print('i='+str(i)+': catalog='+catalog+'; glass='+name)
            del self.library[catalog][glass]

        return

    ## =========================
    def plot_dispersion(self, glass):
        (x, y) = self.get_dispersion(glass)
        plt.plot(x, y)
        plt.title(glass + ' dispersion')
        plt.xlabel('wavelength (nm)')
        plt.ylabel('refractive index')

        ## Enforce the plotting range.
        xmin = min(x)
        xmax = max(x)
        xrange = xmax - xmin
        if (xrange < 1.0): xrange = 1.0
        xbot = xmin - (0.05 * xrange)
        xtop = xmax + (0.05 * xrange)

        ymin = min(y)
        ymax = max(y)
        yrange = ymax - ymin
        if (yrange < 1.0E-9): yrange = 1.0
        ybot = ymin - (0.05 * yrange)
        ytop = ymax + (0.05 * yrange)

        plt.axis([xbot,xtop,ybot,ytop])

        return

    ## =========================
    def plot_catalog_property_diagram(self, catalog, prop1='nd', prop2='vd', show_labels=True):
        '''
        Plot a scatter diagram of one glass property against another.

        A "property" can be: nd, vd, cr, fr, ar, sr, pr, n0, n1, n2, n3, tce, density, dpgf. Note that
        if "prop1" and "prop2" are left unspecified, then the result is an Abbe diagram.

        If catalog=='all', then all glasses from the entire library are plotted.

        Parameters
        ----------
        catalog : str
            Which catalog to plot.
        prop1 : str
            The glass data property to show along the abscissa (x-axis).
        prop2 : str
            The glass data property to show along the ordinate (y-axis).
        show_labels : bool
            Whether to show the glass name labels near the data points.
        '''

        if (catalog == 'all'):
            catalogs = self.library.keys()
        else:
            catalogs = [catalog]

        glassnames = []
        p1 = []
        p2 = []

        for cat in catalogs:
            for glass in self.library[cat]:
                if (catalog == 'all') and (glass == 'AIR'): continue
                if (catalog == 'all') and (abs(self.library[cat][glass]['vd']) < 1.0E-6): continue
                glassnames.append(glass)

                if (prop1 in ('n0','n1','n2','n3')):
                    idx = int(prop1[1])
                    if ('interp_coeffs' not in self.library[cat][glass]):
                        print('Calculating dispersion coefficients for "' + glass + '" ...')
                        self.get_interp_dispersion(glass)
                    p1.append(self.library[cat][glass]['interp_coeffs'][idx])
                else:
                    p1.append(self.library[cat][glass][prop1])

                if (prop2 in ('n0','n1','n2','n3')):
                    idx = int(prop2[1])
                    if ('interp_coeffs' not in self.library[cat][glass]):
                        print('Calculating dispersion coefficients for "' + glass + '" ...')
                        self.get_interp_dispersion(glass)
                    p2.append(self.library[cat][glass]['interp_coeffs'][idx])
                else:
                    p2.append(self.library[cat][glass][prop2])

        fig = plt.figure()
        ax = plt.subplot(1,1,1)
        ax.plot(p1, p2, 'bo', markersize=5)
        #DataCursor.DataCursor([ax])     ## turn on the feature of getting a box pointing to data points
        plt.title('catalog "' + catalog + '": ' + prop1 + ' vs. ' + prop2)
        plt.xlabel(prop1)
        plt.ylabel(prop2)

        ## Enforce the plotting range.
        xmin = min(p1)
        xmax = max(p1)
        xrange = xmax - xmin
        if (xrange < 1.0): xrange = 1.0
        xbot = xmin - (0.05 * xrange)
        xtop = xmax + (0.05 * xrange)
        xdist = 0.01 * xrange               ## for plotting text near the data points

        ymin = min(p2)
        ymax = max(p2)
        yrange = ymax - ymin
        if (yrange < 1.0E-9): yrange = 1.0
        ybot = ymin - (0.05 * yrange)
        ytop = ymax + (0.05 * yrange)
        ydist = 0.01 * yrange               ## for plotting text near the data points

        plt.axis([xbot,xtop,ybot,ytop])

        if show_labels:
            ## Plot all of the glass labels offset by (5,5) pixels in (x,y) from the data point.
            trans_offset = offset_copy(ax.transData, fig=fig, x=5, y=5, units='dots')
            for i in arange(alen(glassnames)):
                #print('i=%i: glassname=%s, p1=%f, p2=%f' % (i, glassnames[i], p1[i], p2[i]))
                plt.text(p1[i], p2[i], glassnames[i], fontsize=7, zorder=0, transform=trans_offset)

        return


## =============================================================================
## End of ZemaxLibrary class
## =============================================================================

def read_library(glassdir, catalog=None):
    '''
    Get a list of all '*.agf' files in the directory, then call `parse_glassfile()` on each one.

    Parameters
    ----------
    glassdir : str
        The directory where we can find all of the *.agf files.
    catalog : str, optional
        If there is only one catalog of interest within the directory, then read only this one.

    Returns
    -------
    glass_library : dict
        A dictionary in which each entry is a glass catalog.

    Example
    -------
    >>> glasscat = read_zemax.read_glasscat('~/Zemax/Glasscat/')
    >>> nd = glasscat['schott']['N-BK7']['nd']
    '''

    glassdir = os.path.normpath(glassdir)
    files = glob.glob(os.path.join(glassdir,'*.[Aa][Gg][Ff]'))

    ## Get the set of catalog names. These keys will initialize the glasscat dictionary.
    glass_library = {}

    for f in files:
        catalog_name = os.path.basename(f)[:-4]
        if (catalog != None) and (catalog.lower() != catalog_name): continue
        glass_library[catalog_name] = parse_glass_file(f)

    return(glass_library)


## =============================================================================
def parse_glass_file(filename):
    '''
    Read a Zemax glass file (*.agf') and return its contents as a Python dictionary.

    Parameters
    ----------
    filename : str
        The file to parse.

    Returns
    -------
    glass_catalog : dict
        The dictionary containing glass data for all classes in the file.
    '''

    f = open(filename, 'r')
    glass_catalog = {}

    for line in f:
        if not line.strip(): continue
        if line.startswith('CC '): continue
        if line.startswith('NM '):
            nm = line.split()
            glassname = nm[1]
            glass_catalog[glassname] = {}
            glass_catalog[glassname]['dispform'] = int(nm[2])
            glass_catalog[glassname]['nd'] = float(nm[4])
            glass_catalog[glassname]['vd'] = float(nm[5])
            glass_catalog[glassname]['status'] = int(nm[7]) if (len(nm) >= 8) else 0
            glass_catalog[glassname]['meltfreq'] = int(nm[8]) if ((len(nm) >= 9) and (nm.count('-') < 0)) else 0
        elif line.startswith('ED '):
            ed = line.split()
            glass_catalog[glassname]['tce'] = float(ed[1])
            glass_catalog[glassname]['density'] = float(ed[3])
            glass_catalog[glassname]['dpgf'] = float(ed[4])
        elif line.startswith('CD '):
            cd = line.split()[1:]
            glass_catalog[glassname]['cd'] = [float(a) for a in cd]
        elif line.startswith('TD '):
            td = line.split()[1:]
            glass_catalog[glassname]['td'] = [float(a) for a in td]
        elif line.startswith('OD '):
            od = line.split()[1:]
            if (od.count('-') > 0): od[od.index('-')] = '-1'
            glass_catalog[glassname]['od'] = [float(a) for a in od]
        elif line.startswith('LD '):
            ld = line.split()[1:]
            glass_catalog[glassname]['ld'] = [float(a) for a in ld]
        elif line.startswith('IT '):
            it = line.split()[1:]
            it_row = [float(a) for a in it]
            if ('it' not in glass_catalog[glassname]):
                glass_catalog[glassname]['IT'] = {}
            glass_catalog[glassname]['IT']['wavelength'] = it_row[0]
            glass_catalog[glassname]['IT']['transmission'] = it_row[1]
            glass_catalog[glassname]['IT']['thickness'] = it_row[2]

    f.close()

    return(glass_catalog)

## =============================================================================
def find_catalog_for_glassname(glass_library, glassname):
    for catalog in glass_library:
        if glassname in glass_library[catalog]:
            return(catalog)
    return(None)

## =================================================================================================
def polyeval_Horner(x, poly_coeffs):
    '''
    Use Horner's rule for polynomial evaluation.

    Assume a polynomial of the form \
        p = c[0] + (c[1] * x) + (c[2] * x**2) + (c[3] * x**3) + ... + (c[N] * x**N).

    Parameters
    ----------
    x : array_like
        The abscissa at which to evaluate the polynomial.
    poly_coeffs : array_like
        The vector of polynomial coefficients.

    Returns
    -------
    p : ndarray
        The polynomial evaluated at the points given in x.
    '''

    ncoeffs = alen(poly_coeffs)
    p = zeros(alen(x))
    for n in arange(ncoeffs-1,-1,-1):
        p = poly_coeffs[n] + (x * p)
        #print('n=%i, c=%f' % (n, coeffs[n]))
    return(p)


## =============================================================================================
if (__name__ == '__main__'):
    glasslib = ZemaxGlassLibrary('/home/nh/Zemax/Glasscat/') #, catalog='schott')
    print(glasslib.nglasses)
    print(glasslib.catalogs)
    #print(glasslib.glasses)
    #glasslib.pprint('schott')
    #glasslib.pprint()

    glasslib.plot_dispersion('N-BK7')
    #plt.figure()
    #glasslib.plot_catalog_property_diagram('all', prop1='vd', prop2='nd')
    #glasslib.plot_catalog_property_diagram('schott', prop1='n0', prop2='n1')
    glasslib.plot_catalog_property_diagram('all', prop1='nd', prop2='dispform')
    plt.show()
