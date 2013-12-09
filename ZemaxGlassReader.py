#! /usr/bin/env python
'''
This file contains a set of utilities for reading Zemax glass (*.agf) files, analyzing glass
properties, and displaying glass data.

See LICENSE.txt for a description of the MIT/X license for this file.
'''

from numpy import *
import os, glob, sys
import matplotlib.pyplot as plt
import matplotlib.transforms
from matplotlib.transforms import offset_copy
import DataCursor
import colorsys
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
    dir : str
        The directory where the glass catalog files are stored.
    catalog : float

    Methods
    -------
    pprint
    simplify_schott_catalog
    get_dispersion
    get_polyfit_dispersion
    cull_library

    '''

    def __init__(self, dir=None, wavemin=400.0, wavemax=700.0, nwaves=300, catalog='all', sampling_domain='wavelength',
                 degree=3, debug=False):
        '''
        Initialize the glass library object.

        Parameters
        ----------
        wavemin : float, optional
            The shortest wavelength in the spectral region of interest.
        wavemax : float, optional
            The longest wavelength in the spectral region of interest.
        nwaves : float, optional
            The number of wavelength samples to use.
        catalog : str
            The catalog or list of catalogs to look for in "dir".
        sampling_domain : str, {'wavelength','wavenumber'}
            Whether to sample the spectrum evenly in wavelength or wavenumber.
        degree : int, optional
            The polynomial degree to use for fitting the dispersion spectrum.
        '''

        self.debug = debug
        self.degree = degree                    ## the degree of polynomial to use when fitting dispersion data
        #self.basis = basis                     ## the type of basis to use for polynomial fitting ('Taylor','Legendre')
        self.sampling_domain = sampling_domain  ## the domain ('wavelength' or 'wavenumber') in which to evenly sample the data

        if (dir == None):
            dir = os.path.dirname(__file__) + '/AGF_files/'

        self.dir = dir
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

        if (catalog == None):
            catalogs = self.library.keys()
        elif (len(catalog) > 1) and isinstance(catalog, list):
            catalogs = catalog
        else:
            catalogs = [catalog]

        for catalog in self.library:
            if (catalog not in catalogs): continue
            print(catalog + ':')
            for glassname in self.library[catalog]:
                if (glass != None) and (glassname != glass.upper()): continue
                glassdict = self.library[catalog][glassname]
                print('  ' + glassname + ':')
                print('    nd       = ' + str(glassdict['nd']))
                print('    vd       = ' + str(glassdict['vd']))
                print('    dispform = ' + str(glassdict['dispform']))
                if ('tce' in glassdict):
                    print('    tce      = ' + str(glassdict['tce']))
                if ('density' in glassdict):
                    print('    density  = ' + str(glassdict['density']))
                if ('dpgf' in glassdict):
                    print('    dpgf     = ' + str(glassdict['dpgf']))
                if ('cd' in glassdict):
                    print('    cd       = ' + str(glassdict['cd']))
                if ('td' in glassdict):
                    print('    td       = ' + str(glassdict['td']))
                if ('od' in glassdict):
                    print('    od       = ' + str(glassdict['od']))
                if ('ld' in glassdict):
                    print('    ld       = ' + str(glassdict['ld']))
                if ('interp_coeffs' in glassdict):
                    print('    coeffs   = ' + repr(glassdict['interp_coeffs']))

        print('')
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
    def get_dispersion(self, glass, catalog):
        '''
        For a given glass, calculate the dispersion curve (refractive index as a function of wavelength in nm).

        If sampling_domain=='wavenumber' then the curve is still returned in wavelength units, but the sampling
        will be uniform in wavenumber and not uniform in wavelength. Note that we need to know both the
        catalog and the glass name, and not just the glass name, because some catalogs share the same glass names.

        Parameters
        ----------
        glass : str
            The name of the glass we want to know about.
        catalog : str
            The catalog containing the glass.

        Returns
        -------
        indices : ndarray
            A numpy array giving the sampled refractive index curve.
        '''

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
        else:
            print('Dispersion formula #' + str(dispform) + ' (for glass=' + glass + ' in catalog=' + catalog + ') is not a valid choice.')
            indices = ones_like(w) * nan
            #raise ValueError('Dispersion formula #' + str(dispform) + ' (for glass=' + glass + ' in catalog=' + catalog + ') is not a valid choice.')

        ## Zemax's dispersion formulas all use wavelengths in um. So, to compare "ld"
        ## and wavemin,wavemax we first convert the former to nm and then, when done
        ## we convert to um.
        if (amin(self.waves) < ld[0]*1000.0):
            print('Truncating fitting range since wavemin=%f, but ld[0]=%f ...' % (amin(self.waves), ld[0]))
            indices[self.waves < ld[0]*1000.0] = 0.0
        if (amax(self.waves) > ld[1]*1000.0):
            print('Truncating fitting range since wavemax=%f, but ld[1]=%f ...' % (amax(self.waves), ld[1]))
            indices[self.waves > ld[1]*1000.0] = 0.0

        ## Convert waves in um back to waves in nm for output.
        self.library[catalog][glass]['indices'] = indices
        return(self.waves, indices)

    ## =========================
    def get_polyfit_dispersion(self, glass, catalog):
        '''
        Get the polynomial-fitted dispersion curve for a glass.

        Note that we need to know both the catalog and the glass name, and not just the glass name,
        because some catalogs share the same glass names.

        Parameters
        ----------
        glass : str
            Which glass to analyze.
        catalog : str
            The catalog containing the glass.
        '''

        if ('interp_indices' in self.library[catalog][glass]):
            return(self.waves, self.library[catalog][glass]['interp_indices'])

        ## Generate a vector of wavelengths in nm, with samples every 1 nm.
        (waves, indices) = self.get_dispersion(glass, catalog)

        okay = (indices > 0.0)
        if not any(okay):
            return(waves, ones_like(waves)*nan)

        x = linspace(-1.0, 1.0, alen(waves[okay]))
        coeffs = polyfit(x, indices[okay], self.degree)
        coeffs = coeffs[::-1]       ## reverse the vector so that the zeroth degree coeff goes first
        self.library[catalog][glass]['interp_coeffs'] = coeffs

        interp_indices = polyeval_Horner(x, coeffs)
        self.library[catalog][glass]['interp_indices'] = interp_indices

        return(waves, interp_indices)

    ## =============================================================================
    def cull_library(self, key1, tol1, key2=None, tol2=None):
        '''
        Reduce all catalogs in the library such that no two glasses are simultaneously
        within (+/- tol1) of key1 and (+/- tol2) of key2.

        Parameters
        ----------
        key1 : str
            The first parameter to analyze. This can be, e.g., "nd" or "dispform". Any key in the \
            glass data dictionary.
        tol1 : float
            The `tolerance` value: if the `key1` properties of any two glasses are within +/-tol1 \
            of one another, then remove all but one from the library.
        key2 : str
            The second parameter to analyze.
        tol2 : float
            The second `tolerance` value: if the `key1` and `key2` properties of any two glasses \
            are within +/-tol1 and +/-tol2 of one another simultaneously, then remove all but one \
            such glass from the library.
        '''

        keydict1 = {}
        keydict2 = {}
        names = []
        keyval1 = []
        keyval2 = []

        for catalog in self.library:
            for glass in self.library[catalog]:
                names.append(catalog+'_'+glass)
                catalogs.append(catalog)

                if (key1 in self.library[catalog][glass]):
                    keyval1.append(self.library[catalog][glass][key1])
                else:
                    keyval1.append(self.library[catalog][glass][None])

                if (key2 != None):
                    if (key2 in self.library[catalog][glass]):
                        keyval2.append(self.library[catalog][glass][key2])
                    else:
                        keyval2.append(self.library[catalog][glass][None])

        names_to_remove = []
        keyval1 = array(keyval1)
        keyval2 = array(keyval2)

        for i in arange(alen(names)):
            if (key2 == None):
                idx = where(abs(keyval1[i] - keyval1) < tol1)
                names_to_remove.append([name for name in names[idx] if name != names[i]])
            else:
                idx = where((abs(keyval1[i] - keyval1) < tol1) and (abs(keyval2 - keyval2[i]) < tol2))
                #print('%3i %3i %5.3f %5.3f %6.3f %6.3f %12s %12s --> REMOVE %3i %12s' % (i, j, keyval1[i], keyval1[j], keyval2[i], keyval2[j], names_all[i], names_all[j], j, names_all[j]))
                names_to_remove.append([name for name in names[idx] if name != names[i]])

        ## Remove the duplicates from the "remove" list, and then delete those glasses
        ## from the glass catalog.
        names_to_remove = unique(names_to_remove)
        for glass in names_to_remove:
            (catalog,glass) = glass.split('_')
            #print('i='+str(i)+': catalog='+catalog+'; glass='+name)
            del self.library[catalog][glass]

        return

    ## =========================
    def plot_dispersion(self, glass, catalog, polyfit=False, fiterror=False):
        '''
        Plot the glass refractive index curve as a function of wavelength.

        Parameters
        ----------
        glass : str
            The name of the glass to analyze.
        catalog : str
            The catalog containing the glass.
        polyfit : bool
            Whether to also display the polynomial fit to the curve.
        fiterror : bool
            If `polyfit` is True, then `fiterror` indicates whether a fitting error should also be \
            displayed, using the LHS y-axis.
        '''

        (x, y) = self.get_dispersion(glass, catalog)
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(111)
        ax.plot(x, y, 'b-', linewidth=2)

        if polyfit:
            (x2, y2) = self.get_polyfit_dispersion(glass, catalog)
            ax.plot(x2, y2, 'ko', markersize=4, zorder=0)

        plt.title(glass + ' dispersion')
        plt.xlabel('wavelength (nm)')
        plt.ylabel('refractive index')

        if polyfit and fiterror:
            fig.subplots_adjust(right=0.85)
            F = plt.gcf()
            (xsize,ysize) = F.get_size_inches()
            fig.set_size_inches(xsize+5.0,ysize)
            err = y2 - y
            ax2 = ax.twinx()
            ax2.set_ylabel('fit error')
            ax2.plot(x2, err, 'r-')

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

        ax.axis([xbot,xtop,ybot,ytop])

        return

    ## =========================
    def plot_catalog_property_diagram(self, catalog='all', prop1='nd', prop2='vd', show_labels=True):
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
        elif isinstance(catalog, list) and (len(catalog) > 1):
            catalogs = catalog
        elif isinstance(catalog, str):
            catalogs = [catalog]

        colors = get_colors(len(catalogs))
        glassnames = []
        all_p1 = []
        all_p2 = []

        fig = plt.figure(figsize=(12,6))
        ax = plt.gca()
        ax.set_color_cycle(colors)

        ## Collect lists of the property values for "prop1" and "prop2", one catalog at a time.
        ## Plot each catalog separately, so that each can be displayed with unique colors.
        for i,cat in enumerate(catalogs):
            p1 = []
            p2 = []
            for glass in self.library[cat]:
                if (catalog == 'all') and (glass == 'AIR'): continue
                if (catalog == 'all') and (abs(self.library[cat][glass]['vd']) < 1.0E-6): continue

                if (prop1 in ('n0','n1','n2','n3','n4','n5','n6','n6','n8','n9')):
                    idx = int(prop1[1])
                    if ('interp_coeffs' not in self.library[cat][glass]):
                        print('Calculating dispersion coefficients for "' + glass + '" ...')
                        self.get_polyfit_dispersion(glass, cat)
                    if ('interp_coeffs' in self.library[cat][glass]):
                        p1_coeffs = self.library[cat][glass]['interp_coeffs'][idx]
                    else:
                        print('Could not find valid interpolation coefficients for "' + glass + '" glass ...')
                        continue
                else:
                    p1.append(self.library[cat][glass][prop1])

                if (prop2 in ('n0','n1','n2','n3','n4','n5','n6','n6','n8','n9')):
                    idx = int(prop2[1])
                    if ('interp_coeffs' not in self.library[cat][glass]):
                        print('Calculating dispersion coefficients for "' + glass + '" ...')
                        self.get_polyfit_dispersion(glass, cat)
                    if ('interp_coeffs' in self.library[cat][glass]):
                        p2_coeffs = self.library[cat][glass]['interp_coeffs'][idx]
                    else:
                        print('Could not find valid interpolation coefficients for "' + glass + '" glass ...')
                        continue
                else:
                    p2.append(self.library[cat][glass][prop2])

            glassnames.append(glass)
            p1.append(p1_coeffs)
            p2.append(p2_coeffs)

            plt.plot(p1, p2, 'o', markersize=5)
            all_p1.extend(p1)
            all_p2.extend(p2)

        #DataCursor.DataCursor([ax])     ## turn on the feature of getting a box pointing to data points
        plt.title('catalog "' + catalog + '": ' + prop1 + ' vs. ' + prop2)
        plt.xlabel(prop1)
        plt.ylabel(prop2)

        ## Enforce the plotting range.
        xmin = min(all_p1)
        xmax = max(all_p1)
        xrange = xmax - xmin
        if (xrange < 1.0): xrange = 1.0
        xbot = xmin - (0.05 * xrange)
        xtop = xmax + (0.05 * xrange)
        xdist = 0.01 * xrange               ## for plotting text near the data points

        ymin = min(all_p2)
        ymax = max(all_p2)
        yrange = ymax - ymin
        if (yrange < 1.0E-9): yrange = 1.0
        ybot = ymin - (0.05 * yrange)
        ytop = ymax + (0.05 * yrange)
        ydist = 0.01 * yrange               ## for plotting text near the data points

        plt.axis([xbot,xtop,ybot,ytop])
        leg = plt.legend(catalogs, prop={'size':10}, loc='best')
        leg.draggable()
        #leg = plt.legend(catalogs, prop={'size':10}, bbox_to_anchor=(1.2,1))

        if show_labels:
            ## Plot all of the glass labels offset by (5,5) pixels in (x,y) from the data point.
            trans_offset = offset_copy(ax.transData, fig=fig, x=5, y=5, units='dots')
            for i in arange(alen(glassnames)):
                #print('i=%i: glassname=%s, p1=%f, p2=%f' % (i, glassnames[i], p1[i], p2[i]))
                plt.text(all_p1[i], all_p2[i], glassnames[i], fontsize=7, zorder=0, transform=trans_offset, color='0.5')

        return


## =============================================================================
## End of ZemaxLibrary class
## =============================================================================

def read_library(glassdir, catalog='all'):
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

    if (len(catalog) > 1) and isinstance(catalog, list):
        catalogs = catalog
    else:
        catalogs = [catalog]

    ## Get the set of catalog names. These keys will initialize the glasscat dictionary.
    glass_library = {}

    for f in files:
        print('Reading ' + f + ' ...')      #zzz
        this_catalog = os.path.basename(f)[:-4]
        if (this_catalog.lower() not in catalogs) and (catalog != 'all'): continue
        glass_library[this_catalog] = parse_glass_file(f)

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
            glass_catalog[glassname]['exclude_sub'] = 0 if (len(nm) < 7) else int(nm[6])
            glass_catalog[glassname]['status'] = 0 if (len(nm) < 8) else int(nm[7])
            glass_catalog[glassname]['meltfreq'] = 0 if ((len(nm) < 9) or (nm.count('-') > 0)) else int(nm[8])
        elif line.startswith('ED '):
            ed = line.split()
            glass_catalog[glassname]['tce'] = float(ed[1])
            glass_catalog[glassname]['density'] = float(ed[3])
            glass_catalog[glassname]['dpgf'] = float(ed[4])
            glass_catalog[glassname]['ignore_thermal_exp'] = 0 if (len(ed) < 6) else int(ed[5])
        elif line.startswith('CD '):
            cd = line.split()[1:]
            glass_catalog[glassname]['cd'] = [float(a) for a in cd]
        elif line.startswith('TD '):
            td = line.split()[1:]
            glass_catalog[glassname]['td'] = [float(a) for a in td]
        elif line.startswith('OD '):
            od = line.split()[1:]
            od = string_list_to_float_list(od)
            glass_catalog[glassname]['relcost'] = od[0]
            glass_catalog[glassname]['cr'] = od[1]
            glass_catalog[glassname]['fr'] = od[2]
            glass_catalog[glassname]['sr'] = od[3]
            glass_catalog[glassname]['ar'] = od[4]
            if (len(od) == 6):
                glass_catalog[glassname]['pr'] = od[5]
            else:
                glass_catalog[glassname]['pr'] = -1.0
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

## =================================================================================================
def string_list_to_float_list(x):
    '''
    Convert a list of strings to a list of floats, where a string value of '-' is mapped to a
    floating point value of -1.0, and an empty input list produces a length-10 list of -1.0's.

    Parameters
    ----------
    x : list
        The list of strings to convert

    Returns
    -------
    res : list of floats
        The converted results.
    '''
    npts = len(x)
    if (npts == 0) or ((npts == 1) and (x[0].strip() == '-')):
        return([-1.0]*10)

    res = []
    for a in x:
        if (a.strip() == '-'):
            res.append(-1.0)
        else:
            res.append(float(a))

    return(res)

## =================================================================================================
def find_catalog_for_glassname(glass_library, glassname):
    '''
    Search for the catalog containing a given glass.

    Note that this is not a perfect solution --- it is common for multiple catalogs to share glass
    names, and this function will only return the first one it finds.

    Parameters
    ----------
    glass_library : ZemaxGlassLibrary
        The glass library to search through.
    glassname : str
        The name of the glass to search for.

    Returns
    -------
    catalog : str
        The name of the catalog where the glass is found. If not found, then return None.
    '''
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

## =================================================================================================
def get_colors(num_colors):
    '''
    Make a list of 16 discernably different colors that can be used for drawing plots.

    Returns
    -------
    mycolors : list of floats
        A 16x4 list of colors, with each color being a 4-vector (R,G,B,A).
    '''

    mycolors = [None]*16
    mycolors[0]  = [0.0,0.0,0.0,1.0]        ## black
    mycolors[1]  = [1.0,0.0,0.0,1.0]        ## red
    mycolors[2]  = [0.0,0.0,1.0,1.0]        ## blue
    mycolors[3]  = [0.0,0.5,0.0,1.0]        ## dark green
    mycolors[4]  = [1.0,0.5,0.0,1.0]        ## orange
    mycolors[5]  = [0.0,0.5,0.5,1.0]        ## teal
    mycolors[6]  = [1.0,0.0,1.0,1.0]        ## magenta
    mycolors[7]  = [0.0,1.0,0.0,1.0]        ## lime green
    mycolors[8]  = [0.5,0.5,0.0,1.0]        ## olive green
    mycolors[9]  = [1.0,1.0,0.0,1.0]        ## yellow
    mycolors[10] = [0.5,0.0,0.0,1.0]        ## maroon
    mycolors[11] = [0.5,0.0,0.5,1.0]        ## purple
    mycolors[12] = [0.7,0.7,0.7,1.0]        ## bright grey
    mycolors[13] = [0.0,1.0,1.0,1.0]        ## aqua
    mycolors[14] = [0.4,0.4,0.4,1.0]        ## dark grey
    mycolors[15] = [0.0,0.0,0.5,1.0]        ## navy blue
    return(mycolors[:num_colors])

## =============================================================================================
if (__name__ == '__main__'):
    wavemin = 400.0
    wavemax = 700.0

    glasslib = ZemaxGlassLibrary(catalog='all')
    #glasslib = ZemaxGlassLibrary(degree=5)
    print('Number of glasses found in the library: ' + str(glasslib.nglasses))
    print('Glass catalogs found:', glasslib.catalogs)
    print(glasslib.glasses)

    glasslib.plot_dispersion('N-BK7', 'schott')
    glasslib.plot_dispersion('SF66', 'schott', polyfit=True, fiterror=True)
    #glasslib.plot_catalog_property_diagram('all', prop1='vd', prop2='nd')
    #glasslib.plot_catalog_property_diagram('all', prop1='nd', prop2='dispform')
    #glasslib.plot_catalog_property_diagram('schott', prop1='n0', prop2='n1')

    glasslib.plot_catalog_property_diagram('all', prop1='n0', prop2='n1')

    #glasslib.pprint('schott')          ## print the glass info found in the Schott glass catalog
    glasslib.pprint()                   ## print the glass infor for the entire library of glasses
    #glasslib.pprint('schott','SF66')   ## print the glass info for SF66 in the Schott glass catalog

    glasslib = ZemaxGlassLibrary(degree=5, wavemin=7500.0, wavemax=14500.0, catalog='infrared')
    print('Number of glasses found in the library: ' + str(glasslib.nglasses))
    print('Glass catalogs found:', glasslib.catalogs)
    glasslib.plot_dispersion('ZNS_BROAD', 'infrared')
    plt.show()
