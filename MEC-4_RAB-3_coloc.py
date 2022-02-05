# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 11:40:21 2021

@author: alkad
"""

#import basic libraries for plotting, data structures and signal processing
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas
from scipy.signal import find_peaks, peak_widths, butter, filtfilt
from scipy.ndimage import gaussian_filter, minimum_filter
import imageio
import os
import fnmatch
import datetime
import seaborn as sns

#%%
fpath = 'G:/My Drive/ECM manuscript/github codes/MEC-4_RAB-3_vesicle_colocalization/sample_data/input_files/' #filepath where the data is
imgfiles = fnmatch.filter(os.listdir(fpath), '*.tif')

dfpath = 'G:/My Drive/ECM manuscript/github codes/MEC-4_RAB-3_vesicle_colocalization/sample_data/output_files/'                
toa = str(datetime.datetime.today()).split()
today = toa[0]
now = toa[1]
timestamp = today.replace('-','')+'-'+now.replace(':','')[:6]

#%%
#PARAMETERS
code_version = os.path.basename(__file__)                  #stores the filename of the current script in f
mu_per_px = 0.252     #pixels to microns conversion factor (60x, 2x2 binning)

max_dist=595          #maximum distance from cell body to analyze (200 um = 795 px, 150 um=595 px)
dist = np.arange(max_dist)*mu_per_px                      #pixel to microns conversion


sigma_bgf=10
sigma_nf_2=20
sigma_nf_3=5

# Design the Buterworth filter
N  = 2    # Filter order
Wn = .5 # Cutoff frequency
B, A = butter(N, Wn, output='ba')

def neurite_fluorescence(img):
    n = img[3:7, 0:]                       #extract rows to use for neurite
    bg = np.concatenate((img[0:2, 0:], img[8: , 0:]))   #extract rows to use for background
    rawf = np.mean(n, axis=0)               #calculate average raw neurite fluorescence
    bgf = gaussian_filter(np.mean(bg, axis=0), sigma=sigma_bgf)             #calculate average background fluorescence
    nf = rawf - bgf                         #calculate background subtracted neurite fluorescence
    for i in range(0,len(nf)): 
        if nf[i]<0: nf[i]=0
    
    fnf_1 = filtfilt(B,A, nf)
    fnf_2 = minimum_filter(nf, sigma_nf_2)
    fnf_3 = gaussian_filter(fnf_2, sigma_nf_3)
    fnf = fnf_1-fnf_3
    return(fnf_1,fnf_3,fnf)

def height_cutoff(fnf):
    avnoise = np.mean(fnf[fnf < np.percentile(fnf, 75)])
    stdnoise = np.std(fnf[fnf < np.percentile(fnf, 75)])
    height = avnoise + 5*stdnoise
    return(height)

def peakfinder(nf,fnf,height):
    peaks_pos=find_peaks(fnf, height=height, prominence=0.5*height)[0]
    pw=peak_widths(nf, peaks_pos, rel_height=0.5)[0]
    fpmi=[fnf[i] for i in peaks_pos]
    #only keep wide peaks if they are also bright
    indices=[]
    for i in np.arange(0, len(peaks_pos)):
        if pw[i]>4 and pw[i]<8 and fpmi[i]<1.5*height: indices = indices+[i]
        elif pw[i]>=8 and fpmi[i]<3*height: indices = indices+[i]
    peaks_pos=np.delete(peaks_pos,indices)
    pw=np.delete(pw, indices)
    fpmi=np.delete(fpmi,indices)
    pd=peaks_pos*mu_per_px
    pmi=[nf[i] for i in peaks_pos]
    return(peaks_pos, pd, pw, pmi, fpmi)


def colocalization(A,B):
    coloc=[]
    n=0
    for peak in A:
        yn='no'
        for p in np.arange(peak-2, peak+3):
            if p in B:
                yn = 'yes'
                n=n+1
                break
        coloc=np.append(coloc, yn)
    return(coloc,n)

#specify columns of the pandas dataframe and excel sheets
cols_Data =     ['Strain', 'Allele', 'ImageID', 'Distance', 'Green neurite intensity', 'Green baseline intensity', 'Red neurite intensity']
cols_Peaks =    ['Strain', 'Allele', 'ImageID', 'Channel', 'Puncta distance', 'Puncta max intensity', 'Colocalization']
cols_Analysis = ['Strain', 'Allele', 'ImageID', 'Average green fluorescence', 'Average baseline fluorescence', 'Height_cutoff_green', 'Height_cutoff_red', 'Total green peaks', 'Total red peaks', 'Coloc_GR', 'Coloc_RG', 'Frac_GR', 'Frac_RG']
cols_Rejects =  ['ImageID', 'Max neurite length', 'Reason']


#initialize Pandas DataFrames
df_Data = pandas.DataFrame()
df_Peaks = pandas.DataFrame()
df_Analysis = pandas.DataFrame()
df_Rejects = pandas.DataFrame()
strain_key=pandas.DataFrame({('x134', 'GN1016', 'WT', 0),
                             ('x137', 'GN1027','mec-9(u437)', 0),
                             ('x136', 'GN1026','mec-1(e1738)', 0)
                             }, columns=['StrainID','Strain name','Allele', 'n'])
# strain_key=strain_key.set_index('Strain')


#%%

for x in imgfiles:                            #create loop for number of images in folder
    img_G = imageio.imread(fpath+x)[:,:,1]    #import image and store it in a list of lists
    img_R = imageio.imread(fpath+x)[:,:,0]    #import image and store it in a list of lists

    imsize = np.shape(img_G)                  #calculate image size
    if imsize[1] <max_dist:
        reason = 'Neurite length is less than minimum length'
        frame = pandas.DataFrame([[x, imsize[1]*mu_per_px, reason]], columns=cols_Rejects)
        df_Rejects = df_Rejects.append(frame)
        continue   #reject images where neurite length is less than 150 um

    
    #extract info from filename
    strain = x.split('_')[1].split('-')[0]
    row_index=strain_key[(strain_key['StrainID']==strain)|(strain_key['Strain name']==strain)].index[0]
    allele = strain_key.loc[row_index,'Allele']
    count = strain_key.loc[row_index,'n'] + 1
    
    nG, bsG, fnG = neurite_fluorescence(img_G[:,:max_dist])
    nR, bsR, fnR = neurite_fluorescence(img_R[:,:max_dist])
    all_data1 = pandas.DataFrame({'Strain':[strain]*max_dist, 'Allele':[allele]*max_dist, 'ImageID':[x]*max_dist, 'Distance':dist, 'Green neurite intensity':nG, 'Green baseline intensity':bsG, 'Red neurite intensity':nR}, columns=cols_Data)
    df_Data=df_Data.append(all_data1)
    
    heightG = 8
    heightR = height_cutoff(fnR)
    
    peaks_posG, pdG, pwG, pmiG, fpmiG = peakfinder(nG, fnG, heightG)
    peaks_posR, pdR, pwR, pmiR, fpmiR = peakfinder(nR, fnR, heightR)

    colocGR, nGR = colocalization(peaks_posG,peaks_posR)        
    colocRG, nRG = colocalization(peaks_posR,peaks_posG)  

    all_peaksG = pandas.DataFrame({'Strain':[strain]*len(pdG), 'Allele':[allele]*len(pdG), 'ImageID':[x]*len(pdG), 'Channel':['Green']*len(pdG), 'Puncta distance':pdG, 'Puncta max intensity':pmiG, 'Colocalization':colocGR}, columns=cols_Peaks)
    df_Peaks=df_Peaks.append(all_peaksG)
    all_peaksR = pandas.DataFrame({'Strain':[strain]*len(pdR), 'Allele':[allele]*len(pdR), 'ImageID':[x]*len(pdR), 'Channel':['Red']*len(pdR), 'Puncta distance':pdR, 'Puncta max intensity':pmiR, 'Colocalization':colocRG}, columns=cols_Peaks)
    df_Peaks=df_Peaks.append(all_peaksR)
                
    if len(pdG)==0:frac_GR='N/A'
    else: frac_GR=nGR/len(pdG)
    if len(pdR)==0:frac_RG='N/A'
    else: frac_RG=nRG/len(pdR)

    frame = pandas.DataFrame([[strain, allele, x, np.mean(nG), np.mean(bsG), heightG, heightR, len(pdG), len(pdR), nGR, nRG, frac_GR, frac_RG]], columns=cols_Analysis)
    df_Analysis = df_Analysis.append(frame)
    
    
    plt.figure(1, figsize=(16,8))
    grid = plt.GridSpec(10, 1, wspace=0.5, hspace=0.5)

    plt.subplot(grid[0, :])
    plt.imshow(img_G[:,:max_dist], vmin=0,vmax=100)   

    plt.subplot(grid[1, :])
    plt.imshow(img_R[:, :max_dist], vmin=0,vmax=150)

    plt.subplot(grid[2:6, :])
    plt.title(x+' Green fluorescence')
    plt.xlabel('Distance (um)')
    plt.ylabel('Intensity (AU)')
    plt.axis([0, max(dist), 0, 100])
    plt.plot(dist, nG, 'g-')
    # plt.plot(dist,bsG,'g')
    plt.plot(pdG, pmiG, 'go')
    # plt.hlines(pwG[1], pwG[2]*mu_per_px, pwG[3]*mu_per_px)
    plt.eventplot(list(all_peaksG['Puncta distance'][all_peaksG['Colocalization']=='yes']), linelengths=100)

    plt.subplot(grid[6:, :])
    plt.title(x+' Red peaks')
    plt.xlabel('Distance (um)')
    plt.ylabel('Intensity (AU)')
    plt.axis([0, max(dist), 0, max(fnR)])
    plt.plot(dist, fnR, 'r-')
    plt.plot(pdR, fpmiR, 'ro')
    # plt.hlines(pwR[1], pwR[2]*mu_per_px, pwR[3]*mu_per_px)
    plt.eventplot(list(all_peaksR['Puncta distance'][all_peaksR['Colocalization']=='yes']), linelengths=100)
    plt.hlines(heightR, 0, max(dist), 'r')

    sns.set_style('white')
    sns.set_style('ticks', {'xtick.direction': 'in', 'ytick.direction': 'in'})
    sns.despine(offset=5, trim=False)
    
    plt.rcParams.update({'font.size': 10})
    plt.rcParams['svg.fonttype'] = 'none'

    plt.savefig(dfpath+timestamp+'_'+x+'_plot.png')
    plt.show()
    plt.close()

    strain_key.at[row_index,'n']=count
#%%
df_temp=pandas.DataFrame()
for allele in strain_key['Allele']:
    npeaksG = len(df_Peaks[(df_Peaks['Allele']==allele)&(df_Peaks['Channel']=='Green')])
    npeaksR = len(df_Peaks[(df_Peaks['Allele']==allele)&(df_Peaks['Channel']=='Red')])
    frac_gr=len(df_Peaks[(df_Peaks['Allele']==allele)&(df_Peaks['Channel']=='Green')&(df_Peaks['Colocalization']=='yes')])/npeaksG
    frac_rg=len(df_Peaks[(df_Peaks['Allele']==allele)&(df_Peaks['Channel']=='Red')&(df_Peaks['Colocalization']=='yes')])/npeaksR
    frame = pandas.DataFrame([[allele, npeaksG, npeaksR, frac_gr, frac_rg]], columns=['Allele', 'npeaksG', 'npeaksR', 'frac_gr', 'frac_rg'])
    df_temp = df_temp.append(frame)
df_new=strain_key.merge(df_temp, how='outer', on='Allele')

#%%
#save data to excel file
wb = pandas.ExcelWriter(dfpath+timestamp+'_Analysis.xlsx', engine='xlsxwriter')
df_Analysis.to_excel(wb, sheet_name='Analysis')
df_new.to_excel(wb, sheet_name='Summary')
df_Rejects.to_excel(wb, sheet_name='Rejects')
wb.save()

df_Data.to_pickle(dfpath+timestamp+'_Data.pkl')
df_Peaks.to_pickle(dfpath+timestamp+'_Peaks.pkl')
df_Analysis.to_pickle(dfpath+timestamp+'_Analysis.pkl')
