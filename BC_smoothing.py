##############################################################################
### Script to calculate the cavity thicknes of a Biosensor for every pixel ###
##############################################################################

#############
### INPUT ###
#############


# enter folder with data, no subfolders allowed
folder_list = ['40x_0.6_63ms'] 


# chose wavelength range and step-width

wave_start = 550    # [nm]
wave_end = 750      # [nm]


# enter average deviation of experiment to simulation in nanometer, "1" is a good value to start

tolerance = 1

# define parameters for minima detection  

lookahead_min = 5 # something like peak width for the minima
delta = 7    # something like peak height

# chose elastomer thickness range , the smaller the range the faster the program. If you are not sure, just take d_min = 1000, d_max = 19000

d_min= 6000   # [nm]
d_max= 11000 # [nm]

# parameters for smoothing

pos_x = 640
pos_y = 512

# smoothing in x-y direction

# enter True if you want to enable this smoothing
x_y_smooth = False
# enter sigma for the gaussian smoothing
x_y_sigma = 2
# folder to save smoothed images
folder_smmoth_x_y = '40x_0.6_63ms_x_y_smoothed'

use_thickness_limits = True # Enter "True" if you want to do calculation with thickness limits and "False" if not. I recommend starting with "False"

thickness_limit = 50 # [nm] enter the thickness limit (if thickness was found, next on will be: last_thickness +- thickness_limit)


# parameters for printing
# color map is calculated like (mean_thickness - color_min, mean_thickness + color_max) 

color_min = 500
color_max = 500

############################
### END of INPUT SECTION ###
############################



#############################
#### start of the program ###
#############################

#import cython_all_fit as Fit # import self-written cython code
import numpy as np
import time
import os 
import Image as im
from scipy import ndimage
from scipy import misc
import multiprocessing as mp
import matplotlib.pyplot as plt

t_a_start = time.time() # start timer for runtime measurement

if __name__ == '__main__':
    for folder in folder_list:
        # enter number of cpu cores, this has to be an integer number!
        # number of physical cores is a good start, but you can try with a larger as well

        multi_p = True   # True for multiprocessing, False for single core (Windows)
        cores = 4

        
        # enter name of simulation_file

        sim_file = 'Sim_0.5Cr_20Au_Elastomer_RT601_15Au_500_750nm.txt'

        lookahead_max = lookahead_min-1 # for the maxima --> should not be larger than lookahead_min

        # make wavelength list

        wave_step = 1       # [nm]
        
        waves=[]

        waves=[wave_start + i*wave_step for i in xrange((wave_end-wave_start)/wave_step + 1)]

        ## read image data 
        dateien=os.listdir(folder)
        dateien.sort()
        
        #generates an empty array --> image grey values 
        alle=np.zeros(((wave_end-wave_start)/wave_step + 1,1024,1280),np.uint16)
        alle_smooth=np.zeros(((wave_end-wave_start)/wave_step + 1,1024,1280),np.uint16)
        
        # define function to convert the image-string to an array
        def image2array(Img):
            newArr= np.fromstring(Img.tostring(),np.uint8)
            newArr= np.reshape(newArr, (1024,1280))
            return newArr

        # read every image in folder and check if it is in the wavelength range --> 
        # write grey values into array
        
        counter=0
        print 'reading images from folder: ', folder
        for i in xrange(len(dateien)):
            if dateien[i][-5:]=='.tiff':
                if int(dateien[i][:3]) >= wave_start and int(dateien[i][:3]) <= wave_end:
                    #print dateien[i]
                    print counter
                    Img=im.open(folder + '/' + dateien[i]).convert('L')
                    alle[counter]=image2array(Img)
        # smoothing in x-y direction
                    if x_y_smooth == True:
                        Img_s = ndimage.gaussian_filter(Img, sigma=x_y_sigma)
                        alle_smooth[counter] = image2array(Img_s)
                        misc.imsave(folder_smmoth_x_y + '/' + dateien[i],Img_s)
                    counter+= 1

        # function to print profiles for smoothed and not smoothed image in same plot
        def print_profiles(pos_x,pos_y,print_name,waves=waves,alle=alle,alle_smooth=alle_smooth):
            plt.figure(0)
            plt.plot(waves, alle[:,pos_y,pos_x], label='lambda smooth')
            #if x_y_smooth == True:
            #    plt.plot(waves, alle_smooth[:,pos_y, pos_x], label='x_y_smooth')
            plt.legend()
            plt.grid()
            plt.axis([wave_start,wave_end,0,250])
            #plt.show()
            plt.savefig('lambda_smooth_video' + '/' + print_name + '.png', bbox_inches='tight', pad_inches=0)
            plt.close()

        # save images in folder to make video
        # string = 'frame00000'
        # for pos_x in range(1280):
        #     print_name = string[:(len(string)-len(str(pos_x)))] + str(pos_x)
        #     print_profiles(pos_x,pos_y, print_name)
        #     print pos_x

        

        # 1D smoothing in lambda direction

        alle_lambda_smooth = alle.copy()

        # define function for 1D smoothing
        def smooth(x,window_len=2,window='hanning'):
            """smooth the data using a window with requested size.
            
            This method is based on the convolution of a scaled window with the signal.
            The signal is prepared by introducing reflected copies of the signal 
            (with the window size) in both ends so that transient parts are minimized
            in the begining and end part of the output signal.
            
            input:
                x: the input signal 
                window_len: the dimension of the smoothing window; should be an odd integer
                window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                    flat window will produce a moving average smoothing.

            output:
                the smoothed signal
                
            example:

            t=linspace(-2,2,0.1)
            x=sin(t)+randn(len(t))*0.1
            y=smooth(x)
            
            see also: 
            
            numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
            scipy.signal.lfilter
         
            TODO: the window parameter could be the window itself if an array instead of a string
            NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
            """

            if x.ndim != 1:
                raise ValueError, "smooth only accepts 1 dimension arrays."

            if x.size < window_len:
                raise ValueError, "Input vector needs to be bigger than window size."


            if window_len<3:
                return x


            if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
                raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


            s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
            #print(len(s))
            if window == 'flat': #moving average
                w=np.ones(window_len,'d')
            else:
                w=eval('np.'+window+'(window_len)')

            y=np.convolve(w/w.sum(),s,mode='valid')
            return y[(window_len/2-1):-(window_len/2)]

        def print_lambda_profiles(pos_x,pos_y,print_name,waves=waves,alle=alle,alle_smooth=alle_lambda_smooth):
            plt.figure(0)
            #plt.plot(waves, alle[:,pos_y,pos_x], label='raw data')
            plt.plot(waves, alle_smooth[:,pos_y, pos_x], label='lambda smooth')
            plt.legend()
            plt.grid()
            plt.axis([wave_start,wave_end,0,250])
            #plt.show()
            plt.savefig('lambda_smooth_video' + '/' + print_name + '.png', bbox_inches='tight', pad_inches=0)
            plt.close()
        for zeile in range(1024):
            print zeile
            for spalte in range(1280):
                alle_lambda_smooth[:,zeile,spalte] = ndimage.gaussian_filter1d(alle_lambda_smooth[:,zeile,spalte],1) 

        #save images in folder to make video
        string = 'frame00000'
        for pos_x in range(1280):
            print_name = string[:(len(string)-len(str(pos_x)))] + str(pos_x)
            print_lambda_profiles(pos_x,pos_y, print_name)
            print pos_x


        

       