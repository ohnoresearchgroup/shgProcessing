import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class SHGfromCCD():
    def __init__(self,path,time_interval = 1):
        #import data into dataframe
        df = pd.read_csv(path)

        #store time interval
        self.time_interval = time_interval

        #create array to store wavelengts
        df_wls = df[df["Frame"] == 1]
        self.wls = df_wls['Wavelength'].values
        
        #store number of pixels and number of frames
        self.num_pixels = len(self.wls)
        self.num_frames = df["Frame"].max()

        #create time axis
        self.time = np.arange(0, self.time_interval*self.num_frames, self.time_interval)

        #remove extra columns and reshape into array of intensities for each frame
        df.drop(columns=["ROI","Wavelength","Frame","Row","Column"],inplace=True)
        self.data = df["Intensity"].values.reshape(-1, self.num_pixels)

    def backgroundSubtract(self,bounds=[515,520]):
        #find indices that correspond to 
        lower_lim = bounds[0]
        upper_lim = bounds[1]
        lower_idx = np.abs(self.wls - lower_lim).argmin()
        upper_idx = np.abs(self.wls - upper_lim).argmin()

        #array to hold intensities
        self.intensities = np.zeros(self.num_frames)

        #iterate through each frame to background subtract
        for i in range(self.num_frames):
            #slice out narrower range to fit and bg subtract
            y = self.data[i][lower_idx:upper_idx]
            x = self.wls[lower_idx:upper_idx]

            #find indices of inner region with actual peak 
            lower_inner_idx = np.abs(x - (lower_lim + 1)).argmin()
            upper_inner_idx = np.abs(x - (upper_lim - 1)).argmin()

            x_fit = np.concatenate((x[0:lower_inner_idx],x[upper_inner_idx:]))
            y_fit = np.concatenate((y[0:lower_inner_idx],y[upper_inner_idx:]))

            #fit background region with line, subtract it
            slope, intercept = np.polyfit(x_fit, y_fit, 1)
            y_line = x*slope + intercept
            y_sub = y - y_line

            #calculate intensity through sum of background subtracted
            intensity = np.sum(y_sub[lower_inner_idx:upper_inner_idx])

            #store intensity
            self.intensities[i] = intensity

            #plot the first and last to be sure it worked well
            if (i == 0 or i == self.num_frames-1):
                plt.figure()
                plt.plot(x,y)
                plt.plot(x_fit,y_fit,'.')
                plt.plot(x,y_line)
                plt.xlabel('Wavelength [nm]')
                plt.ylabel('Intensity')
                plt.title("Frame " + str(i))

                plt.figure()
                plt.plot(x[lower_inner_idx:upper_inner_idx],y_sub[lower_inner_idx:upper_inner_idx])
                plt.xlabel('Wavelength [nm]')
                plt.ylabel('Intensity')
                plt.title("Frame " + str(i))

    def plotIntensities(self):
        plt.plot(self.time,self.intensities)
        plt.xlabel('Time')