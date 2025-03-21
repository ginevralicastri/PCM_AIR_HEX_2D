"""
Created on Sat Nov  5 15:20:07 2022

@author: Manuela Baracani

This script unload the TWINS DT files from the ftp server and join them.


TO USE THE SCRIPT:

    1) In the variable <path> put  r' + the directory in which you want to save the output file
    
    2) Decide the date from which collect data

    3) Press run
    
"""


### MODULES ###

import ftplib
# import glob
import os
import pandas as pd
import io
from datetime import datetime, timedelta
import requests
import json

# from ftplib import FTP
# from dateutil import parser


### VARIABLES ###

# directory of output file
path = r'C:\Users\ginevra.licastri\Desktop\Python Projects\PCM_AIR_HEX_2D\data'
 
# starting date of correct data monitoring (Twin A)
start_date_A = datetime.strptime("22-06-2024", '%d-%m-%Y')

# starting date of correct data monitoring (Twin B)
start_date_B = datetime.strptime("22-06-2024", '%d-%m-%Y')

# starting date of correct data monitoring (Climate)
start_date_CS = datetime.strptime("22-06-2024", '%d-%m-%Y')

delta_hour_Solcast = 0 # "0" for legal hour, "1" for solar hour

delta_hour_Polito = 0 # "0" for legal hour, "1" for solar hour

# frequency to which resample climate dataframes to join
resample_freq_CLIMATE = "5min"

# frequency to which resample test cell dataframes to join
resample_freq_TESTCELL = "5min"

#-------------------------------------------------------------------------------------------------------------



### ftp server credentials
ftp=ftplib.FTP(host='ftp.polito.it', user='ftp-denerg', passwd='Denerg-TWINS-20$19')       
ftp.login(user='ftp-denerg')



### FUNCTIONS ###

## Unload TWINS DT data   
def unload_and_join_from_ftp_TWINS(ftpfiles,
                             start_date,
                             output_filename):
        
    for file in ftpfiles:
        
        # datetime
        datetime_str = file[4:12] 

        date_file = datetime.strptime(datetime_str, '%Y%m%d') 
    
        
        # checking if date is after the starting of monitoring
        if date_file >= start_date:  
            print(date_file)

            # download file from ftp
            download_file = io.BytesIO()
            ftp.retrbinary('RETR ' + str(file), download_file.write)
            download_file.seek(0)

            
            # open content 
            open_file = pd.read_csv(download_file,  
                                    index_col = "Timestamp",
                                    parse_dates = True,
                                    engine='python')  
            
            # correct datetime
            open_file.index=open_file.index.round('min')
            
            # Resample to 15-minute intervals and calculate the mean
            resampled_data = open_file.resample(resample_freq_TESTCELL).mean(numeric_only=True)
            
            # move in the directory of the output file
            os.chdir(path)  
            
            # check if the output file already exist: if yes append values
            if os.path.isfile(output_filename):
                concat_file = pd.read_csv(output_filename,
                                          index_col = "Timestamp",
                                          parse_dates = True)       
                
                # check if the data have been already saved
                if concat_file.tail(1).index < resampled_data.head(1).index:

                    # append values
                    concat_file = pd.concat([concat_file,resampled_data], axis=0)
                    
                    # save file
                    concat_file.to_csv(output_filename)
             
            # check if the output file already exist: if not create file   
            else:
                open_file.to_csv(output_filename)
                
                
# Unload Solcast data
def unload_and_join_from_ftp_CS(ftpfiles,
                             start_date,
                             output_filename):
                  
    for file in ftpfiles:
        
        # datetime
        datetime_str = file[0:10] 
    
        date_file = datetime.strptime(datetime_str, '%Y_%m_%d') 
    
        # hour
        hour_file = datetime.strptime(file[0:16], '%Y_%m_%d_%H_%M')
        # hour_file.round('')
    
     
        # checking if date is after the starting of monitoring
        if date_file >= start_date_CS:  
            print(date_file)
    
            # download file from ftp
            download_file = io.BytesIO()
            ftp.retrbinary('RETR ' + str(file), download_file.write)
            download_file.seek(0)
            
            # open content 
            open_file = pd.read_csv(download_file,  
                                    usecols = [1,2,3,4,5],
                                    engine='python') 
            
            # correct datetime with UTC delta
            open_file['Timestamp'] = hour_file + timedelta(hours=delta_hour_Solcast)
            
            # set datetime as index
            open_file.set_index('Timestamp', drop = True, inplace=True)
            
            # correct datetime
            open_file.index=open_file.index.round('30min')
    
            # I move in the directory of the output file
            os.chdir(path)  
            
            # check if the output file already exist: if yes append values
            if os.path.isfile(output_filename):
                concat_file = pd.read_csv(output_filename,
                                          index_col = "Timestamp",
                                          parse_dates = True)       
                
                # check if the data have been already saved
                if concat_file.tail(1).index < open_file.head(1).index:
    
                    # append values
                    concat_file = pd.concat([concat_file,open_file], axis=0)
                    
                    # save file
                    concat_file.to_csv(output_filename)
             
            # check if the output file already exist: if not create file   
            else:
                open_file.to_csv(output_filename)
                        
       



def getValuesMeteo(start_date,end_date):
    """
    When called returns a a multi-row dataframe of the measures performed within the time 
    interval defined by from_date (char) and to_date (char). The time interval requested begins 
    from the 00:00 of the from_date and finishes at 00:00 of the to_date.
    
    +++Warning: The maximum interval of time that can be requested is 1 year. 
    If the period exceeds one year the system returns an error and the application stops +++

    Parameters
    ----------
    start_date : specific date str type in format %Y-%m-%d
    end_date : specific date str type in format %Y-%m-%d

    Returns
    -------
    data_cur : DataFrame type to be saved into csv file

    """
    val="http://smartgreenbuilding.polito.it/webservices/LivingLABWS/TEBE_BAEDAgroup/meter.asp?getValuesMeteo?from=%s&to=%s" %(start_date,end_date)
    response = requests.get(val)
    data = json.loads(response.text)
    return pd.DataFrame.from_dict(data)               
#----------------------------------------------------------------------------------------------------#                
### OPERATIONS ###        
#----------------------------------------------------------------------------------------------------#                


## TWIN A DIRECTORY


# change folder
ftp.cwd('TWINS_A/01_TC1')  

# file list (sorted depending on the entire file name)
# ftpfiles=sorted(ftp.nlst())

# file list (sorted depending on the  datetime)
ftpfiles = sorted(ftp.nlst(), key=lambda s: s[3:])


# name of the output file
A_filename = "20_12-11_01-SA.csv".format(datetime.now().date())

# Unloading and saving (function)
unload_and_join_from_ftp_TWINS(ftpfiles,
                             start_date_A,
                             A_filename)  

A_dataframe = pd.read_csv(A_filename, index_col="Timestamp", parse_dates=True)




## TWIN B DIRECTORY

### ftp server credentials
ftp=ftplib.FTP(host='ftp.polito.it', user='ftp-denerg', passwd='Denerg-TWINS-20$19')       
ftp.login(user='ftp-denerg')

# change folder
ftp.cwd('TWINS_B/01_PS+monitoring')  

# file list (sorted depending on the entire file name)
# ftpfiles=sorted(ftp.nlst())

# file list (sorted depending on the  datetime)
ftpfiles = sorted(ftp.nlst(), key=lambda s: s[3:])


# name of the output file
B_filename = "Twin_B_data__unload on {}.csv".format(datetime.now().date())

# Unloading and saving (function)
unload_and_join_from_ftp_TWINS(ftpfiles,
                             start_date_B,
                             B_filename)  

B_dataframe = pd.read_csv(A_filename, index_col="Timestamp", parse_dates=True)

# SOLCAST DIRECTORY


### ftp server credentials
ftp=ftplib.FTP(host='ftp.polito.it', user='ftp-denerg', passwd='Denerg-TWINS-20$19')       
ftp.login(user='ftp-denerg')

# change folder
ftp.cwd('solcast_data')  

# file list
ftpfiles=sorted(ftp.nlst()) 

# name of the output file
solcast_filename = "Solcast_merged_data__unload on {}.csv".format(datetime.now().date())

# Unload Solcast data
unload_and_join_from_ftp_CS(ftpfiles,
                              start_date_CS,
                              solcast_filename)


## JOIN CLIMATE DATA

## Request Polito meteo station data


## Unload Polito Station data

end_date=datetime.now().date() + timedelta(days=1)
Polito_data=getValuesMeteo(start_date_CS.date(),end_date)
output_meteo_filename = "Meteo station_data__unload on {}.csv".format(datetime.now().date())
Polito_data.to_csv(output_meteo_filename)
 
# hour
hour_polito = pd.to_datetime(Polito_data["DATA_ORA_MINUTI_RILEVAMENTO_BASE_ITA"], dayfirst=True, format='%d/%m/%Y %H:%M:%S')

# cleaning dataframe
df_Polito = Polito_data[["DIRVento","VELVento","PIOGGIA","pressATM","r_DIFF","r_DIR_h","r_DIR_n","radGLOBale","tempARIA","umiditaREL"]]

# transforming values in numbers
df_Polito = df_Polito.apply(pd.to_numeric)
    
# correct datetime with UTC delta
df_Polito['Timestamp'] = hour_polito + timedelta(hours=delta_hour_Polito)

# set datetime as index
df_Polito.set_index('Timestamp', drop = True, inplace=True)

#!!! rename the headers

# rename columns
df_Polito.columns = [f'Polito_{x}' for x in df_Polito.columns]

# Interpolate to obtain same minutes data
Polito_res = df_Polito.resample(resample_freq_CLIMATE).mean(numeric_only=True).interpolate(method='linear')


# Open Solcast data


# Check if the Solcast file exists before trying to open it
if os.path.isfile(os.path.join(path, solcast_filename)):
    solcast_file = pd.read_csv(os.path.join(path,solcast_filename),  
                                        index_col = "Timestamp",
                                        parse_dates = True,
                                        usecols = ["Timestamp","ghi","ebh","dni","dhi","cloud_opacity"],
                                        engine='python') 
    
    #!!! rename the headers
    
    # rename columns
    solcast_file.columns = [f'Solcast_{x}' for x in solcast_file.columns]
    
    
    # Interpolate to obtain same minutes data
    Solcast_res = solcast_file.resample(resample_freq_CLIMATE).asfreq()
        
    
## Join to weather station data


## open TWIN A data (temperature, radiation (from Hukserflux and SPN1) and wind)
# Check if the TWIN A file exists before trying to open it
if os.path.isfile(os.path.join(path, A_filename)):
    TWIN_A_file = pd.read_csv(os.path.join(path,A_filename),  
                                        index_col = "Timestamp",
                                        parse_dates = True,
                                        usecols = ["Timestamp","Tae (degC (Ave))","Tae_South (degC (Ave))","SPN1_Diff_Hor (mV (Ave))","SPN1_Glob_Hor (mV (Ave))","SPN1_Sun (V (Ave))","Pyra_Out_Ver (mV (Ave))","wind speed (m/s)","wind dir (deg)"],
                                        engine='python')
    # Convert index to DatetimeIndex
    TWIN_A_file.index = pd.to_datetime(TWIN_A_file.index)
    # rename columns
    TWIN_A_file.columns = [f'Twin_A_{x}' for x in TWIN_A_file.columns]
    
    # Interpolate to obtain same minutes data
    TWIN_A_res = TWIN_A_file.resample(resample_freq_CLIMATE).mean(numeric_only=True).interpolate(method='linear')

## open TWIN B data (radiation)
# Check if the TWIN B radiation file exists before trying to open it
if os.path.isfile(os.path.join(path, B_filename)):
    TWIN_B_file = pd.read_csv(os.path.join(path,B_filename),  
                                        index_col = "Timestamp",
                                        parse_dates = True,
                                        usecols = ["Timestamp","Pyra_Out_Vert (mV)"],
                                        engine='python') 
    
    # Convert index to DatetimeIndex
    TWIN_B_file.index = pd.to_datetime(TWIN_B_file.index)
    # rename columns
    TWIN_B_file.columns = [f'Twin_B_{x}' for x in TWIN_B_file.columns]
    
    # Interpolate to obtain same minutes data
    TWIN_B_res = TWIN_B_file.resample(resample_freq_CLIMATE).mean(numeric_only=True).interpolate(method='linear')

## Join all climatic data


# Append values from Polito data, if available
climatic_file = Polito_res


# Append values from TWIN A, if available
if not TWIN_A_file.empty:
    TWIN_A_res = TWIN_A_file.resample(resample_freq_CLIMATE).mean(numeric_only=True).interpolate(method='linear')
    climatic_file = pd.concat([climatic_file, TWIN_A_res], axis=1)

# Append values from TWIN B, if available
if not TWIN_B_res.empty:
    climatic_file = pd.concat([climatic_file, TWIN_B_res], axis=1)

# Append values from Solcast data, if available
# if not Solcast_res.empty:
#     climatic_file = pd.concat([climatic_file, Solcast_res], axis=1)

# Join the directory and filename
output_filename_climate = "Climate_merged_data__unload on {}.csv".format(datetime.now().date())
output_path_climate = os.path.join(path, output_filename_climate)

# Normalize the path to replace backslashes with forward slashes
output_path_climate = os.path.normpath(output_path_climate)



# check if the output file already exist: if yes append values
if os.path.isfile(output_path_climate):
    # Read the existing file if it exists
    existing_file = pd.read_csv(output_path_climate, index_col="Timestamp", parse_dates=True)

    # check if the data have been already saved
    if existing_file.tail(1).index < climatic_file.head(1).index:
        # Append the data
        merged_data = pd.concat([existing_file, climatic_file], axis=0)
        # save file
        merged_data.to_csv(output_path_climate)
# check if the output file already exist: if not create file

else:
    # Save the climatic data to a new output file
    climatic_file.to_csv(output_path_climate)