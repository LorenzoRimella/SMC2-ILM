def prevalence(Y_ecoli):
    
    Y_sum         = np.nansum(Y_ecoli, axis = 0)
    Y_sum_samples = np.sum(1-np.isnan(Y_ecoli), axis = 0)
    
    if Y_sum[0]==0:
        Y_const_sum = 13
    else:
        Y_const_sum = Y_sum[0]
        
    if Y_sum_samples[0]==0:
        Y_const_sum_samples = 100
    else:
        Y_const_sum_samples = Y_sum_samples[0]
    
    for i in range(len(Y_sum)):
        if all(np.isnan(Y_ecoli[:,i])):
            Y_sum[i] = Y_const_sum
            Y_sum_samples[i] = Y_const_sum_samples
        else:
            Y_const_sum = Y_sum[i]
            Y_const_sum_samples = Y_sum_samples[i]
        

    return Y_sum/Y_sum_samples

def prevalence_m1(prev_input):
    
    prev = prev_input
    prev_const = prev[:,0]
    for i in range((prev.shape[1])):
        if all(prev[:,i]==-1):
            prev[:,i] = prev_const
        else:
            prev_const = prev[:,i]        

    return prev


import numpy as np

def old_date(t):
    
    t = t+ 149

    year_2019 = np.cumsum([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    year_2020 = year_2019[11]+np.cumsum([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    year_2021 = year_2020[11]+np.cumsum([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    
    
    if any(t<year_2019):
        month = np.min(np.argwhere(t<year_2019))+1
        
        if month==1:
            day   = t+1
        else:
            day   = t-year_2019[month-2]+1
        
        if day<10:
            day_string = "0"+str(int(day))
            
        else:
            day_string = str(int(day))
            
        if month<10:
            month_string = "0"+str(int(month))
            
        else:
            month_string = str(int(month))
            
        
        date = day_string + "-" + month_string + "-2019"
        
    elif any(t<year_2020):
        month = np.min(np.argwhere(t<year_2020))+1
        
        if month==1:
            day   = t-year_2019[11]+1
            
        else:
            day   = t-year_2020[month-2]+1
        
        if day<10:
            day_string = "0"+str(int(day))
            
        else:
            day_string = str(int(day))
            
        if month<10:
            month_string = "0"+str(int(month))
            
        else:
            month_string = str(int(month))
            
        
        date = day_string + "-" + month_string + "-2020"
        
    else:
        month = np.min(np.argwhere(t<year_2021))+1
        
        if month==1:
            day   = t-year_2020[11]+1
            
        else:
            day   = t-year_2021[month-2]+1
        
        if day<10:
            day_string = "0"+str(int(day))
            print(day)
            
        else:
            day_string = str(int(day))
            
        if month<10:
            month_string = "0"+str(int(month))
            
        else:
            month_string = str(int(month))
            
        
        date = day_string + "-" + month_string + "-2021"
        
#     if t>650:
#         date = ""
        
    return date


def format_func(value, tick_number):
    # find number of multiples of pi/2
    return old_date(value)