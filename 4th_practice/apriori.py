import json
import itertools
import numpy as np


weather_options = ['Rainy','Overcast','Sunny']
temperature_options = ['Cool','Mild','Hot']
humidity_options = ['Normal','High']
wind_options = ['False','True']
play_options = ['No','Yes']

sum_options = ['Rainy','Overcast','Sunny','Cool','Mild','Hot','Normal','High','False','True','No','Yes']

def read_data(filename):
    
    with open(filename) as json_file:
        data = json.load(json_file)
        
    return data

def get_individual_support(optionsvar, data):
    
    ind_supp = {}
    
    for item in optionsvar:
        ind_supp[item] = 0
        for key in data:
            if item in data[key]:
                ind_supp[item] += 1
                
    return ind_supp

def get_individual_supports(data):
    
    weather_supp = get_individual_support(weather_options, data)
    temperature_supp = get_individual_support(temperature_options, data)
    humidity_supp = get_individual_support(humidity_options, data)
    wind_supp = get_individual_support(wind_options, data)
    play_supp = get_individual_support(play_options, data)
    
    return weather_supp, temperature_supp, humidity_supp, wind_supp, play_supp

def find_in_list(to_find, list_s):
    
    n = len(to_find)
    
    retvar = 0
    
    for i in range(n):
        if to_find[i] in list_s:
            retvar += 1
            
    retbool = retvar == n
    
    return retbool

def get_n_individual_supports(data):
    
    n = len(data['1'])
    
    returnvar = []
    
    for i in range(2,n):
        
        permut = list(itertools.combinations(sum_options, i))
        m = len(permut)
        permut_supp = np.zeros(m)
        
        for j in range(m):
            
            if len(set(list(permut[j]))-set(weather_options)) != 0 and len(set(list(permut[j]))-set(temperature_options)) != 0 and len(set(list(permut[j]))-set(humidity_options)) != 0 and\
               len(set(list(permut[j]))-set(wind_options)) != 0 and len(set(list(permut[j]))-set(play_options)) != 0:

                permutvar = 0
                
                for key in data:
                    if find_in_list(list(permut[j]), data[key]):
                        permutvar += 1
                        
                permut_supp[j] = permutvar
            
        finvar = [permut, permut_supp]
        returnvar.append(finvar)                
        
    return returnvar

def find_in_individualsupports(individual_supports, to_be_found):
    
    indexvar = individual_supports[len(to_be_found)-2][0].index(to_be_found)
    
    return individual_supports[len(to_be_found)-2][1][indexvar]