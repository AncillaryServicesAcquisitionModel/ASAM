# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 17:17:56 2021
Unit_test ramping

Copy-paste the latest version of ramping_mark-up method in here.
Remove 'self' from def


@author: Otto
"""


import pandas as pd
import math
from pandas import Series, DataFrame
import numpy as np
from random import randrange, choice
import os



def ramping_markup(direction='upward', of_quantity = None, asset = None, unit_test=None):
        """Method: provides a list of mark_ups for risks of infeasible ramps.
        - It returns a list of the mark-ups (for later addition to other mark_ups).
        - Infeasible ramps lead to potential imbalance costs (or savings) and fuel costs (or savings).
        - Please consider the ASAM documentation for more explanation on the ramp mark-up.
        - The method is structured as follows:
            1. Data collection
            2. risk quantity determination
            3. Risk price determination
            4. Mark-up determination
        Note:
        - Currently delivery periods > 1 (block orders) are NOT implemented. No error is raised if delivery period is > 1.
        - Positive values for costs are actual cost from agent perspective. Negative values are savings.
        - Offered orders are operational capacity orders. No check on other asset constraints in this method.
        - Offer quantity are provided in list in size of model.schedules_horizon
        """

        if not of_quantity:
            #if no quantity to be calculated
            return([],[])
        if all(v==0 for v in of_quantity):
            #only zero quantity
            return([0]*len(of_quantity))

        if not unit_test:
            srmc = asset.srmc #EUR/MWh
            pmax = asset.pmax #MW
            ramp_limit_up = asset.ramp_limit_up * pmax #p.u. of pmax per MTU * MW
            ramp_limit_down = asset.ramp_limit_down * pmax

            #test if there are unfeasible ramps
            check_ramps = asset.schedule.loc[self.model.schedules_horizon.index].copy()
            check_ramps['offered']= of_quantity
            if direction == 'upward':
                check_ramps['delta_ramp'] =  check_ramps['rem_ramp_constr_avail_up'] - check_ramps['offered']
            elif direction == 'downward':
                check_ramps['delta_ramp'] = check_ramps['rem_ramp_constr_avail_down'] - check_ramps['offered']
            if not (check_ramps['delta_ramp'] < 0).any():
                #no unfeasible ramps. markup is 0.
                return([0]*len(of_quantity))

            #collect and join data
            #add get expected imbalance prices for the known day-ahead price to available capacity and dispatch schedule
            eIBP = self.model.rpt.eIBP.loc[self.model.schedules_horizon.index].copy()
            av_cap = asset.schedule.loc[self.model.schedules_horizon.index].copy()
            av_cap = pd.concat([av_cap,eIBP[['expected_IBP_short','expected_IBP_long']]], axis=1)
            av_cap.reset_index(inplace=True)
        else:
            #unit test input data of mark-up method
            direction #upward or downward
            of_quantity #list with MW values. Must have same length as av_cap
            asset = None # not needed, because for unit test asset value are provided seperately
            #unit_test is dictionary
            av_cap = unit_test['av_cap_input'] #df with available capacity and eIBP
            srmc = unit_test['asset_input']['srmc']
            ramp_limit_up = unit_test['asset_input']['ramp_limit_up'] #MW per MTU
            ramp_limit_down = unit_test['asset_input']['ramp_limit_down'] #MW per MTU
            if len(of_quantity) != len(av_cap):
                raise Exception ('invalid unit test input')

        #mark-up calculation per offer
        markup_lst = []
        for b in range(len(of_quantity)):
            #check if this quantity has a unfeasible ramp
            if not unit_test:
                if check_ramps['delta_ramp'].iloc[b] >= 0:
                    #feasible -> no mark-up
                    markup_lst += [0]
                    continue
            if of_quantity[b] == 0:
                markup_lst += [0]
                continue

            #RISK quantity DETERMINATION
            #risk quantity in case of ramping is
            #   required required pre-ramp + required post ramp minus scheduled dispatch

            #required ramp to dispatch the offered quantity
            av_cap['required_ramp']=0


            if direction == 'upward':
                #new asset dispatch value to deliver offer quantity
                new_commit = of_quantity[b] + av_cap['commit'].iloc[b].copy()
            elif direction == 'downward':
                new_commit = av_cap['commit'].iloc[b].copy() - of_quantity[b]
            #this is an assumption of this method without control
            delivery_duration = 1
            #mtu of start delivery period
            t_delivery_start = b  #t < start delivery period Ts
            t_delivery_end = b + delivery_duration #t > end delivery period Te
            #ramp (dispatch value list) before delivery MTU
            pre_ramp=[]
            #ramp (dispatch value list) after delivery MTU
            post_ramp=[]
            success= False
            i = 1
            #CALCULATE PRE-RAMP
            while success == False:
                if t_delivery_start - i < 0:
                    #out of schedule_horizon ramps are ignored
                    break
                if direction == 'upward':
                    max_ramp_to_delMTU = new_commit - i * ramp_limit_up
                    if max_ramp_to_delMTU > av_cap['commit'].iloc[t_delivery_start - i]:
                        pre_ramp +=[max_ramp_to_delMTU]
                    else:
                        pre_ramp +=[av_cap['commit'].iloc[t_delivery_start - i]]
                        success = True
                elif direction == 'downward':
                    max_ramp_to_delMTU = new_commit + i * ramp_limit_down
                    if max_ramp_to_delMTU < av_cap['commit'].iloc[t_delivery_start - i]:
                        pre_ramp +=[max_ramp_to_delMTU]
                    else:
                        pre_ramp +=[av_cap['commit'].iloc[t_delivery_start - i]]
                        success = True
                i += 1
            pre_ramp = list(reversed(pre_ramp)).copy()

            #CALCULATE POST-RAMP
            i = 1
            success = False
            while success == False:
                if t_delivery_end + i > len(av_cap) -1:
                    #out of horizon_schedule ramps are ignored
                    break
                if direction == 'upward':
                    max_ramp_to_delMTU = new_commit - i * ramp_limit_down
                    if max_ramp_to_delMTU > av_cap['commit'].iloc[t_delivery_end + i - 1]:
                        post_ramp +=[max_ramp_to_delMTU]
                    else:
                        post_ramp += [av_cap['commit'].iloc[t_delivery_end + i - 1]]
                        success = True
                elif direction == 'downward':
                    max_ramp_to_delMTU = new_commit + i * ramp_limit_up
                    if max_ramp_to_delMTU < av_cap['commit'].iloc[t_delivery_end + i -1]:
                        post_ramp +=[max_ramp_to_delMTU]
                    else:
                        post_ramp += [av_cap['commit'].iloc[t_delivery_end + i -1 ]]
                        success = True
                i += 1

            #add pre and post ramp to av_cap
            if pre_ramp:
                av_cap['required_ramp'].iloc[t_delivery_start - len(pre_ramp):t_delivery_start ] = pre_ramp
            if post_ramp:
                av_cap['required_ramp'].iloc[t_delivery_end: t_delivery_end + len(post_ramp)]= post_ramp

            mask = av_cap['required_ramp'] ==0
            av_cap['risk_quantity'] = av_cap['required_ramp'].where(mask,av_cap['required_ramp'].sub(av_cap['commit'],  fill_value=0))


            #RISK PRICE DETERMINATION
            #In case of ramping, the risk price is determined by srmc (fuel costs/savings and imbalance price (for non-delivery).
            av_cap['risk_price'] = 0
            mask= av_cap.loc[av_cap['risk_quantity'] !=0]['risk_quantity'] > 0
            av_cap['risk_price'].loc[av_cap['risk_quantity'] !=0] = (- av_cap['expected_IBP_long'].loc[
                    av_cap['risk_quantity'] !=0] + srmc).where(mask,- av_cap['expected_IBP_short'].loc[
                    av_cap['risk_quantity'] !=0] + srmc) #EUR/MWh

            #MARK-UP DETERMINATION
            markup = int(round(av_cap['risk_price'].mul(av_cap['risk_quantity']).sum() /(of_quantity[b]) * delivery_duration/4,0)) #EUR/MWh

            if direction == 'downward':
                #in line with notation markup for downward/buy orders is multiplied by -1, so that mark-up can be added to a price.
                markup= - markup

            #add markup of offer to list
            markup_lst += [markup]

        if not unit_test:
            return(markup_lst)
        else:
            #return also av_cap in case of unit test.
            return (markup_lst, av_cap)




def read_input_data(path,filename):

    simulation_parameters = pd.read_excel(os.path.join(path,filename), sheetname=None)
    simulation_parameters['asset_input'] = pd.read_excel(os.path.join(path,filename), sheetname='asset_input', index_col=0).squeeze()
    return (simulation_parameters)


#input directory
idir=r"C:\unit_tests"
idir=r"C:\Users\Otto\Documents\Doctor\python\RedispatchModel1.0_datacollector\unit_tests"

#Files names of test scenarios
inames = ['unit_test_ramp_markup_upwards.xlsx',
          'unit_test_ramp_markup_downwards.xlsx',
          'unit_test_ramp_markup_upwards_2.xlsx',
          'unit_test_ramp_markup_downwards_2.xlsx',
          'unit_test_ramp_markup_upwards_3.xlsx',
          'unit_test_ramp_markup_downwards_3.xlsx',
          'unit_test_ramp_markup_upwards_4.xlsx',
          'unit_test_ramp_markup_downwards_4.xlsx',
          'unit_test_ramp_markup_upwards_5.xlsx',
          'unit_test_ramp_markup_downwards_5.xlsx']
test_results =DataFrame(
        index=inames, columns=['markups_result','expected_markups_output', 'number_unequal_result_parameters','unequal_results'])
joint_dict={}

for iname in inames:
    print(" -------------------------------" )
    print(" start test:                    ",iname)
    test_data = read_input_data(idir, iname)
    mark_ups, av_cap_result = ramping_markup(
                   direction=test_data['asset_input']['direction'],
                   of_quantity = test_data['av_cap_input']['offered_quantity'].tolist(),
                   asset = None,
                   unit_test = {key: test_data[key] for key in ['asset_input','av_cap_input']})

    comp_names = ['result','expected_result']
    joint_dict[iname] = pd.concat([av_cap_result, test_data['expected_av_cap_output']],axis=1,keys=comp_names).swaplevel(
            0,1,axis=1).sort_index(axis=1, ascending=True,level=[0,1])

    #compare results with expected results.
    #The following lists and dictionaries are overwritten in every comparisson loop
    all_params=[]
    params_equal=[]
    unequal={}
    all_av_cap_results={}
    idx = pd.IndexSlice
    #comparing each parameter of all dataframes of the results (is equivalent?)
    for result in joint_dict:
        params=list(joint_dict[result].columns.get_level_values(0).unique())
        for param in params:
            eq=joint_dict[result].loc[:,idx[param,comp_names[0]]].equals(joint_dict[result].loc[:,idx[param,comp_names[1]]])
            all_params+=[(result,param)]
            params_equal+=[eq]
    #Dataframe storing per parameter if results (test and baseline) are equivalent or not
    comparisson=DataFrame(list(zip(all_params,params_equal)), columns=['parameter', 'is_equivalent'])
    for i in range(len(comparisson.loc[comparisson['is_equivalent']==False])):
        param=comparisson.loc[comparisson['is_equivalent']==False]['parameter'].iloc[i]
        unequal[param]=joint_dict[param[0]].loc[:, idx[param[1],:]]



    #store test results
    test_results.loc[iname, 'markups_result']=[x for x in mark_ups if x !=0][0]
    test_results.loc[iname, 'expected_markups_output']= test_data['expected_markup'].loc[0,'value']
    test_results.loc[iname,'number_unequal_result_parameters']=len(unequal.copy())
    test_results.loc[iname,'unequal_results']=[unequal.copy()]

    if len(unequal)==0:
        print('The test results are equivalent')
    else:
        print('The following parameters of the test are not the same:')
        for key in unequal.keys():
            print(key)










