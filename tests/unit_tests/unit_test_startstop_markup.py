# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 17:17:56 2021
Unit_test startstop

Copy-paste the latest version of startstop_mark-up method in here.
Remove 'self' from def

price_start_lst, start_markup = self.startstop_markup(
                                    direction = 'upward', of_quantity = startblocks,
                                    asset = a, gct = gate_closure_MTU, partial_call=True)
@author: Samuel
"""


import pandas as pd
import math
from pandas import Series, DataFrame
import numpy as np
from random import randrange, choice
import os
from ast import literal_eval



def startstop_markup (direction='upward', of_quantity = None,
                          asset = None, gct = None, partial_call = False,
                          order_granularity= 1, minimum_call = 1, unit_test=None):
        """
        Method: provides a list of mark_ups for risks of orders from startup or shut down capacity.
        Optional a mark-up can be added regarding the risk of partial call (for types limit orders).
        - It returns a list of the mark-ups (for later addition to other mark_ups).
        - start-up and shut down of assets lead to potential imbalance costs (or savings) and fuel costs (or savings),
          sfixed start-up/shut-down costs costs, and in case of overlapping with scheduled start and stop ramps,
          it can lead to savings of fixed start-up/shut-down costs.
        - Please consider the ASAM documentation for more explanation on the ramp mark-up.
        - The method is structured as follows:
            1. Data collection
            2. risk quantity determination
            3. Risk price determination
            4. Mark-up determination


        Note:
        - Assumed  input is nested list [[day],[time],[duration]] of block orders,
          whereby the offer quantity is equal to pmin.
        - No opportunity costs included.

        - Positive values for costs are actual cost from agent perspective. Negative values are savings.
        - Notation regarding imbalance: When the expected respective imbalance price is positive,
          short agents pay, long agents receive. In all balancing control states.
        - The method enables imbalanace design with dual pricing, as imbalance prices for short and long are provided.
          However, they may be the same (single-pricing situations).

        - If partial_call is True, the start-stop mark-up also contains a partial call mark-up
        - Partial call risk considers missing fixed start stop cost and additional imbalance.
        - Order granularity (MW) and minimum call (MW) determine the considered partial call risk quantity
        - Assumed probability for various partial call scenario's is a uniform discrete distribution.
        - Furthermore it is assumed that order granularity and minimum call, as well as offered
          quantity are natural numbers.
        """
        if not of_quantity[0]:
            #if no start stop blocks are avaiable empty lists are returned
            return([])

        #COLLECTION AND JOINING OF DATA
        if not unit_test:
            srmc = asset.srmc #eur/mwh
            pmax = asset.pmax #MW
            pmin = asset.pmin #MW
            ramp_limit_start_up = asset.ramp_limit_start_up # p.u. pmax per ISP
            ramp_limit_shut_down = asset.ramp_limit_shut_down # p.u. pmax per ISP
            start_up_cost = asset.start_up_cost #eur
            shut_down_cost =  asset.shut_down_cost
            min_down_time = asset.min_down_time #ISPs
            min_up_time = asset.min_up_time

            #expected imbalance prices
            eIBP = self.model.rpt.eIBP.loc[self.model.schedules_horizon.index[gct:]]
            av_cap = asset.schedule.loc[self.model.schedules_horizon.index[gct:]].copy()

            if ((av_cap['commit'] < pmin )&(av_cap['commit']>0)).any():
                import pdb
                pdb.set_trace()
                raise Exception('this method works only correctly in the absence of scheduled dispatch >pmin or 0')
            if pmin < 1:
                #this method does'nt work and makes no sense
                return([],[])

            #get expected imbalance prices for the known DAP
            av_cap = pd.concat([av_cap,eIBP[['expected_IBP_short','expected_IBP_long']]], axis=1)
            av_cap.reset_index(inplace=True)
        else:
            #unit test input data of mark-up method
            direction #upward or downward
            of_quantity #nested list with start or stop blocks [[delday][deltime][duration]]
            asset = None # not needed, because for unit test asset value are provided seperately
            #unit_test is dictionary
            av_cap = unit_test['av_cap_input'] #df with available capacity and eIBP
            srmc = unit_test['asset_input']['srmc']
            ramp_limit_start_up = unit_test['asset_input']['ramp_limit_start_up'] # p.u. pmax per ISP
            ramp_limit_shut_down =  unit_test['asset_input']['ramp_limit_shut_down']# p.u. pmax per ISP
            pmax = unit_test['asset_input']['pmax'] #MW
            pmin = unit_test['asset_input']['pmin'] #MW
            start_up_cost = unit_test['asset_input']['start_up_cost'] #eur
            shut_down_cost = unit_test['asset_input']['shut_down_cost']
            min_down_time = unit_test['asset_input']['min_down_time'] #ISPs
            min_up_time = unit_test['asset_input']['min_up_time']


        markup_lst = []

        #mark-up calculation per offer
        for b in range(len(of_quantity[0])):
            #risk quantity and columns nan
            av_cap['risk_quantity_fuel'] = np.nan
            av_cap['risk_quantity_imbalance'] = np.nan
            av_cap['risk_price_imbalance'] = np.nan
            #lists to capture the risk quantity ramp
            pre_ramp=[]
            post_ramp=[]
            save_post_overlap_ramp = False
            save_pre_overlap_ramp =False
            #read order MTU and duration
            delivery_day = of_quantity[0][b]
            delivery_mtu = of_quantity[1][b]
            delivery_duration = of_quantity[2][b]
            start_pre_ramp = 1
            #t_delivery_start
            t_delivery_start = av_cap.loc[(av_cap['delivery_day'] == delivery_day)&(
                    av_cap['delivery_time'] == delivery_mtu)].index[0]

            if direction == 'upward':
                #minimum duration the asset needs to run
                min_duration = min_up_time
                #end MTU of asset delivery (and start of shut-down ramp)
                t_delivery_end = t_delivery_start + max(delivery_duration, min_duration)
                if t_delivery_end > len(av_cap):
                    #min_duration reduced to fit horizon
                    t_delivery_end = len(av_cap)
                    min_duration = min_duration - (t_delivery_start + max(delivery_duration, min_duration) - len (av_cap))
                #number of mtu to startup
                startup_duration = int(pmin/(pmax * ramp_limit_start_up))
                #number of mtu to shut down
                shutdown_duration =int(pmin/(pmax * ramp_limit_shut_down))
            elif direction == 'downward':
                min_duration = min_down_time
                t_delivery_end = t_delivery_start + max(delivery_duration, min_duration)
                if t_delivery_end > len(av_cap):
                    #min_duration ignored
                    t_delivery_end = len(av_cap)
                    min_duration = min_duration - (t_delivery_start + max(delivery_duration, min_duration) - len (av_cap))
                startup_duration = int(pmin/(pmax *ramp_limit_shut_down))
                shutdown_duration =int(pmin/(pmax *ramp_limit_start_up))
            #t_pre_overlap_start
            t_pre_overlap_start = t_delivery_start - startup_duration - shutdown_duration
            #t_post_overlap_end
            t_post_overlap_end = t_delivery_end + startup_duration+ shutdown_duration

            #additional_minrun_duration
            extra_post_duration =  (min_duration - delivery_duration)

            #ensure that out of schedule_horizon startstop times are ignored
            if t_pre_overlap_start < 0:
                t_pre_overlap_start = 0
            if t_post_overlap_end > len(av_cap):
                t_post_overlap_end = len(av_cap)


            #RISK QUANTITY DETERMINION
            if direction == 'upward':
                #adjust start and stop ramps if beyond schedules horizon.
                if t_delivery_start - startup_duration < 0:
                    start_pre_ramp = startup_duration- t_delivery_start
                if t_delivery_end + shutdown_duration > len(av_cap):
                    shutdown_duration = len(av_cap) - t_delivery_end +1

                scheduled_commitment = 0
                offer_dispatch = pmin
                is_pre_overlap = (av_cap['commit'].iloc[t_pre_overlap_start:t_delivery_start] >=offer_dispatch)
                is_post_overlap =(av_cap['commit'].iloc[t_delivery_end -extra_post_duration: t_post_overlap_end + 1]  >=offer_dispatch)
                #start ramp to delivery the offer minus the scheduled unit commitment
                pre_ramp=[- scheduled_commitment + ramp_limit_start_up * pmax * t for t in range(start_pre_ramp, startup_duration)]
                post_ramp=[- scheduled_commitment + pmin -ramp_limit_shut_down * pmax * t for t in range(1,shutdown_duration)]
                #assumed scheduled ramp (post) from a scheduled start-up before delivery period.
                pre_overlap_ramp = [ramp_limit_shut_down * pmax * t for t in range(1,startup_duration)]
                #assumed scheduled ramp (pre) from a scheduled start-up after delivery period.
                post_overlap_ramp= [pmin - ramp_limit_start_up * pmax * t for t in range(1,shutdown_duration)]

            if direction == 'downward':
                #adjust start and stop ramps if beyond schedules horizon.
                if t_delivery_start - shutdown_duration < 0:
                    start_pre_ramp = shutdown_duration - t_delivery_start
                if t_delivery_end + startup_duration > len(av_cap):
                    startup_duration = len(av_cap) - t_delivery_end +1

                scheduled_commitment = pmin
                offer_dispatch = 0
                is_pre_overlap = (av_cap['commit'].iloc[t_pre_overlap_start:t_delivery_start] == offer_dispatch)
                is_post_overlap =(av_cap['commit'].iloc[t_delivery_end-extra_post_duration: t_post_overlap_end + 1] == offer_dispatch)
                pre_ramp=[- scheduled_commitment + pmin - ramp_limit_shut_down* pmax * t for t in range(start_pre_ramp, shutdown_duration)]
                post_ramp=[- scheduled_commitment + ramp_limit_start_up * pmax * t for t in range(1,startup_duration)]
                #assumed scheduled ramp (post) from a scheduled shut-down before delivery period.
                pre_overlap_ramp= [pmin - ramp_limit_start_up * pmax * t for t in range(1,shutdown_duration)]
                #assumed scheduled ramp (pre) from a scheduled start-up after delivery period.
                post_overlap_ramp = [ramp_limit_shut_down * pmax * t for t in range(1,startup_duration)]

            if is_pre_overlap.any() :
                #overlap with another scheduled start-stop leads to saving of start-up costs
                save_pre_overlap_ramp = True
                pre_overlap_start =(av_cap.iloc[t_pre_overlap_start:t_delivery_start]).loc[is_pre_overlap].index[-1] +1
                av_cap['risk_quantity_fuel'].iloc[pre_overlap_start :t_delivery_start
                      ] = offer_dispatch - scheduled_commitment
                extension = (t_delivery_start -pre_overlap_start)-len(pre_overlap_ramp)
                if extension  > 0:
                    av_cap['risk_quantity_imbalance'].iloc[pre_overlap_start :t_delivery_start
                          ] = [offer_dispatch - sr for sr in  pre_overlap_ramp + [scheduled_commitment] * extension]
                else:
                    av_cap['risk_quantity_imbalance'].iloc[pre_overlap_start :t_delivery_start
                          ] = [offer_dispatch - sr for sr in  pre_overlap_ramp[:extension]]
            else: #no overlapping ramps
                save_pre_overlap_ramp = False
                if not av_cap.iloc[t_delivery_start-len(pre_ramp) :t_delivery_start].empty:
                    av_cap['risk_quantity_fuel'].iloc[t_delivery_start-len(pre_ramp) :t_delivery_start
                          ] = pre_ramp
                av_cap['risk_quantity_imbalance'] = av_cap['risk_quantity_fuel']


            if is_post_overlap.any():
                save_post_overlap_ramp = True
                post_overlap_start =av_cap.iloc[t_delivery_end-extra_post_duration: t_post_overlap_end + 1].loc[is_post_overlap].index[0]
                av_cap['risk_quantity_fuel'].iloc[t_delivery_end-extra_post_duration: post_overlap_start
                      ]= offer_dispatch - scheduled_commitment
                extension = (post_overlap_start - (t_delivery_end-extra_post_duration))-len(post_overlap_ramp)
                if extension  > 0:
                    av_cap['risk_quantity_imbalance'].iloc[t_delivery_end-extra_post_duration: post_overlap_start
                          ] = [offer_dispatch - sr for sr in  extension * [scheduled_commitment] + post_overlap_ramp]
                else:
                    av_cap['risk_quantity_imbalance'].iloc[t_delivery_end-extra_post_duration: post_overlap_start
                          ] = [offer_dispatch - sr for sr in  post_overlap_ramp[-extension:]]
            else: #no overlapping ramps
                save_post_overlap_ramp = False
                if not av_cap.iloc[t_delivery_end - extra_post_duration: t_delivery_end + len(post_ramp)].empty:
                    av_cap['risk_quantity_fuel'].iloc[t_delivery_end - extra_post_duration: t_delivery_end + len(post_ramp)]= (
                            extra_post_duration)* [offer_dispatch - scheduled_commitment]+ post_ramp
                av_cap['risk_quantity_imbalance'] = av_cap['risk_quantity_fuel']


            #RISK PRICE DETERMINATION
            mask= av_cap.loc[av_cap['risk_quantity_imbalance'].notnull()]['risk_quantity_imbalance'] > 0
            av_cap['risk_price_imbalance'].loc[av_cap['risk_quantity_imbalance'].notnull()] = (- av_cap['expected_IBP_long'].loc[
                    av_cap['risk_quantity_imbalance'].notnull()]).where(mask,- av_cap['expected_IBP_short'].loc[
                    av_cap['risk_quantity_imbalance'].notnull()]) #EUR/MWh

            #risk_price_fuel =  srmc
            #fuel risk cost
            additional_fuel_cost = (av_cap['risk_quantity_fuel'] * srmc/4).fillna(value = 0).sum() #EUR
            imbalance_risk_cost =  av_cap['risk_price_imbalance'].mul(av_cap['risk_quantity_imbalance']).fillna(value = 0).sum() #EUR


            # check if start stop cost are saved
            if direction == 'upward':
                if save_pre_overlap_ramp == True:
                        saved_pre_overlap_cost = -shut_down_cost #EUR
                        start_up =0 #EUR
                else:
                    saved_pre_overlap_cost = 0
                    start_up = start_up_cost
                if save_post_overlap_ramp == True:
                    saved_post_overlap_cost = -start_up_cost
                    shut_down =0
                else:
                    saved_post_overlap_cost = 0
                    shut_down = shut_down_cost
            elif direction == 'downward':
                if save_pre_overlap_ramp == True:
                    saved_pre_overlap_cost = -start_up_cost
                    shut_down =0
                else:
                    saved_pre_overlap_cost = 0
                    shut_down = shut_down_cost
                if save_post_overlap_ramp == True:
                    saved_post_overlap_cost = -shut_down_cost
                    start_up =0
                else:
                    saved_post_overlap_cost = 0
                    start_up = start_up_cost

            total_cost = start_up + shut_down + imbalance_risk_cost + additional_fuel_cost + saved_pre_overlap_cost + saved_post_overlap_cost
            markup = int(round(total_cost/(pmin * delivery_duration/4),0)) #EUR/MWh


            #PARTIAL CALL MARKUP
            if partial_call == True:
                #risk markup of partial activation 0< > pmin is included
                #assumed probability of partial call is a uniform discrete distribution
                if ((type(pmin) is not np.int32)&(type(pmin) is not np.int64)&(type(pmin) is not int))|(
                        type(minimum_call) is not int)|(type(order_granularity) is not int):
                    raise Exception (
                            "partial call works with a assumption of natural numbers and therefore needs positive integers for pmin,minimum_call and order_granularity")
                #expected value for partial call
                ePC = (minimum_call + pmin)/2
                #constant offer quantity of pmin is assumed for start-stop orders (block order type)
                risk_quantity_pc = pmin -ePC

                #mean expected imbalance price during delivery period
                if direction == 'upward':
                    #assumption in case of partial upward call, dispatch needs to be adjusted to pmin.
                    #market party has long position.
                    #IBP long needs to be reversed, as positive prices means less cost (and risk_quantity_pc is > 0)
                    mean_eibp = - av_cap[['expected_IBP_long']].iloc[t_delivery_start:t_delivery_end].mean().round().values[0]
                elif direction == 'downward':
                    #assumption in case of partial downward call, dispatch needs to be adjusted to 0.
                    #market party gets short position.
                    #IBP short is not reversed as positive prices multiplied with positive risk quantity increase costs
                    mean_eibp =av_cap[['expected_IBP_short']].iloc[t_delivery_start:t_delivery_end].mean().round().values[0]

                risk_price_pc = (markup +  mean_eibp)
                markup_pc = int(round(risk_price_pc * risk_quantity_pc/ pmin))
                #add partial call mark-up to start stop mark-up
                markup = markup + markup_pc

            if direction == 'downward':
                #in line with notation markup for downward/buy orders is multiplied by -1,
                #so that the mark-up can be added to an offer price.
                markup = -markup

            #add mark-up for this order to list
            markup_lst += [markup]

        if not unit_test:
            return(markup_lst)
        else:
            #return also av_cap in case of unit test.
            return (markup_lst, av_cap, Series({'additional_fuel_cost':additional_fuel_cost,
                                                'imbalance_risk_cost':imbalance_risk_cost,
                                                'start_up':start_up,'shut_down':shut_down,
                                                'saved_pre_overlap_cost':saved_pre_overlap_cost,
                                                'saved_post_overlap_cost':saved_post_overlap_cost,
                                                'total_cost':total_cost,'markup':markup}).astype('int64'))

def read_input_data(path,filename):

    simulation_parameters = pd.read_excel(os.path.join(path,filename), sheet_name=None)
    simulation_parameters['asset_input'] = pd.read_excel(os.path.join(path,filename), sheet_name='asset_input', index_col=0).squeeze()
    simulation_parameters['expected_markup'] = pd.read_excel(os.path.join(path,filename), sheet_name='expected_markup', index_col=0).squeeze()

    return (simulation_parameters)


#input directory
idir=r"C:\unit_tests"

#Files names of test scenarios
inames = ['unit_test_startstop_markup_downwards.xlsx','unit_test_startstop_markup_upwards.xlsx',
          'unit_test_startstop_markup_downwards_2.xlsx','unit_test_startstop_markup_upwards_2.xlsx',
          'unit_test_startstop_markup_downwards_3.xlsx','unit_test_startstop_markup_upwards_3.xlsx',
          'unit_test_startstop_markup_downwards_4.xlsx','unit_test_startstop_markup_upwards_4.xlsx',
          'unit_test_startstop_markup_downwards_5.xlsx','unit_test_startstop_markup_upwards_5.xlsx']


test_results =DataFrame(
        index=inames, columns=['markups_result','expected_markups_output',
                               'markup_consequence_for_offer_price',
                               'number_unequal_result_parameters','unequal_results'])

av_cap_dict={}
for iname in inames:
    print(" -------------------------------" )
    print(" start test:                    ",iname)
    test_data = read_input_data(idir, iname)

    mark_ups, av_cap_result, costs = startstop_markup(
                   direction=test_data['asset_input']['direction'],
                   of_quantity = literal_eval(test_data['asset_input']['of_quantity']),
                   asset = None,
                   partial_call = literal_eval(test_data['asset_input']['partial_call']),
                   unit_test = {key: test_data[key] for key in ['asset_input','av_cap_input']})


    costs.name = 'value'
    costs.index.name ='parameter'
    joint_dict={}
    #compare results with expected results.
    comp_names = ['cost_result', 'expected_cost_result']
    joint_dict[iname+'_costs'] = pd.concat([ costs.to_frame(), test_data['expected_markup'].to_frame()],axis=1,keys=comp_names).swaplevel(
            0,1,axis=1).sort_index(axis=1, ascending=True,level=[0,1])

    #The following lists and dictionaries are overwritten in every comparisson loop
    all_params=[]
    params_equal=[]
    unequal={}
    idx = pd.IndexSlice
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
    test_results.loc[iname, 'markups_result']=mark_ups[0] #unit-test assumes that only one order mark-up is calculated
    test_results.loc[iname, 'expected_markups_output']= test_data['expected_markup']['markup']
    test_results.loc[iname,'number_unequal_result_parameters']=len(unequal.copy())
    test_results.loc[iname,'unequal_results']=[unequal.copy()]
    if (mark_ups[0] > 0) & (test_data['asset_input']['direction'] == 'upward'):
        test_results.loc[iname,'markup_consequence_for_offer_price'] =  'offer becomes more expensive'
    elif (mark_ups[0] < 0) & (test_data['asset_input']['direction'] == 'upward'):
        test_results.loc[iname,'markup_consequence_for_offer_price'] =  'offer becomes less expensive'
    elif (mark_ups[0] > 0) & (test_data['asset_input']['direction'] == 'downward'):
        test_results.loc[iname,'markup_consequence_for_offer_price'] =  'offer becomes less expensive'
    elif (mark_ups[0] < 0) & (test_data['asset_input']['direction'] == 'downward'):
        test_results.loc[iname,'markup_consequence_for_offer_price'] =  'offer becomes more expensive'
    elif mark_ups[0] == 0:
        test_results.loc[iname,'markup_consequence_for_offer_price'] =  'offer price does not change'
    if len(unequal)==0:
        print('The test results are equivalent')
    else:
        print('The following parameters of the test are not the same:')
        for key in unequal.keys():
            print(key)
    av_cap_dict [iname] = av_cap_result









