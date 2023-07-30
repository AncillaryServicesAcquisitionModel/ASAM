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


def doublescore_markup(direction='upward', of_quantity = None, asset = None, unit_test= None):
    """Method:
    - Provides a list of mark_ups for risks of double score on redispatch and intraday trade market/mechansim.
    - It returns a list of the mark-ups only (for later addition to other mark_ups).
    - The mark-up is only relevant for angent strategies where the same capacity is offered on two markets (IDM and RDM) simultanously.
    - A 'double-scoring' situation would lead to non-delivery, with consequences regarding imbalance cost and fuel cost.
    - Additional non-delivery penalties are not considered.
    - Please consult the ASAM documentation for mor information on this mark-up.
    - The method is structured as follows:
            1. Data collection
            2. risk quantity determination
            3. Risk price determination
            4. Mark-up determination

    Note:
    - Assumption: capacity is first placed on the IDM, because of coninous trading and instantanous clearing.
                  The capacity that is not (yet) cleared is subsequently also offered for redispatch.
    - Assumption: the risk quantity is determined with a uniform distribution assumption regarding
                  the quantity that is double-scored: #assumption uniform distribution
                  exp value = (minCalled + maxCalled)/2. With minimum called capacity =1 and max. called = offer quantity
    - Currently delivery duration == 1 (block orders) are implemented. No error is raised if delivery duration is > 1.
    - Positive values for costs are actual cost from agent perspective. Negative values are savings.
    - Offered orders are operational capacity orders. No check on asset commitment constraints in this method.

    """

    if not of_quantity:
        #if no quantity to be calculated
        return([],[])
    if all(v==0 for v in of_quantity):
        #only zero quantity
        return([0]*len(of_quantity))

    #COLLECTION OF DATA
    if not unit_test:
        srmc = asset.srmc #eur/mwh

        #get the open offered position from IDM
        buy_position, sell_position =self.model.IDM_obook.get_offered_position(associated_asset=asset.assetID)

        #deduct offered position from available capacity
        if direction == 'upward':
            if asset.schedule['available_up'].loc[
                         self.model.schedules_horizon.index].index.isin(sell_position.index).any():
                av_cap = asset.schedule['available_up'].loc[
                             self.model.schedules_horizon.index].fillna(0).to_frame().join(
                             sell_position).copy()

            else:
                #sell_position empty or outside schedule. No mark-up.
                markup_up = [0]*len(of_quantity)
                return(markup_up)
        elif direction == 'downward':
            if asset.schedule['available_down'].loc[
                         self.model.schedules_horizon.index].index.isin(buy_position.index).any():
                av_cap = asset.schedule['available_down'].loc[
                                 self.model.schedules_horizon.index].fillna(0).to_frame().join(
                                 buy_position).copy()
            else:
                #buy_position empty or outside schedule
                markup_down = [0]*len(of_quantity)
                return(markup_down)

        else:
            raise Exception('direction must be upward or downward')
        av_cap.rename(columns = {'quantity': 'other_market_position'}, inplace = True)
        #expected imbalance prices
        eIBP = self.model.rpt.eIBP.loc[self.model.schedules_horizon.index].copy()
        #get expected imbalance prices for the known DAP
        av_cap = pd.concat([av_cap,eIBP[['expected_IBP_short','expected_IBP_long']]], axis=1)
        av_cap['offer_quantity'] = of_quantity
        #this is an assumption of this method
        delivery_duration = 1
    else:
        #unit test input data of mark-up method
        direction #upward or downward
        of_quantity #list with MW values. Must have same length as av_cap
        asset = None # not needed, because for unit test asset value are provided seperately
        #unit_test is dictionary
        av_cap = unit_test['av_cap_input'] #df with available capacity and eIBP
        srmc = unit_test['asset_input']['srmc']
        #this is an assumption of this method without control
        delivery_duration = 1
        if len(of_quantity) != len(av_cap):
            raise Exception ('invalid unit test input')

    #RISK QUANTITY DETERMINATION
    #assumption uniform distribution, with minimum called capacity =1
    #exp value = (minCalled + maxCalled)/2
    min_IDM_capacity = 1 #MW
    av_cap['risk_quantity'] = (min_IDM_capacity + av_cap['other_market_position']).where(
            av_cap['other_market_position']> 0, 0)/2

    #in line with notation are short positions negative
    if direction =='upward':
        av_cap['risk_quantity'] = - av_cap['risk_quantity']


    #RISK PRICE DETERMINATION
    #In case of double-score mark-up, risk price is determined by imbalance price
    #(penalties are out of scope, but could be added here)
    av_cap['risk_price'] = 0
    mask= av_cap.loc[av_cap['risk_quantity'] !=0]['risk_quantity'] > 0
    av_cap['risk_price'].loc[av_cap['risk_quantity'] !=0] = (- av_cap['expected_IBP_long'].loc[
            av_cap['risk_quantity'] !=0] + srmc).where(mask,- av_cap['expected_IBP_short'].loc[
            av_cap['risk_quantity'] !=0] + srmc) #EUR/MWh

    #MARK-UP DETERMINATION
    av_cap['markup'] = av_cap['risk_price'].mul(av_cap['risk_quantity']).div(
            av_cap['offer_quantity']).replace(to_replace=np.inf, value=0)* delivery_duration/4 #EUR/MWh

    if direction == 'downward':
        #in line with notation markup for downward/buy orders is multiplied by -1, so that mark-up can be added to a price.
        av_cap['markup']= - av_cap['markup']

    av_cap['markup'].fillna(value= 0, inplace=True)
    av_cap['markup'] = av_cap['markup'].round().astype('int64').copy()
    markup_lst = list(av_cap['markup'])

    if not unit_test:
        return(markup_lst)
    else:
        #return also av_cap in case of unit test.
        return (markup_lst, av_cap)

def read_input_data(path,filename):

    simulation_parameters = pd.read_excel(os.path.join(path,filename), sheet_name=None)
    simulation_parameters['asset_input'] = pd.read_excel(os.path.join(path,filename), sheet_name='asset_input', index_col=0).squeeze()
    return (simulation_parameters)


#input directory
idir=r"C:\unit_tests"

#Files names of test scenarios
inames = ['unit_test_doublescore_markup_downwards.xlsx','unit_test_doublescore_markup_upwards.xlsx',
          'unit_test_doublescore_markup_downwards_2.xlsx','unit_test_doublescore_markup_upwards_2.xlsx']
test_results =DataFrame(
        index=inames, columns=['markups_result','expected_markups_output', 'number_unequal_result_parameters','unequal_results'])
joint_dict={}

for iname in inames:
    print(" -------------------------------" )
    print(" start test:                    ",iname)
    test_data = read_input_data(idir, iname)
    mark_ups, av_cap_result = doublescore_markup(
                   direction=test_data['asset_input']['direction'],
                   of_quantity = test_data['av_cap_input']['offer_quantity'].tolist(),
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










