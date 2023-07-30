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
from ast import literal_eval


def opportunity_markup(direction='upward', of_quantity = None, asset = None,
                           success_assumption=None, MTU_of_h_consideration=False, unit_test= None):
    """
    Method:
    - Returns a list of price mark-up for opportunity costs with length of the schedule horizon
    - Assumption that market time unit of the quantity is 15 minutes.
      Meaning the order quantity is devided by 4 to get the EUR/MWh.
    - This opporunity mark-up method considers an imbalance market as opportunity to trade the offered capacity.
      (alternative methods could use the intra-day market as determining for opportunity)
    - Opportunity costs are obtained from input dataframe with pre-calculated opportunity costs
      depending on srmc, day-ahead price and (optionally) MTU of an hour (MTU_of_h_consideration)
      Please consult the documentation for more information.
    - The method is structured as follows:
            1. Data collection
            2. risk quantity determination
            3. Risk price determination
            4. Mark-up determination

    Note:
    - only order duration == 1 MTU (15 minutes) is considered.
    - When MTU_of_h_consideration==True, the method makes a distinction regarding the MTU of the hour.
      For this option the input data (self.model.exodata.opportunity_costs_db) must be delivered accordingly
      with a column 'PTU_of_an_hour' element {1,2,3,4}"""

    #DATA COLLECTION
    if not unit_test:
        #day-ahead prices needed to estimate imbalance prices and then opportunity costs
        if self.model.rpt.prices['DAP'].isnull().all():
            print('no day-ahead prices available. Possibly because DAM is not run.')
            print('Default DA price of 30 EUR/MWh used')
            DAP= 30
            MTU =list(self.model.schedules_horizon.index.get_level_values(1))
            DAP= [30] * len(MTU)
        else:
            DAP = self.model.rpt.prices['DAP'].loc[self.model.schedules_horizon.index].copy()
            MTU = list(DAP.index.get_level_values(1))
            DAP =DAP.tolist()

        if MTU_of_h_consideration==True:
            #MTU of hour list needed to get the specific opportunity cost distribution function
            MTU_of_h = []
            for mtu in MTU:
                mtu_of_h = mtu%4
                if mtu_of_h == 0:
                    mtu_of_h=4
                MTU_of_h += [mtu_of_h]
        if direction == 'upward':
            ramp= asset.ramp_limit_up * asset.pmax #maximum MW difference from one mtu to next.
        elif direction == 'downward':
            ramp= asset.ramp_limit_down * asset.pmax #maximum MW difference from one mtu to next.
        srmc = asset.srmc
        #exogenous opportunity prices dataframe
        odf = self.model.exodata.opportunity_costs_db
    else:
        #unit test input data of mark-up method
        DAP = unit_test['av_cap_input']['DAP'].tolist()
        MTU = unit_test['av_cap_input']['delivery_time'].tolist()
        ramp = unit_test['asset_input']['ramp']
        srmc = unit_test['asset_input']['srmc']
        odf = unit_test['opportunity_costs_input']


    #RISK QUANTITY DETERMINATION
    #the risk quantity is determined by the assumed quantity that the asset may trade on alternative markets.
    #the assumption is provided externally

    df = DataFrame()
    df['offered_quantity'] = of_quantity #MW per mtu
    #max average MW per mtu that can be delivered when order is activated within delivery mtu (linear ramp assumption)
    df['max_ramp_direct_activation'] = ramp/2
    #max average MW per mtu that can be delivered when order is activated during the mtu before delivery mtu (linear ramp assumption)
    df['max_ramp_mtu-1_activation'] = ramp

    if success_assumption=='max_ramp_direct_activation':
        #risk quantity is the assumed rewarded quantity
        df['risk_quantity'] =df[
                'max_ramp_direct_activation'].where(df[
                        'max_ramp_direct_activation']<df['offered_quantity'],df['offered_quantity'])
    elif success_assumption=='offered_quantity':
        df['risk_quantity'] =df['offered_quantity']

    elif success_assumption=='max_ramp_mtu-1_activation':
        df['risk_quantity'] =df[
                'max_ramp_mtu-1_activation'].where(df[
                        'max_ramp_mtu-1_activation']<df['offered_quantity'],df['offered_quantity'])


    #RISK PRICE DETERMINATION
    if direction == 'upward':
        #For sell orders, the opportunity is that part of
        #   the upward capacity, that is rewarded with the imbalance price for long
        #   because additional production/less consumption leads to a long imbalance position
        IBP = 'IB_price_long'
        direction_factor = 1
    elif direction == 'downward':
        IBP = 'IB_price_short'
        #The opportunity mark_up is deducted from srmc for the downward (buy) orders
        direction_factor = -1
    else:
        raise Exception ('direction not known, must be upwarda or downwards')

    #pre-calculated opportunity costs
    #depending on srmc, day-ahead price and (optionally) MTU of an hour (MTU_of_h_consideration)
    intrinsic_values = []
    #intrinsic value of available asset capacity (Eur/MWh)
    for p in range(len(DAP)):
        #K-value in odf is the asset srmc.
        #When asset.srmc are higher than highest K-value in odf, the max is used.
        if srmc > odf['K-value'].max():
            K = odf['K-value'].max()
        elif srmc < odf['K-value'].min():
            K = odf['K-value'].min()
        else:
            K = srmc
        try:
            if MTU_of_h_consideration==True:
                value =odf.loc[(odf['price_data']==IBP) & (odf['PTU_of_an_hour']==MTU_of_h[p]) & (
                        odf['DAP_left_bin']<=DAP[p]) & (odf['DAP_right_bin(excl)'] >DAP[p] ) & (
                                odf['K-value'] == K),'Opp_costs_for_K'].iloc[0]
            else:
                value =odf.loc[(odf['price_data']==IBP) & (
                        odf['DAP_left_bin']<=DAP[p]) & (odf['DAP_right_bin(excl)'] >DAP[p] ) & (
                                odf['K-value'] == K),'Opp_costs_for_K'].iloc[0]
        except:
            import pdb
            pdb.set_trace()
        intrinsic_values += [value]
    df['intrinsic_values'] = intrinsic_values #EUR/MWh


    #calculate opporunity cost (EUR!)
    df['opp_cost']=df['risk_quantity'] * df['intrinsic_values']/4
    #calculate the opportunity cost markup (EUR/MWh) for the offered quantity
    df['markup'] = df['opp_cost']/df['offered_quantity']

    if direction == 'downward':
        #in line with notation markup for downward/buy orders is multiplied by -1, so that mark-up can be added to a price.
        df['markup']= - df['markup']

    mask = np.isnan(df['markup'])
    df.loc[~mask, 'markup'] = df.loc[~mask,'markup'].round(0)
    #make np. nan to 0. these orders are filetered out anyway because no 0MW orders are allowed.
    df.loc[mask, 'markup'] = 0
    df['markup']=df['markup'].round().astype(int).copy()
    if not unit_test:
        return(list(df['markup']))
    else:
        df['DAP'] =DAP
        df['markup']=df['markup'].round().astype('int64').copy()
        df['opp_cost']=df['opp_cost'].round().astype('int64').copy()
        return(list(df['markup']), df)





def read_input_data(path,filename):

    simulation_parameters = pd.read_excel(os.path.join(path,filename), sheet_name=None)
    simulation_parameters['asset_input'] = pd.read_excel(os.path.join(path,filename), sheet_name='asset_input', index_col=0).squeeze()
    return (simulation_parameters)


#input directory
idir=r"C:\unit_tests"


#Files names of test scenarios
inames = ['unit_test_opportuntity_markup_downwards.xlsx','unit_test_opportuntity_markup_upwards.xlsx']
test_results =DataFrame(
        index=inames, columns=['number_unequal_result_parameters','unequal_results'])

all_result_df={}
for iname in inames:
    print(" -------------------------------" )
    print(" start test:                    ",iname)
    test_data = read_input_data(idir, iname)
    mark_ups, av_cap_result = opportunity_markup(
                   direction=test_data['asset_input']['direction'],
                   of_quantity = test_data['av_cap_input']['offer_quantity'].tolist(),
                   asset = None,
                   success_assumption = test_data['asset_input']['success_assumption'],
                   MTU_of_h_consideration= literal_eval(test_data['asset_input']['MTU_of_h_consideration']),
                   unit_test = {key: test_data[key] for key in ['asset_input','av_cap_input','opportunity_costs_input']})

    joint_dict={}
    all_result_df[iname] = av_cap_result
    test_data['expected_av_cap_output']['opp_cost'] =test_data['expected_av_cap_output']['opp_cost'].round().astype('int64').copy()
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
    test_results.loc[iname,'number_unequal_result_parameters']=len(unequal.copy())
    test_results.loc[iname,'unequal_results']=[unequal.copy()]

    if len(unequal)==0:
        print('The test results are equivalent')
    else:
        print('The following parameters of the test are not the same:')
        for key in unequal.keys():
            print(key)










