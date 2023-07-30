# -*- coding: utf-8 -*-
"""
Created on  4 October 2020
@author: Samuel Glismann

Script for integration tests of ASAM.

Several scenarios can be simulated and compared with precalculated
baseline results.

It is required to first make the baseline result xlsx files of the test scenario.

The dataframe overview_test_results indicates per scenario how many result parameters are unequal.
These can be explored


"""

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from MarketModel import *
from Asset import *
from Visualization import *
import mesa
from IPython import get_ipython
import os


pd.options.mode.chained_assignment = None  # default='warn'

def read_input_data(path,filename):

    """
    - Method: Exogenous data for the simulation and test reference results are read from excel
    - path is the file directory
    - filename is the name of the input file (including .xlsx)
    - Templates and examples of an input file can be found in the Github repository of ASAM.
    - Baseline results excel file for the test should have the name 'Baseline_' + filename
    - Baseline results are to be located in a subfolder of the directory 'path' with name 'baseline_results'
    """
    #prepare parameters for model
    print('read exogenious data')
    simulation_parameters = pd.read_excel(os.path.join(path,filename), sheet_name=None)
    simulation_parameters['da_residual_load'] = pd.read_excel(os.path.join(path,filename), sheet_name='da_residual_load', header =[0,1])
    simulation_parameters['simulation_task'] = pd.read_excel(os.path.join(path,filename), sheet_name='simulation_task', index_col=0).squeeze()
    baseline_results =pd.read_excel(os.path.join(path+r'\baseline_results','Baseline_'+filename), sheet_name=None)
    #add allreturns as sum
    try:
        baseline_results['all_returns'] =pd.read_excel(os.path.join(path+r'\baseline_results','Baseline_'+filename),sheet_name ='all_returns', index_col=0, header=[0,1]).sum(numeric_only=True).unstack()
    except:
        baseline_results['all_returns'] = DataFrame()
#    #get IBP kde pdfs from hdf5 file
    IBP_kde_pdfs = pd.read_hdf(os.path.join(path+'IBP_pdf_kernels_allISPcs_20201213.h5'), 'IBP_pdf_kernels')
    #add dataframe from pickl to dictionary
    simulation_parameters['IBP_kde_pdfs'] = IBP_kde_pdfs
    return (simulation_parameters,baseline_results)

def join_the_simu_data(res_dict):
    """
    Method: Creates a dictionary of result multiindex dataframes, whereby test results and baseline results are joined
    Input: res_dict is dictionary with two keys: test scenario name and baseline of the scenario. The entries contain all results per simulation
    """
    joint_dict={}

    #Result data set to be joined
    dset = ['simulation_input','key_figures','performance_indicators','interdependence_indicators',
        'IDbuyorders', 'IDsellorders', 'RDMbuyorders','RDMsellorders', 'BEsellorders','BEbuyorders',
        'RDMc_sellorders', 'RDMc_buyorders', 'cleared_prices','all_returns',  'IDc_sellorders',
        'IDc_buyorders','RDM_demand_downward','RDM_demand_upward']
    #make joint multiindex columns df from various simulations per data set
    for res in dset:
        if res == 'simulation_input':
            #remove simulation-specific values
            for simu in all_simus:
                res_dict[simu][res]=res_dict[simu][res].set_index(
                    'parameter').drop(labels=[
                            'simulation_name','sim_start_time','sim_end_time','sim_run_time'],axis=0
                            ).reset_index()

        df= pd.concat([res_dict[simu][res] for simu in all_simus],axis=1,keys=all_simus).swaplevel(
            0,1,axis=1).sort_index(axis=1, ascending=True,level=[0,1])
        joint_dict[res]=df
    return(joint_dict)


"""directories to be entered"""
#input directory
idir=r"C:\test_input_data\\"
#result directory
rdir=r"C:\test_results\\"


#Files names of test scenarios
inames = ['test_scenario_1.xlsx','test_scenario_2.xlsx','test_scenario_3.xlsx', 'test_scenario_4.xlsx']


overview_test_results =DataFrame(index=inames, columns=['number_unequal_result_parameters','unequal_results'])

#simulation per test scenario
for iname in inames:

    print(" -------------------------------" )
    print(" start test:                    ",iname)
    simulation_parameters, baseline_results = read_input_data(idir, iname)
    #
    simulation_parameters['output_path'] = rdir
    ######
    ##run various rounds
    simulation_start_time = datetime.now().replace(microsecond=0)
    ####
    sim_task = simulation_parameters['simulation_task']
    ####
    model = MarketModel(simulation_parameters, seed = sim_task['seed'])
    ####
    for i in range(sim_task['number_steps']):
        model.step()
    ####
    simulation_end_time = datetime.now().replace(microsecond=0)
    simulation_run_time = simulation_end_time - simulation_start_time

    print(">>>>>>>>>>>>>>>>>>END>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("results")

    """Test results are first stored in excel and then re-opened in python
    to ensure exact same formate as the baseline_results from excel"""

    #prepare storing simulation inputs
    simulation_task= model.exodata.sim_task
    simulation_task['sim_start_time'] = simulation_start_time
    simulation_task['sim_end_time'] = simulation_end_time
    simulation_task['sim_run_time'] = str(simulation_run_time)
    simname=simulation_task['simulation_name']
    assetsdf = model.exodata.get_all_assets()
    congestionsdf = model.exodata.congestions
    agent_strategiesdf = model.exodata.agent_strategies
    market_rulesdf = model.exodata.market_rules
    forecast_errorsdf = model.exodata.forecast_errors

    ###excel writing
    writer = pd.ExcelWriter(os.path.join(rdir,'TestResults_'+simname+'.xlsx'), engine='xlsxwriter')
    simulation_task.to_frame().to_excel(writer, sheet_name = 'simulation_input')
    startrow = len(simulation_task)+2
    market_rulesdf.to_excel(writer, sheet_name = 'simulation_input', startrow=startrow)
    startrow += len(market_rulesdf)+2
    assetsdf.to_excel(writer, sheet_name = 'simulation_input', startrow=startrow)
    startrow += len(assetsdf)+2
    congestionsdf.to_excel(writer, sheet_name = 'simulation_input', startrow=startrow)
    startrow += len(congestionsdf)+2

    indicators, allprofitloss =model.rpt.interdependence_indicators()
    indicators.to_excel(writer,sheet_name = 'interdependence_indicators')

    #key indicators (summary)
    keyfigures=model.rpt.final_keyfigures()
    ###change index value for plotting purposes label
    keyfigures.loc['imbalance_market',:]=keyfigures.loc['imbalance_market(scheduled)',:]
    keyfigures.drop('imbalance_market(scheduled)', axis=0, inplace=True)
    keyfigures.to_excel(writer, sheet_name ='key_figures')

    model.rpt.redispatch_PI().to_excel(writer, sheet_name = 'performance_indicators')

    #all order collection for excel store
    all_collections=[]
    collection_names=[]
    if sim_task['run_IDM[y/n]'] =='y':
        all_collections += [model.IDM_obook.sellorders_all_df,
                      model.IDM_obook.buyorders_all_df,
                      model.IDM_obook.cleared_sellorders_all_df,
                      model.IDM_obook.cleared_buyorders_all_df]
        collection_names +=['IDsellorders','IDbuyorders','IDc_sellorders','IDc_buyorders']

    if sim_task['run_RDM[y/n]'] =='y':
        all_collections += [model.red_obook.sellorders_all_df,
                      model.red_obook.buyorders_all_df,
                      model.red_obook.cleared_sellorders_all_df,
                      model.red_obook.cleared_buyorders_all_df,
                      model.red_obook.redispatch_demand_upward_all_df,
                      model.red_obook.redispatch_demand_downward_all_df]
        collection_names += ['RDMsellorders','RDMbuyorders','RDMc_sellorders','RDMc_buyorders', 'RDM_demand_upward','RDM_demand_downward']

    if sim_task['run_DAM[y/n]'] =='y':
        all_collections += [model.DAM_obook.cleared_sellorders_all_df,
                          model.DAM_obook.cleared_buyorders_all_df]
        collection_names +=['DAc_sellorders','DAc_buyorders']

    if sim_task['run_BEM[y/n]'] =='y':
        all_collections += [model.BEM_obook.sellorders_all_df,
                      model.BEM_obook.buyorders_all_df]
        collection_names +=['BEsellorders','BEbuyorders']

    bb=0
    for a in all_collections:
        # if not a.empty:
        a.to_excel(writer,sheet_name=collection_names[bb])
        bb+=1

    model.rpt.get_cleared_prices().to_excel(writer,sheet_name='cleared_prices')
    model.rpt.get_system_dispatch().to_excel(writer,sheet_name='system_dispatch')
    model.rpt.get_all_trade_schedules().to_excel(writer,sheet_name='trade_schedules')
    model.rpt.get_all_returns().to_excel(writer,sheet_name='all_returns')
    writer.save()

    """>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
    print("Comparing test results with baseline results")

    #compare simulation results to baseline results
    test_results  =pd.read_excel(os.path.join(rdir,'TestResults_'+simname+'.xlsx'), sheet_name=None)
    #add allreturns as sum
    try:
        test_results['all_returns'] =pd.read_excel(os.path.join(rdir,'TestResults_'+simname+'.xlsx'),
                    sheet_name ='all_returns', index_col=0, header=[0,1]).sum(numeric_only=True).unstack()
    except:
        test_results['all_returns'] = DataFrame()

    all_simus = ['Baseline_'+simname,simname]
    #join the simulation data in various dataframes
    joint_dict= join_the_simu_data({'Baseline_'+simname:baseline_results, simname:test_results})
    #The following lists and dictionaries are overwritten in every test loop
    all_params=[]
    params_equal=[]
    unequal={}
    idx = pd.IndexSlice
    #comparing each parameter of all dataframes of the results (is equivalent?)
    for result in joint_dict:
        params=list(joint_dict[result].columns.get_level_values(0).unique())
        for param in params:
            eq=joint_dict[result].loc[:,idx[param,all_simus[0]]].equals(joint_dict[result].loc[:,idx[param,all_simus[1]]])
            all_params+=[(result,param)]
            params_equal+=[eq]
    #Dataframe storing per parameter if results (test and baseline) are equivalent or not
    test_result=DataFrame(list(zip(all_params,params_equal)), columns=['parameter', 'is_equivalent'])
    for i in range(len(test_result.loc[test_result['is_equivalent']==False])):
        param=test_result.loc[test_result['is_equivalent']==False]['parameter'].iloc[i]
        unequal[param]=joint_dict[param[0]].loc[:, idx[param[1],:]]

    #Exclusion of parameters with random results

    if simulation_parameters['market_rules'].set_index('design_variable'
                                                       )['IBM']['pricing_method'
                                                                                ] != 'exogenious':
        #random IBM prices lead to unequal results when comparing simulations. Therefore they are excluded.
        #However, realised imbalance may lead to differences in costs, profits and returns.
        excl_param = [('cleared_prices', 'IBP_long'),
                     ('cleared_prices', 'IBP_short'),
                      ('cleared_prices', 'control_state')]
        for ex_param in excl_param:
            if ex_param in list(unequal.keys()):
                del unequal[ex_param]

    if len(unequal)==0:
        print('The test results are equivalent')
    else:
        print('The following parameters of the test are not the same:')
        for key in unequal.keys():
            print(key)
            print(unequal[key])
    overview_test_results.loc[iname,'number_unequal_result_parameters']=len(unequal.copy())
    overview_test_results.loc[iname,'unequal_results']=[unequal.copy()]
if (overview_test_results['number_unequal_result_parameters']==0).all():
    print('-------------all tests have equal results with the baseline results')
else:
     print('-------------Some tests show different results than the baseline results')

print(" done------------------------------------")






