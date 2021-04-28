# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 15:24:23 2017
@author: Samuel Glismann

Main class to run ASAM simulations.
This script reads input files, initiates an ASAM model from it, starts the simulation,
collects results and stores them in an excel file.



"""

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from MarketModel import *
from Visualization import *
import mesa
from IPython import get_ipython
import os


pd.options.mode.chained_assignment = None  # default='warn'
def read_input_data(path,filename):
    """
    - Method: read exogenous data for the simulation from excel
    - path is the file directory
    - filename is the name of the input file (including .xlsx)
    """

    #prepare parameters for model
    print('read exogenious data')
    simulation_parameters =pd.read_excel(path+filename, sheetname=None)
    simulation_parameters['da_residual_load'] = pd.read_excel(path+filename, sheetname='da_residual_load', header =[0,1])
    simulation_parameters['simulation_task'] = pd.read_excel(path+filename, sheetname='simulation_task', index_col=0).squeeze()

    #get IBP kde pdfs from hdf5 file
    IBP_kde_pdfs = pd.read_hdf(path+'IBP_pdf_kernels_allISPcs_20201213.h5', 'IBP_pdf_kernels')
    #add dataframe from pickl to dictionary
    simulation_parameters['IBP_kde_pdfs'] = IBP_kde_pdfs
    return (simulation_parameters)

"""directories to be entered"""
#input directory
idir=r'input_data/'
#output directory
rdir=r'results/'
#filename
iname = "example_scenario.xlsx"

#read simulation input file
simulation_parameters = read_input_data(idir, iname)
#
simulation_parameters['output_path'] = rdir

simulation_start_time = datetime.now().replace(microsecond=0)
sim_task = simulation_parameters['simulation_task']

#initiate model
model = MarketModel(simulation_parameters, seed = sim_task['seed'])
#run simulation steps
for i in range(sim_task['number_steps']):
    model.step()

simulation_end_time = datetime.now().replace(microsecond=0)
simulation_run_time = simulation_end_time - simulation_start_time

print(">>>>>>>>>>>>>>>>>>END Simulation>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("process results for output")


#prepare storing simulation inputs
simulation_task= model.exodata.sim_task
simulation_task['sim_start_time'] = simulation_start_time
simulation_task['sim_end_time'] = simulation_end_time
simulation_task['sim_run_time'] = str(simulation_run_time)
simname=simulation_task['simulation_name']+'_'
assetsdf = model.exodata.get_all_assets()
congestionsdf = model.exodata.congestions
agent_strategiesdf = model.exodata.agent_strategies
market_rulesdf = model.exodata.market_rules
forecast_errorsdf = model.exodata.forecast_errors


###excel writing main input data
stamp= str(datetime.now().replace(microsecond=0))
stamp=stamp.replace('.','')
stamp=stamp.replace(':','_')
writer = pd.ExcelWriter(rdir+'ModelResults'+simname+stamp+'.xlsx', engine='xlsxwriter')
simulation_task.to_frame().to_excel(writer, sheet_name = 'simulation_input')
startrow = len(simulation_task)+2
market_rulesdf.to_excel(writer, sheet_name = 'simulation_input', startrow=startrow)
startrow += len(market_rulesdf)+2
agent_strategiesdf.to_excel(writer, sheet_name = 'simulation_input', startrow=startrow)
startrow += len(agent_strategiesdf)+2
assetsdf.to_excel(writer, sheet_name = 'simulation_input', startrow=startrow)
startrow += len(assetsdf)+2
congestionsdf.to_excel(writer, sheet_name = 'simulation_input', startrow=startrow)
startrow += len(congestionsdf)+2
forecast_errorsdf.to_excel(writer, sheet_name = 'simulation_input', startrow=startrow)
startrow += len(forecast_errorsdf)+2

#interdependence indicators
indicators, allprofitloss =model.rpt.interdependence_indicators()
indicators.to_excel(writer,sheet_name = 'interdependence_indicators')

keyfigures=model.rpt.final_keyfigures()
###change index value for plotting purposes label
keyfigures.loc['imbalance_market',:]=keyfigures.loc['imbalance_market(scheduled)',:]
keyfigures.drop('imbalance_market(scheduled)', axis=0, inplace=True)
keyfigures.to_excel(writer, sheet_name ='key_figures')

model.rpt.redispatch_PI().to_excel(writer, sheet_name = 'performance_indicators')

#get results of reporters as DataFrame (mesa)
agentdf = model.dc.get_agent_vars_dataframe()
agentdf= agentdf.unstack(1)
agentdf.sort_index(axis=1, inplace=True)
agentdf.to_excel(writer, sheet_name = 'AgentResults')

#add mark-up analyses (mark-up is added to order dataframes)
model.rpt.mark_ups_analysis()

#all order collection per round
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

if sim_task['run_DAM[y/n]']=='y':
    all_collections += [model.DAM_obook.sellorders_all_df,
                      model.DAM_obook.buyorders_all_df,
                      model.DAM_obook.cleared_sellorders_all_df,
                      model.DAM_obook.cleared_buyorders_all_df]
    collection_names +=['DAMsellorders','DAMbuyorders','DAc_sellorders','DAc_buyorders']

if sim_task['run_BEM[y/n]'] =='y':
    all_collections += [model.BEM_obook.sellorders_all_df,
                  model.BEM_obook.buyorders_all_df]
    collection_names +=['BEsellorders','BEbuyorders']

bb=0
for a in all_collections:
    if not a.empty:
        a.to_excel(writer,sheet_name=collection_names[bb])
    bb+=1

model.rpt.get_cleared_prices().to_excel(writer,sheet_name='cleared_prices')
model.rpt.get_system_dispatch().to_excel(writer,sheet_name='system_dispatch')
model.rpt.get_all_trade_schedules().to_excel(writer,sheet_name='trade_schedules')
model.rpt.get_all_returns().to_excel(writer,sheet_name='all_returns')
writer.save()


print(" done------------------------------------")






