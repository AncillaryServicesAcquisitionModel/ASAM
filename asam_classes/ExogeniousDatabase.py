# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 16:45:23 2017
@author: Samuel Glismann
"""

from mesa import Agent, Model
import pandas as pd
from pandas import Series, DataFrame
import random
import numpy as np
from mesa import Agent, Model


class ExoData():
    def __init__(self, model, simulation_parameters):
        self.model = model
        self.sim_task = None
        self.market_rules= None
        self.asset_portfolios = None
        self.congestions = None #todo: change to redispatch_demand
        self.agent_strategies = None
        self.forecast_errors= None
        self.DA_residual_load = None
        self.opportunity_costs_db=None
        self.sim_name =None
        self.output_path =None
        self.IBP_kde_pdfs =DataFrame()
        self.IBP_exo_prices=None
        self.solver_name = None
        self.control_state_probabilities = DataFrame()
        self.read_check_parameters(simulation_parameters)



    def read_check_parameters(self,simulation_parameters):
        """TODO: place more checks on the input parameters"""
        print('check input data')
        if not isinstance(simulation_parameters, dict):
            raise Exception('simulations parameters are not provided as dictionary')

        try:
            self.sim_task=simulation_parameters['simulation_task']
        except:
            raise Exception('simulation task not part of input parameters')

        self.sim_name = self.sim_task['simulation_name']
        self.output_path = simulation_parameters['output_path']

        try:
            self.market_rules=simulation_parameters['market_rules'].set_index('design_variable')
        except:
            raise Exception('market_rules not part of input parameters')
        try:
            self.agent_strategies=simulation_parameters['agent_strategies']
        except:
            raise Exception('agent_strategies not part of input parameters')
        try:
            self.asset_portfolios=simulation_parameters['portfolios']
            if 'Type' in self.asset_portfolios:
                #exclude 'artificial generators' from system_pmax for residual load, congestion scaling
                system_pmax = self.asset_portfolios['pmax'].loc[
                        self.asset_portfolios['Type']!='small flex aggregated'].sum()
            else:
               system_pmax = self.asset_portfolios['pmax'].sum()
        except:
            raise Exception('asset portfolios not part of input parameters')

        #optional input parameters
        try:
            if (self.sim_task['congestions'] =='from_scenario')|((
                    self.sim_task['residual_load_scenario'] !='flat_resload_profile')&(
                    self.sim_task['residual_load_scenario'] !='24h_residual_load_profile'))|(
                    self.sim_task['forecast_errors'] =='from_scenario'):
                idx = pd.IndexSlice
                #comes as multicolumn index. here it gets converted.

                self.DA_residual_load =simulation_parameters['da_residual_load'].loc[
                    :,idx[self.sim_task['residual_load_scenario'],:]]



                self.DA_residual_load.columns = self.DA_residual_load.columns.droplevel(0)

                #delete all rows containing NA in 'residual_load_DA' column.
                #This can happen if various RES load scenarios are in input data.
                self.DA_residual_load.dropna(axis=0,subset=['residual_load_DA'], inplace=True)


                #convert p.u. to MW and round to full MW
                mask = self.DA_residual_load.columns.isin(['residual_load_DA','FE_residual_load','congestion_MW',
                        'load_DA_cor','wind_DA', 'sun_DA'])
                self.DA_residual_load.loc[:, mask] = self.DA_residual_load.loc[:,mask] * system_pmax
                self.DA_residual_load.loc[:, mask] =self.DA_residual_load.loc[:, mask].round(0).astype('int64')

        except:
            if self.market_rules.loc['acquisition_method','DAM']=='single_hourly_auction':
                raise Exception('DA_residual_load not part of input parameters. required for DAM single_hourly_auction')

        try:
            if self.sim_task['congestions'] =='from_scenario':
                #forecast error is obtained fom residual load data. forecast error in tab are ignored
                self.congestions = self.DA_residual_load[['delivery_day','delivery_time','congestion_MW'
                                                          ,'redispatch_areas_down',
                                                          'redispatch_areas_up',
                                                          'identification_day',
                                                          'identification_MTU']]
                self.congestions.set_index(['delivery_day','delivery_time'], inplace=True)
            elif self.sim_task['congestions'] =='exogenious':
                self.congestions = simulation_parameters['congestions']

        except:
            if (self.sim_task['run_RDM[y/n]']=='y')&(self.sim_task['congestions']=='None'):
                import pdb
                pdb.set_trace()
                raise Exception('congestion required input parameters, when running redispatch simulation')
        try:
            if self.sim_task['forecast_errors'] =='from_scenario':
                #forecast error is obtained fom residual load data. forecast error in tab are ignored
                self.forecast_errors = self.DA_residual_load[['delivery_day','delivery_time','FE_residual_load']]

            elif self.sim_task['forecast_errors'] =='exogenious':
                self.forecast_errors=simulation_parameters['forecast_errors']
            else:
                pass
        except:
            raise Exception('forecast error input required, when running exogenious forecast error allocation')

        try:
            self.opportunity_costs_db=simulation_parameters['opportunity_costs']
        except:
            if (self.agent_strategies =='opportunity_markup').any().any():
                raise Exception('opportunity costs estimates required for agent strategies')
        try:
            self.IBP_kde_pdfs=simulation_parameters['IBP_kde_pdfs']
        except:
            #this is not yet fully consistent.
            if self.market_rules.loc['acquisition_method','BEM']=='control_states_only':
                raise Exception('Balancing energy method based on probability samples requires IBP pdfs to estimate  BEP')
        try:
            self.control_state_probabilities= simulation_parameters['control_state_probabilities']
        except:
            #this is not yet fully consistent.
            if self.market_rules.loc['acquisition_method','BEM']=='control_states_only':
                raise Exception('Balancing energy method based on probability samples requires control_state_probs to estimate control state')
        try:
            self.IBP_exo_prices= simulation_parameters['IBP_exo_prices']
        except:
            #this is not yet fully consistent.
            if self.market_rules.loc['pricing_method','IBM']=='exogenious':
                raise Exception('Imbalance pricing method -exogenious- expects a dataframe with short and long prices as well as timestamps')


        if self.sim_task['start_day'] =='from_scenario':
            self.sim_task['start_day'] = self.DA_residual_load['delivery_day'].iloc[0]
        if self.sim_task['start_MTU'] =='from_scenario':
            self.sim_task['start_MTU'] = self.DA_residual_load['delivery_time'].iloc[0]
        if self.sim_task['number_steps'] == 'from_scenario':
            self.sim_task['number_steps'] = len(self.DA_residual_load)

        try:
            #used for PyPSA
            self.solver_name = self.sim_task['solver_name']
        except:
            raise Exception ('Solver name needed, that is installed and recognisable by Pyomo, e.g. "glpk", "gurobi", "cbc"')



    def get_DA_resload(self, timestamps_df, mode = None):
        """timestamps_df must contain 'delivery_day' and must contain
          either 'delivery_hour' or 'delivery_time' (for later use)"""

        if mode == 'flat_resload_profile':
            system_pmax = self.asset_portfolios['pmax'].sum()
            resload = [0.8 * system_pmax] * len(timestamps_df)
        elif mode == '24h_residual_load_profile':
            #here a list of 24h values can be edited manually.
            #These values will be used as residual load for every DA market simulation (independent from day en time)
            data_lst = [
                    0.6,
                    0.6,
                    0.6,
                    0.6,
                    0.6,
                    0.6,
                    0.7,
                    0.7,
                    0.7,
                    0.7,
                    0.7,
                    0.7,
                    0.9,
                    0.9,
                    0.9,
                    0.9,
                    0.9,
                    0.9,
                    0.7,
                    0.7,
                    0.7,
                    0.7,
                    0.7,
                    0.7]
            system_pmax = self.asset_portfolios['pmax'].sum()
            resload = [i*system_pmax for i in data_lst]
            #ensure that the residual load curve starts start of the DA auction (especially in the first step)
            #done by doubeling the list and starting at the hour
            resload = resload * 2
            resload=resload[len(resload) - len(timestamps_df):]
        else:
            #residual load from scenarios for specific timestamps
            if 'delivery_hour' in timestamps_df.columns:
                #DA runs on hours.So residual load must be grouped to hours
                resload = self.DA_residual_load.groupby(by=['delivery_day', 'delivery_hour']).mean()['residual_load_DA']
                #filter on day and mtu list
                resload = resload.loc[resload.index.isin(list(
                        timestamps_df.set_index(['delivery_day','delivery_hour']).index.values))].copy()
                resload = list(resload.astype('int'))
            else:
                raise Exception(' get_DA_load has delivery_mtu not yet implemented. only delivery_hour')
        return (resload)

    def allocate_exo_errors(self, mode='exogenious'):
        """a positive error (in p.u. of pmax) is considered as a long trade position.
            a negative error means a as a short trade position. Short may thus lead to increasing dispatch of that agent."""
        if mode == 'exogenious':
            new_error = self.forecast_errors.loc[(self.forecast_errors['identification_day']==self.model.clock.get_day())&(
                    self.forecast_errors['identification_MTU']==self.model.clock.get_MTU())].reset_index()
            if new_error.empty:
                print('no new exog. forecast errors identified')
            else:
                print('new forcast error identified:')
                print(new_error)
                for i in range(len(new_error)):
                    error_DF =  self.model.schedules_horizon.copy()
                    error_DF.rename(columns = {'commit':'new_error'}, inplace = True)
                    error_DF['new_error'].loc[(slice(new_error.loc[i,'error_start_day'].astype('float64'),new_error.loc[i,'error_end_day'].astype('float64')),
                                                         slice(new_error.loc[i,'error_start_time'].astype('float64'),new_error.loc[i,'error_end_time'].astype('float64'))
                                                         )] = new_error.loc[i,'error_magnitude_pu']
                    if new_error.loc[i,'who'] == 'all':
                        for agent in self.model.schedule.agents:
                            agent_pmax = self.asset_portfolios.loc[self.asset_portfolios['asset_owner'] == agent.unique_id,'pmax'].sum()
                            #error in MW
                            error_DF['forecast_error']=error_DF['new_error'] * agent_pmax
                            agent.trade_schedule['forecast_error'] = agent.trade_schedule['forecast_error'].add(error_DF['forecast_error'], fill_value = 0)
                            agent.unchanged_position = 'forecast_error'
                    elif new_error.loc[i,'who'] in self.model.MP_dict:
                        agent = self.model.MP_dict[new_error.loc[i,'who']]
                        agent_pmax = self.asset_portfolios.loc[self.asset_portfolios['asset_owner'] == agent.unique_id,'pmax'].sum()
                        #error in MW
                        error_DF['forecast_error'] = error_DF['new_error'] * agent_pmax
                        agent.trade_schedule['forecast_error'] = agent.trade_schedule['forecast_error'].add(error_DF['forecast_error'], fill_value = 0)
                        agent.unchanged_position = 'forecast_error'

        elif mode == 'from_scenario':
            #ATTENTION THIS DOES NOT WORK YET
            error_DF =  self.model.schedules_horizon.copy()
            error_DF.rename(columns = {'commit':'new_error'}, inplace = True)

        elif new_error.loc[i,'who'] == 'system_e_randomly_distributed':
            """the given value is the sum of all forecast errors in the system
            proportional to the installed capacity. This value is uniformly
            distributed over the agents. can also be upto 25% negative for an agent,
            while the total system error is positive (and vice versa)."""
            system_pmax = self.asset_portfolios['pmax'].sum()
            dividers = sorted(random.sample(range(int(-0.25* abs(system_pmax)),
                                                  int(abs(system_pmax))), len(self.model.schedule.agents) - 1))
            random_err = [a - b for a, b in zip(dividers + [int(abs(system_pmax))], [0] + dividers)]
            #shuffle the list to make sure that error is distributed randomly
            random.shuffle(random_err)
            i = 0
            for agent in self.model.schedule.agents:
                error_DF['forecast_error']=(error_DF['new_error'] * random_err[i]).round()
                agent.trade_schedule['forecast_error'] = agent.trade_schedule['forecast_error'].add(error_DF['forecast_error'], fill_value = 0)
                agent.unchanged_position = 'forecast_error'
                i += 1
        else:
            raise Exception('forecast error recepient not known')

    def get_all_assets(self):
        """return the total asset portfolio of initialized agents"""
        assetsdf = DataFrame()
        for agent in self.model.schedule.agents:
            all_ids = agent.assets.index.values
            # this is a place holder for a Day-ahead market result dispatch method
            for i in range(len(all_ids)):
                a = agent.assets.loc[all_ids[i],:].item()
                df = a.get_as_df()
#                df.insert(loc = 0, column ='agent_id',value = agent.unique_id)
                assetsdf = pd.concat([assetsdf, df], ignore_index=True)
        return(assetsdf)



    def generate_forecast_errors(self):
        for agent in self.model.schedule.agents:
            agent.trade_schedule['forecast_error'] = 0


    def IB_default_price (self, day, time):
        self.imbalance_price = 35
        return (self.imbalance_price)











