# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 09:30:00 2017
@author: Samuel Glismann

Market Model class of ASAM.

This class:
    1. initiates the model
    2. has a simulation step function, which triggers activities of all agents

"""

from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
from random import randrange, choice
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from Time import *
from Orderbook import *
from Asset import *
from MarketParty import *
from MarketOperator import *
from GridSystemOperator import *
from Reports import *
from ExogeniousDatabase import *
from Visualization import *



class MarketModel(Model):
    def __init__(self, simulation_parameters, seed = None):
        #seed != none allows for the same random numbers in various runs.

        #exogenious database
        self.exodata = ExoData(self, simulation_parameters)


        #Scheduler class of Mesa RandomActivation of market party agents
        self.schedule = RandomActivation(self)

        #areas
        self.gridareas = list(self.exodata.asset_portfolios['location'].unique())

        # Time class for a clock
        self.clock = Time(self, startday = self.exodata.sim_task['start_day'],
                          step_size = "15_minutes", startMTU = self.exodata.sim_task['start_MTU'],
                          step_numbers = self.exodata.sim_task['number_steps'],
                          DA_GCT = self.exodata.market_rules.loc['gate_closure_time','DAM'])

        #DataFrame that provides the time horizon for setting schedules and forward looking of agents
        self.schedules_horizon = DataFrame()

        if self.exodata.sim_task['run_DAM[y/n]']=='y':
            #Orderbook day-ahead market
            self.DAM_obook =Orderbook (self, ob_type='DAM')
            #Initiate market operator day-ahead
            self.DA_marketoperator = MO_dayahead(self, self.DAM_obook, self.exodata.market_rules['DAM'])

        if self.exodata.sim_task['run_IDM[y/n]']=='y':
            #Orderbook for intraday market
            self.IDM_obook = Orderbook (self,ob_type='IDM')
            #Initiate market operator for redispatch
            self.ID_marketoperator = MO_intraday(self, self.IDM_obook, self.exodata.market_rules['IDM'])

        if self.exodata.sim_task['run_RDM[y/n]']=='y':
            #Orderbook for redispatch
            self.red_obook = Orderbook (self, ob_type='redispatch')
            #Initiate market operator for redispatch
            self.RD_marketoperator = MO_redispatch(self, self.red_obook, self.exodata.market_rules['RDM'])

        if self.exodata.sim_task['run_BEM[y/n]']=='y':
            #Orderbook balancing market/mechanism
            self.BEM_obook =Orderbook (self, ob_type='BEM')
            #Initiate market operator balancing
            self.BE_marketoperator = MO_balancing_energy(self, self.BEM_obook, self.exodata.market_rules['BEM'])

        if self.exodata.sim_task['run_IBM[y/n]']=='y':
            #Orderbook imbalance market/mechanism
            self.IBM_obook = Orderbook (self, ob_type=None)
            #Initiate market operator imbalance
            self.IB_marketoperator = MO_imbalance(self, self.IBM_obook, self.exodata.market_rules['IBM'])

        self.plots = self.exodata.sim_task['plots_during_simulation']

        #create Grid and System Operator agent (e.g. TSO and/or DSO)
        self.aGridAndSystemOperator = GridSystemOperator("Grid_and_System_Operator", self)

        #dictionary referening to all market party agents
        self.MP_dict = {}
        # Create MP agents from exodata
        for i in self.exodata.asset_portfolios['asset_owner'].unique():
            #temp DF to make code better to read
            df = self.exodata.asset_portfolios.loc[self.exodata.asset_portfolios['asset_owner'] == i].reset_index()
            lst_assets =[]
            for k in range(len(df)):
                newasset = Asset(self, assetowner = i, assetname= str(df.loc[k,'asset_name']),
                                 pmax = df.loc[k,'pmax'].astype('int64'), pmin = df.loc[k,'pmin'].astype('int64'),
                                 location = df.loc[k,'location'], srmc = df.loc[k,'srmc'].astype('int64'),
                                 ramp_limit_up = df.loc[k,'ramp_limit_up'],
                                 ramp_limit_down = df.loc[k,'ramp_limit_down'],
                                 min_up_time = df.loc[k,'min_up_time'],
                                 min_down_time = df.loc[k,'min_down_time'],
                                 start_up_cost = df.loc[k,'start_up_cost'].astype('int64'),
                                 shut_down_cost = df.loc[k,'shut_down_cost'].astype('int64'),
                                 ramp_limit_start_up = df.loc[k,'ramp_limit_start_up'],
                                 ramp_limit_shut_down = df.loc[k,'ramp_limit_shut_down'])
                lst_assets.append([str(df.loc[k,'asset_name']), newasset])

            #asset portfolio provided to MarketParty class as DF with key and Asset() instances
            asset_portfolio = DataFrame(lst_assets, columns = ['ID', 'object'])
            asset_portfolio.set_index(['ID'], inplace = True)
            #get agent strategy from exodata
            if str(i) in self.exodata.agent_strategies['agent'].values:
                strategy = Series(self.exodata.agent_strategies.loc[self.exodata.agent_strategies['agent']==str(i)].squeeze())

            elif 'All' in self.exodata.agent_strategies['agent'].values:
                #if not specifically defined
                strategy = Series(self.exodata.agent_strategies.loc[self.exodata.agent_strategies['agent']=='All'].squeeze())
            else:
                raise Exception ('no usable strategy found for agent ',i)
            a = MarketParty(str(i), self, assets = asset_portfolio, agent_strategy = strategy)
            self.schedule.add(a)
            self.MP_dict[str(i)] = a


        #initiate Reports() class
        self.rpt = Reports(self)

        print('___Simulation task:')
        print(self.exodata.sim_task)
        print('___Simulated Portfolio:' )
        print(self.exodata.get_all_assets())

        #visualisation class for plotting
        self.visu = Visualizations(self)

        #get Dicts to use in MESA build-in datacollection and reporting methods
        self.dc = DataCollector(model_reporters = self.rpt.model_reporters,
                                agent_reporters = self.rpt.agent_reporters,
                                tables = self.rpt.table_reporters)

    def step(self):
        '''Advance the model by one step.'''
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>STEP>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("testmodel step", self.schedule.steps +1)
        print("testmodel MTU: ", self.clock.get_MTU())
        print("testModel day:", self.clock.get_day())

        #give a number to the agent steps, to later track the (random) order.
        self.agent_random_step_index = 0
        self.schedules_horizon = self.clock.asset_schedules_horizon().copy()

        if self.exodata.sim_task['run_DAM[y/n]']=='y':
            self.DA_marketoperator.clear_dayahead()

        #forecast error (assigned to agents_trade position)
        if self.exodata.sim_task['forecast_errors']=='exogenious':
            self.exodata.allocate_exo_errors()

        #determine expected imbalances prices based on day-ahead price in schedule_horizon format
        self.rpt.update_expected_IBP()

        #the MESA step counter is +1 after(!) schedule.step()
        self.schedule.step()

        if self.exodata.sim_task['run_RDM[y/n]']=='y':
            print("TSO agent determines congestions and redispatch demand")
            new_congestions = self.aGridAndSystemOperator.determine_congestions()
            self.aGridAndSystemOperator.redispatch_demand(new_congestions)
            self.RD_marketoperator.clear_redispatch()
        else:
            print('no TSO redispatch in this simulation')

        if self.exodata.sim_task['run_BEM[y/n]']=='y':
            self.BE_marketoperator.determine_control_state()

        if self.exodata.sim_task['run_IBM[y/n]']=='y':
            self.IB_marketoperator.imbalance_clearing()
            self.IB_marketoperator.imbalance_settlement()

        #note that the intraday market is cleared instantanously during every agent step
        #MESA data collector function
        self.dc.collect(self)

        if self.exodata.sim_task['save_intermediate_results[y/n]']=='y':
            self.rpt.save_market_stats(mode='every_step')
        elif self.exodata.sim_task['save_intermediate_results[y/n]']=='n':
            self.rpt.save_market_stats(mode='at_end')

        self.aGridAndSystemOperator.check_market_consistency()
        self.aGridAndSystemOperator.update_imbalances_and_returns(positions =[
                'imbalance_redispatch','imbalance_market(realized)',
                'imbalance_market(scheduled)' ])

        #plots
        if self.plots =='every_step':
            #Ensure that agent schedules are updated.
            #Otherwise dispatch and trade schedules are not plotted in synch. with simulation step.
            #The unchanged_position variable ensures that in a next step the updates can be skipped.
            for agent in self.schedule.agents:
                agent.update_trade_schedule(positions=['DA_position','ID_position','RD_position','BE_position'])
                agent.set_asset_commit_constraints()
                agent.portfolio_dispatch()
            self.visu.show_trade_per_agent()
            self.visu.show_dispatch_per_agent()
            self.visu.show_return_per_agent()
            self.visu.show_system_balance()

        elif self.plots =='every_change':
            #only plot when something changed
            something_changed=[]
            for agent in self.schedule.agents:
                if agent.unchanged_position == False:
                    something_changed +=[True]
            if something_changed:
                #Ensure that agent schedules are updated.
                #Otherwise dispatch and trade schedules are not plotted in synch. with simulation step.
                #The unchanged_position variable ensures that in a next step the updates can be skipped.
                for agent in self.schedule.agents:
                    agent.update_trade_schedule(positions=['DA_position','ID_position','RD_position','BE_position'])
                    agent.set_asset_commit_constraints()
                    agent.portfolio_dispatch()
                self.visu.show_trade_per_agent()
                self.visu.show_dispatch_per_agent()
                self.visu.show_return_per_agent()
                self.visu.show_system_balance()

        elif self.plots =='at_end':
            #no plots during simulation
            pass

        #calculate timestamp of last round
        day, MTU=   self.clock.calc_timestamp_by_steps(self.schedule.steps -1, 0)
        if (day==self.clock.end_date[0])&(MTU==self.clock.end_date[1]):

            if (self.plots =='at_end')|(self.plots =='every_change'):
                if (self.plots =='every_change'):
                    if (len(something_changed)>0):
                        #plots and updates already executed
                        pass
                else:
                    print(">>>>>>> Final update of trade schedules and dispatch optimalisation")
                    #Ensure that agent schedules are updated one last time
                    for agent in self.schedule.agents:
                        agent.update_trade_schedule(positions=['DA_position','ID_position','RD_position','BE_position'])
                        agent.set_asset_commit_constraints()
                        agent.portfolio_dispatch()
                    self.visu.show_trade_per_agent()
                    self.visu.show_dispatch_per_agent()
                    self.visu.show_return_per_agent()
                    self.visu.show_system_balance()
            self.visu.show_cost_distribution()
            self.visu.show_redispatch_PI()
            self.visu.show_redispatch_summary()
            self.visu.show_cleared_prices()

