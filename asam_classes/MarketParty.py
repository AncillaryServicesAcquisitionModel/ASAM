# -*- coding: utf-8 -*-
"""
Created on Fri May  5 11:44:08 2017
@author: Samuel Glismann

Market Party agent class of ASAM.
A Market party agent:
            - has assets (ID, Pmax, Pmin, SMRC, Location)
            - trade schedule
            - a financial balance sheet

Methods are:
    - init
    - step
    - update_trade_schedule
    - set_asset_contraints
    - portfolio_optimization
    - place_ID_orders
    - place_RD_orders
    - place_BE_orders
    - small random quantities
    - start_stop_blocks
    - intra-day markup
    - opportunity_markups
    - start_stop_markups
    - ramping_markup
    - doublescore_markup

Note: for portfolio_optimization PyPSA and Pyomo are applied.

"""
from mesa import Agent, Model
from mesa.time import RandomActivation
from random import randrange, choice
import pandas as pd
import math
import pypsa
from pandas import Series, DataFrame
import numpy as np
from OrderMessage import *
from pyomo.environ import (ConcreteModel, Var, Objective,
                           NonNegativeReals, Constraint, Reals,
                           Suffix, Expression, Binary, SolverFactory)

from pypsa.opt import (l_constraint, l_objective, LExpression, LConstraint)

class MarketParty(Agent):

    def __init__(self, unique_id, model, assets = None, agent_strategy = None):
        super().__init__(unique_id, model)
        self.model = model
        # Series of strategy items
        self.strategy = agent_strategy
        #DF with Asset() instances and unique ID as index
        if len(assets.index) != len(assets.index.unique()):
            raise Exception('asset ids must be unique per agent',assets.index)
        else:
            self.assets = assets
        self.money = 0
        self.step_rank = None
        #Variable to skip portfolio optimization (if no trade and no forecast changes,
        #no optimization is calculated)
        self.unchanged_position = False #values: True, False, 'forecast error'
        #1000 as standard imbalance risk price. This is only used to provide an artificial price for market orders
        try:
            self.imbalance_risk_price = self.model.IB_marketoperator.IBP_fixed
        except:
            self.imbalance_risk_price = 1000

        #Some strategy consistency checks
        if not(self.strategy.loc['IBM_pricing']=='marginal_orderbook_strategy')|(
               self.strategy.loc['IBM_pricing']=='market_order_strategy')|(
               self.strategy.loc['IBM_pricing']=='impatience_curve'):
            raise Exception ('imbalance pricing strategy not implemented in agents')
        if (self.strategy.loc['IBM_pricing']=='impatience_curve')|(self.strategy.loc['IBM_quantity']=='impatience_curve'):
            if not self.strategy.loc['IBM_pricing']==self.strategy.loc['IBM_quantity']:
                raise Exception ('Agent strategy impatient_curve must be appled for both quantity and pricing strategy')

        #All orders should be in a non-multiindex format. The dataframes are emptied per round
        orderlabels =['agent_id','associated_asset','delivery_location','quantity','price',
                      'delivery_day','delivery_time','order_type','init_time', 'order_id',
                      'direction','matched_order','cleared_quantity','cleared_price','rem_vol', 'due_amount']
        self.accepted_red_orders = DataFrame( columns = orderlabels) #emptied in set_asset_commit_constraints()
        self.accepted_ID_orders = DataFrame(columns = orderlabels)  #emptied in update_trade_schedule()
        self.accepted_BE_orders = DataFrame(columns = orderlabels)  #emptied in update_trade_schedule()
        self.accepted_DA_orders = DataFrame(columns = orderlabels) #emptied in update_trade_schedule()
        self.ordercount = 1
        self.trade_schedule = model.clock.asset_schedules_horizon() #manipulated every round. All trade positions where delivery periode>current time remain unchanged
        self.trade_schedule= self.trade_schedule.reindex(columns=  [
                'DA_position','ID_position','RD_position','BE_position',
                'forecast_error','total_trade', 'imbalance_position', 'total_dispatch'], fill_value= 0)
        self.financial_return = model.clock.asset_schedules_horizon() #manipulated every round. All returns where delivery periode>current time remain unchanged
        self.financial_return = self.financial_return.reindex(columns = [
                'DA_return','ID_return','RD_return', 'BE_return','IB_return',
                'total_return', 'total_dispatch_costs', 'profit'])


        #initiate PyPSA model for optimal asset dispatch
        self.commit_model = pypsa.Network()
        self.commit_model.add("Bus","bus")
        self.commit_model.add("Load","trade_position",bus="bus")
        #add a generator that captures the unfeasible trade commitment (open short position)
        self.commit_model.add("Generator",'short_position' ,bus="bus",
               committable=True,
               p_min_pu= 0,
               marginal_cost=self.imbalance_risk_price,
               p_nom=10000)
        #add a negative generator unit to capture unfeasible trade commitment (open long positions)
        #This generator has a negative price.
        self.commit_model.add("Generator",'long_position',bus="bus",
               committable = True,
               p_min_pu= 1,
               p_max_pu= 0,
               p_nom=-10000,
               marginal_cost= -self.imbalance_risk_price)
        all_ids = self.assets.index.values

        #Include all assets of the portfolio
        for i in range(len(all_ids)):
            asset = self.assets.loc[all_ids[i],:].item()
            self.commit_model.add("Generator",asset.assetID ,bus="bus",
                   committable=True,
                   p_min_pu= asset.pmin/asset.pmax,
                   marginal_cost = asset.srmc,
                   p_nom=asset.pmax)
            #Agent strategy determines which constraints are taken into account during asset optimization
            if self.strategy['ramp_limits']==True:
                self.commit_model.generators.ramp_limit_up[asset.assetID] = asset.ramp_limit_up
                self.commit_model.generators.ramp_limit_down[asset.assetID] = asset.ramp_limit_down
                #TODO: find out what the impact of ramp_limit start stop is.
                #PyPSA question not yet answered. start_stop ramps only used in startstop price determination.
#                self.commit_model.generators.ramp_limit_start_up[asset.assetID] = asset.ramp_limit_start_up
#                self.commit_model.generators.ramp_limit_shut_down[asset.assetID] = asset.ramp_limit_shut_down
            if self.strategy['start_stop_costs']==True:
                self.commit_model.generators.start_up_cost[asset.assetID] = asset.start_up_cost
                self.commit_model.generators.shut_down_cost[asset.assetID] = asset.shut_down_cost
            if self.strategy['min_up_down_time']==True:
                self.commit_model.generators.min_up_time[asset.assetID] = asset.min_up_time
                self.commit_model.generators.min_down_time[asset.assetID] = asset.min_down_time

    def step(self):
        """
        Step method executes all agent methods.
        Note: the order of the methods matters.
        """
        #add 1 to agent step order and store for report
        self.model.agent_random_step_index += 1
        #trace random step rank
        self.step_rank = self.model.agent_random_step_index

        self.update_trade_schedule(positions=['DA_position','ID_position','RD_position','BE_position'])
        self.set_asset_commit_constraints()
        self.portfolio_dispatch()
        #place_ID_order leads to instatanous IDM clearing
        self.place_ID_orders()
        if not self.accepted_ID_orders.empty:
            print('processing ID clearing result with another trade schedule update and portfolio optimization')
            #after instantanous clearing, another portfolio optimization is needed before redispatch orders can be made
            self.update_trade_schedule(positions=['ID_position'])
            self.portfolio_dispatch()
        self.place_RD_orders()
        self.place_BE_orders()


    def update_trade_schedule(self, positions =[]):
        """
        Method:
            Aggregates all offered orders that led to transactions into a trade schedule.
            Moreover, a financial balance sheet is updated with the financial returns.
        Input:
            positions (list): includes all positions to be updated

            """
        print('start update trade schedule of Agent ', self.unique_id)
        if (self.accepted_red_orders.empty)&(self.accepted_ID_orders.empty)&(
                self.accepted_DA_orders.empty)&(
                        self.accepted_BE_orders.empty)&(
                                self.unchanged_position != 'forecast_error'):
            #no new trades and no new forecast errors in last round
            self.unchanged_position = True
            print('no position has changed of this agent')
        else:
            self.unchanged_position = False
        new_trade_schedule = self.model.schedules_horizon.copy()
        new_trade_schedule = new_trade_schedule.add(self.trade_schedule,fill_value = 0)
        new_trade_returns = self.model.schedules_horizon.copy()
        new_trade_returns = new_trade_returns.add(self.financial_return,fill_value = 0)
        if self.unchanged_position == False:
            for i in positions:
                new_transactions = DataFrame()
                if i == 'DA_position':
                    new_transactions =self.accepted_DA_orders[['delivery_day','delivery_time'
                                                         ,'cleared_quantity','direction', 'due_amount']]
                    k = 'DA_return'
                    #clear accepted_orders DataFrame. Will be filled again after settlement this round
                    self.accepted_DA_orders = self.accepted_DA_orders.iloc[0:0]
                elif i == 'ID_position':
                    new_transactions =self.accepted_ID_orders[['delivery_day','delivery_time'
                                                         ,'cleared_quantity','direction', 'due_amount']]
                    k = 'ID_return'

                    #clear accepted_orders DataFrame. Will be filled again after settlement this round
                    self.accepted_ID_orders = self.accepted_ID_orders.iloc[0:0]
                elif i == 'RD_position':
                    new_transactions =self.accepted_red_orders[['delivery_day','delivery_time'
                                                         ,'cleared_quantity','direction', 'due_amount']]
                    k = 'RD_return'
                    #accepted redispatch orders are cleared in set_asset_commit_constraints ()
                elif i == 'BE_position':
                    new_transactions =self.accepted_BE_orders[['delivery_day','delivery_time'
                                                         ,'cleared_quantity','direction', 'due_amount']]
                    k = 'BE_return'
                    #clear accepted_orders DataFrame. Will be filled again after settlement this round
                    self.accepted_BE_orders = self.accepted_BE_orders.iloc[0:0]
                else:
                    raise Exception('position to be updated unknown')
                if new_transactions.empty:
                    pass #do nothing. next position.
                else:
                    #make sell orders negative
                    mask = new_transactions['direction'] == 'buy'
                    new_transactions[i] = new_transactions['cleared_quantity'].where(mask,-1*new_transactions['cleared_quantity']).astype('float64')
                    new_transactions[k] = new_transactions['due_amount']
                    new_transactions.set_index(['delivery_day','delivery_time'], inplace=True)
                    #sum (saldo) of trades from the agent per timestamp
                    new_transactions = new_transactions.groupby(level =[0,1]).sum()
                    #add to 'position' column in self.trade_schedule
                    new_trade_schedule[i] = new_trade_schedule[i].add(new_transactions[i], fill_value = 0)
                    #add to 'return' column in self.financial_return
                    new_trade_returns[k] = new_trade_returns[k].add(new_transactions[k], fill_value = 0)
        #overwrite self.trade_schedule
        self.trade_schedule = new_trade_schedule.copy()
        #overwrite self.financial returns
        self.financial_return = new_trade_returns.copy()
        #calculate total schedule. This value can be positive or negative and larger than the sum of all asset pmax.
        self.trade_schedule['total_trade'] = self.trade_schedule[['DA_position','ID_position','RD_position','BE_position','forecast_error']].sum(axis=1)
        #calculate total return.
        self.financial_return['total_return'] = self.financial_return[['DA_return','ID_return','RD_return', 'BE_return','IB_return']].sum(axis=1)

    def set_asset_commit_constraints (self):
        """
        Method:
            Collects redisaptch transactions from last simulation step
            and update a contraint dataframe per asset.

            In case an asset is associated with a redispatch tranaction,
            additional constraints are applicable to the dispatch optimization.

            In case of upward redispatch transaction, the asset is bound to a
            dispatch above the last dispatch schedule + upward redispatch quantity .
            In case of a downward redisaptch, the asset is bound to a dispatch
            below the last disaptch schedule - downward redispatch quantity
        """

        if self.accepted_red_orders.empty:
            #In this case there are not time varying dispatch contraints on assets from redispatch,
            #other than previous contraints in asset.constraint_df.
            #However, the asset.constraint_df needs to be updated with dispatch schedule horizon.
            all_ids = self.assets.index.values
            #update asset.contraint_df
            for i in range(len(all_ids)):
                asset = self.assets.loc[all_ids[i],:].item()
                asset.calc_dispatch_constraints(self.model.schedules_horizon)
        else:
            all_ids = self.assets.index.values
            #update asset.contraint_df
            i=0
            for i in range(len(all_ids)):
                asset = self.assets.loc[all_ids[i],:].item()

                #get redispatch transactions of that asset
                RDM_asset_schedule = self.accepted_red_orders.loc[self.accepted_red_orders['associated_asset'] == asset.assetID,
                                                                  ['delivery_day', 'delivery_time','cleared_quantity','direction']]
                if RDM_asset_schedule.empty:
                    #even though no redisaptch has been cleared for the agent,
                    #contraints are calculated to get the right size of the contraint_df
                    new_red_schedule = self.model.schedules_horizon.copy()
                    asset.calc_dispatch_constraints(new_red_schedule)
                else:
                    new_red_schedule = self.model.schedules_horizon.copy()
                    #make temporary dataframe from accepted bids that can be added to redispatch asset schedule
                    #make buy orders negative to use them for a dispatch reduction (downward)
                    mask = RDM_asset_schedule['direction'] == 'sell'
                    RDM_asset_schedule['cleared_quantity'] = RDM_asset_schedule['cleared_quantity'].where(mask,-RDM_asset_schedule['cleared_quantity'])

                    RDM_asset_schedule.set_index(['delivery_day','delivery_time'], inplace = True)
                    RDM_asset_schedule.sort_index(level = 1, inplace = True)
                    RDM_asset_schedule = RDM_asset_schedule.groupby(level =[0,1]).sum()
                    RDM_asset_schedule.rename(columns = {'cleared_quantity': 'commit'}, inplace = True)

                    #place commitment in schedule horizon format
                    new_red_schedule['commit'] = RDM_asset_schedule['commit'].copy()
                    new_red_schedule['commit'] = new_red_schedule['commit'].loc[self.model.schedules_horizon.index].fillna(value=0).copy()
                    asset.calc_dispatch_constraints(new_red_schedule)
            #remove all accepted redispatch orders from this round.
            self.accepted_red_orders = self.accepted_red_orders.iloc[0:0]


    def portfolio_dispatch(self):
        """
        Method:

        PyPSA used to determine an optimal asset comitment, given the trade position of the agent.
        - the total trade position implemented as 'load'
        - a slack generator with high costs is used to capture the unfeasible dispatch ('open position)
        - if the total trade position is negative it is not captured by the slack generator,
          but cut out in advance from the schedule and added to imbalance.
        - the method also calculates and administerns imbalances and profits of the agent
        - The method furthermore determines the available capacity per asset.

        """

        print('start portfolio optimization of Agent ', self.unique_id)

        #all assets id's of this agent
        all_ids = self.assets.index.values
        must_run_commit= DataFrame()
        if self.unchanged_position == True:
            print('no additional portfolio optimization needed because nothing has changed for this agent')
            #add dispatch cost of current (thus realized) dispatch to bank account
            day,mtu = self.model.clock.calc_timestamp_by_steps(self.model.schedule.steps,0)
            if (day,mtu) in self.financial_return.index:
                self.money += self.financial_return.loc[(day,mtu),'total_dispatch_costs'].copy()
        else:
            #convert model time index to PyPSA snapshot format (no tuples or multiindex)
            snap = DataFrame(index=self.model.schedules_horizon.index)
            snap=snap.reset_index()
            snap['strIndex']=snap['delivery_day'].map(str)+str('_')+snap['delivery_time'].map(str)
            self.commit_model.set_snapshots(snap['strIndex'])

            #add the trade position to network model load.
            #   Note that positive values mean consumption in PyPSA.
            #   Positive trade positions, however, mean long position. Trade schedule is therefore multiplied by -1
            relevant_trades = -self.trade_schedule.loc[self.model.schedules_horizon.index].copy()
            if len(relevant_trades['total_trade'])== len(snap['strIndex']):
                #adjust relevant trades by setting negative values ('generation') on 0
                if (relevant_trades['total_trade']<0).any():
                    relevant_trades['total_trade'] = relevant_trades['total_trade'].where(relevant_trades['total_trade']>=0.0,0)
                    print('Agent{} has a total schedule with values < 0'.format(self.unique_id))
                #make list from series
                commit_lst = list(relevant_trades['total_trade'].fillna(value=0))
            else:
                raise Exception ('there is a issue with the snapshot timestamps and asset trade_schedule timestamps')
            #assigne trade position to commit_model
            self.commit_model.loads_t.p_set['trade_position']=  commit_lst
            #calculate timestamp of last round
            day, MTU=   self.model.clock.calc_timestamp_by_steps(self.model.schedule.steps -1, 0)
            #Assign time variant asset constraints in p.u. of pmax to each asset of agent
            for i in range(len(all_ids)):
                asset = self.assets.loc[all_ids[i],:].item()
                if len(snap['strIndex']) == len(asset.constraint_df.loc[self.model.schedules_horizon.index]):
                    #to be given to pypsa
                    pmin_t = list(asset.constraint_df['p_min_t'].loc[self.model.schedules_horizon.index]/asset.pmax)
                    pmax_t = list(asset.constraint_df['p_max_t'].loc[self.model.schedules_horizon.index]/asset.pmax)

                    #Asset must-run contraints to be added to must_run_commit df (from upward redispatch),
                    #for later use as extra cpontraint in PyPSA optimal power flow
                    if (asset.constraint_df['upward_commit']>0).any():
                        #constraint snapshots
                        up_const = asset.constraint_df[['dispatch_limit', 'p_max_t']].loc[self.model.schedules_horizon.index].loc[
                                        asset.constraint_df['upward_commit']!=0].reset_index()
                        up_const['gen_name'] = asset.assetID
                        up_const ['strIndex']=up_const['delivery_day'].map(str)+str('_')+up_const['delivery_time'].map(str)
                        #make an index from generatorname and snapshot for later use in pypsa constraint method
                        up_const.set_index(['gen_name','strIndex'], inplace=True)
                        must_run_commit = pd.concat([must_run_commit, up_const])
                else:
                    print(asset.assetID)
                    raise Exception ('there is a issue with the snapshot timestamps and asset constraints timestamps')
                self.commit_model.generators_t.p_max_pu[asset.assetID] = pmax_t
                self.commit_model.generators_t.p_min_pu[asset.assetID] = pmin_t

                #ensure that minimum up & downtimes > optimization horizon are corrected
                if asset.min_up_time > len(snap):
                    self.commit_model.generators.min_up_time[asset.assetID] = len(snap)
                if asset.min_down_time > len(snap):
                    self.commit_model.generators.min_down_time[asset.assetID] = len(snap)

                #ensure that the initial disaptch status taken from last dispatch (relevant for start stop costs)
                try:
                    last_dispatch = asset.schedule.loc[(day,MTU), 'commit']
                    if last_dispatch > 0:
                        self.commit_model.generators.initial_status[asset.assetID]=1
                    else:
                        self.commit_model.generators.initial_status[asset.assetID]=0
                except:
                    #in the first step there is no previous dispatch.
                    if not self.model.DA_marketoperator.test_init_dispatch.empty:
                        #test init dispatch
                        self.commit_model.generators.initial_status[asset.assetID]=self.model.DA_marketoperator.test_init_dispatch[asset.assetID]
                    else:
                        #assume 1
                        self.commit_model.generators.initial_status[asset.assetID]=1

            def red_commit_constraint(network, snapshots):
                """this method gives an extra must-run constraint to generators that have been
                  commited to upward redispatch"""
                if must_run_commit.empty:
                    pass
                else:
                    gen_p_bounds = {(gen_sn) : (must_run_commit.loc[gen_sn,'dispatch_limit'],
                                    must_run_commit.loc[gen_sn,'p_max_t'])
                                    for gen_sn in must_run_commit.index.values}
                    red_must_run={}
                    for gen_sn in must_run_commit.index.values:
                        red_must_run[gen_sn] = [[(1, network.model.generator_p[gen_sn])],"><", gen_p_bounds[gen_sn]]
                    l_constraint(network.model, "must_run", red_must_run, list(must_run_commit.index.values))

            #run linear optimal power flow
            try:
                lopf_status= self.commit_model.lopf(self.commit_model.snapshots, extra_functionality = red_commit_constraint,
                                       solver_name= self.model.exodata.solver_name, free_memory={'pypsa'})

            except:
                print(self.commit_model.generators_t.p_max_pu)
                print(self.commit_model.generators_t.p_min_pu)
                import pdb
                pdb.set_trace()

            #process loadflow results to dispatch and trade schedule
            opt_dispatch = self.commit_model.generators_t.p.copy()
            opt_dispatch['long_position']= -opt_dispatch['long_position']

            if opt_dispatch.empty:
                print('Issue with agent ',self.unique_id)
                print(self.trade_schedule)
                raise Exception('optimal power flow did not find a solution')
            #convert index again
            opt_dispatch.index = opt_dispatch.index.str.split(pat='_', expand =True)
            opt_dispatch.index.set_names(['delivery_day','delivery_time'], inplace=True)
            #make inters from index
            opt_dispatch.reset_index(inplace=True)
            opt_dispatch[['delivery_day','delivery_time']] = opt_dispatch[['delivery_day','delivery_time']].astype('int64')
            opt_dispatch.set_index(['delivery_day','delivery_time'], inplace=True)
            #calculate total dispatch (excluding the dummy generator/storage)
            opt_dispatch['total_dispatch'] =opt_dispatch.sum(axis=1)- opt_dispatch['long_position'] - opt_dispatch['short_position']
            self.trade_schedule.loc[self.model.schedules_horizon.index,'total_dispatch'] = opt_dispatch['total_dispatch'].loc[self.model.schedules_horizon.index]

            #positive trade schedule values mean that more is bought than sold (long),
            #negative trade schedule values mean short position (more sold than bought)
            #dispatch positive means injection to grid, negative means consumption
            #total trade schedule + total dispatch = imbalance position.
            # a positive imbalance position is a long imbalance position ->more produced than (net) sold.
            #  a negative imbalance position is a short imbalance position ->less produced than (net) sold
            self.trade_schedule['imbalance_position'].loc[
                    self.model.schedules_horizon.index] =  self.trade_schedule[
                            'total_dispatch'].loc[self.model.schedules_horizon.index
                    ] + self.trade_schedule['total_trade'].loc[
                            self.model.schedules_horizon.index]
            if (self.trade_schedule['imbalance_position'].loc[self.model.schedules_horizon.index]!=0).any():
                print('IMBALANCE of agent: ', self.unique_id)


            #calculate dispatch costs (also devided by 4 as we have 15 minute steps)
            dispatch_cost = opt_dispatch.copy()
            for j in opt_dispatch.loc[:,(opt_dispatch.columns != 'long_position')& (
                    opt_dispatch.columns != 'short_position')& (
                    opt_dispatch.columns != 'total_dispatch')].columns:
                #variable cost
                dispatch_cost['var_cost_'+j] = dispatch_cost[j] * self.commit_model.generators.loc[j,'marginal_cost']/4

                #calculate startstop cost
                startstopcost =Series([1]*len(dispatch_cost),index =dispatch_cost.index).where(dispatch_cost[j] > 0,0)
                #starts=1, stops=-1
                startstopcost=startstopcost - startstopcost.shift(1)
                #take into account initial status
                if (self.commit_model.generators.initial_status[j]==1)&(int(round(dispatch_cost[j].iloc[0]))==0):
                    startstopcost.iloc[0] = -1
                elif (self.commit_model.generators.initial_status[j]==0)&(int(round(dispatch_cost[j].iloc[0]))>0):
                    startstopcost.iloc[0] = 1
                else:
                    startstopcost.iloc[0] = 0
                startstopcost.loc[startstopcost==1] =self.commit_model.generators.start_up_cost[j]
                startstopcost.loc[startstopcost==-1] =self.commit_model.generators.shut_down_cost[j]
                dispatch_cost['fix_cost_'+j] = startstopcost
            dispatch_cost.drop(opt_dispatch.columns, axis=1,inplace = True)
            #calculate total dispatch cost (startstop cost included)
            dispatch_cost['total_dispatch_costs'] = dispatch_cost.sum(axis=1)

            #dispatch costs are by definition negative
            self.financial_return.loc[self.model.schedules_horizon.index,'total_dispatch_costs'] = -dispatch_cost[
                    'total_dispatch_costs'].loc[self.model.schedules_horizon.index].copy()

            #calculate total profit
            self.financial_return['profit'] = self.financial_return[['total_return','total_dispatch_costs']].sum(axis=1)


            #add dispatch cost of current (thus realized) dispatch to bank account
            day,mtu = self.model.clock.calc_timestamp_by_steps(self.model.schedule.steps,0)
            self.money += self.financial_return.loc[(day,mtu),'total_dispatch_costs'].copy()
        #end of if-else unchanged_position == True

        """assign dispatch values to asset schedule and calculate available capacity"""
        i=0
        for i in range(len(all_ids)):
            asset = self.assets.loc[all_ids[i],:].item()
            #enlarge asset schedule time index
            mask =self.model.schedules_horizon.index.isin(asset.schedule.index)
            asset.schedule = asset.schedule.append(self.model.schedules_horizon.loc[~mask,asset.schedule.columns])
            if self.unchanged_position == False: #if unchanged == True this step is skipped
                #add optimal dispatch results to asset schedule, without overwriting the past, i.e. realized dispatch
                mask = asset.schedule.index.isin(opt_dispatch.index)
                asset.schedule['commit'] = asset.schedule['commit'].where(~mask,opt_dispatch[asset.assetID].copy())
                #ensure that no very small values from the solver stay in results
                asset.schedule['commit'] =asset.schedule['commit'].round(0).astype(int)

            #calculate available capacity based on constraint df temp pmax and pmin
            asset.schedule['p_max_t'].loc[self.model.schedules_horizon.index] = asset.constraint_df['p_max_t']
            asset.schedule['p_min_t'].loc[self.model.schedules_horizon.index] = asset.constraint_df['p_min_t']

            asset.schedule['available_up'] = (asset.schedule['p_max_t'] - asset.schedule['commit']).where(
                    (asset.schedule['commit'] >= asset.pmin) & (
                            asset.schedule['p_max_t'] > asset.schedule['commit']) & (
                                    asset.constraint_df['downward_commit'] == 0), 0)
            #available down is also a positive value!!
            asset.schedule['available_down'] = (asset.schedule['commit']-asset.schedule['p_min_t']).where(
                    (asset.schedule['commit'] > asset.schedule['p_min_t']) & (asset.constraint_df['upward_commit'] == 0), 0)

            if ((asset.schedule['available_up'] < 0).any())|((asset.schedule['available_down'] < 0).any()):
                import pdb
                pdb.set_trace()
                raise Exception('available_up and available_down must be values >= 0')
            mask = (self.trade_schedule['imbalance_position'].loc[self.model.schedules_horizon.index] != 0)
            if (mask == True).any():
                #Note: MTU with imbalance are restricted for usual bidding.
                #The market party applies a intraday bidding strategy to these MTU
                asset.schedule['available_up'].loc[self.model.schedules_horizon.index
                          ] = asset.schedule['available_up'].loc[
                                  self.model.schedules_horizon.index].where(~mask, 0)
                asset.schedule['available_down'].loc[self.model.schedules_horizon.index
                          ] = asset.schedule['available_down'].loc[
                                  self.model.schedules_horizon.index].where(~mask, 0)

            #available capacity with contraint that it is <=ramp rate from one MTU to the next
            asset.schedule['ramp_constr_avail_up'] =asset.schedule['available_up'].where(
                    asset.schedule['available_up']<=asset.ramp_limit_up *asset.pmax, asset.ramp_limit_up *asset.pmax)
            asset.schedule['ramp_constr_avail_down'] =asset.schedule['available_down'].where(
                    asset.schedule['available_down']<=asset.ramp_limit_down *asset.pmax, asset.ramp_limit_down *asset.pmax)
            #available capacity with constraint that it is <= remaining ramps considering commited ramp t-1 and t+1
            #Pt - Pt-1
            asset.schedule['commit_ramp_t-1'] = asset.schedule['commit'] - asset.schedule['commit'].shift(1)
            #Pt+1 - Pt
            asset.schedule['commit_ramp_t+1'] = asset.schedule['commit'].shift(-1) - asset.schedule['commit']
            #correct nan of first and last time stamp. Assume commit ramp of 0.
            asset.schedule['commit_ramp_t-1'].fillna(value = 0, inplace=True)
            asset.schedule['commit_ramp_t+1'].fillna(value = 0, inplace=True)
            asset.schedule['rem_ramp_constr_avail_up'] = asset.schedule.apply(
                    lambda x: min(- x['commit_ramp_t-1'] + asset.ramp_limit_up *asset.pmax,
                                  x['commit_ramp_t+1'] + asset.ramp_limit_up *asset.pmax,
                                  x['available_up']), axis =1)
            asset.schedule['rem_ramp_constr_avail_down'] = asset.schedule.apply(
                    lambda x: min(x['commit_ramp_t-1'] + asset.ramp_limit_down * asset.pmax,
                                  - x['commit_ramp_t+1'] + asset.ramp_limit_up * asset.pmax,
                                  x['available_down']), axis =1)
            #if the portfolio dispatch optimization allows start stop changes >ramp limits, negative remaining ramps are to be avoided
            asset.schedule['rem_ramp_constr_avail_up']=asset.schedule['rem_ramp_constr_avail_up'].where(asset.schedule['rem_ramp_constr_avail_up']>=0, 0)
            asset.schedule['rem_ramp_constr_avail_down']=asset.schedule['rem_ramp_constr_avail_down'].where(asset.schedule['rem_ramp_constr_avail_down']>=0, 0)


    def place_RD_orders(self):
        """
        Method: determine order quantity and order price and other order attributes.
        Then make an order message and send it to the redispatch order book.

        Note: order messages contain many orders. To reduce computation time, the order messages
        are composed from lists, instead of manipulating DataFrames. This makes
        the sorting of the content of the lists (i.e. the filling of the lists) crucial.
        """
        #check if redipatch is part of simulation task
        if self.model.exodata.sim_task['run_RDM[y/n]']=='n':
            print('Agent {}:no redispatch in simlulation task'.format(self.unique_id))
        elif self.model.RD_marketoperator.rules['order_types']== 'IDCONS_orders':
            #no dediacted redispatch orders allowed. Redispatch via IDCONS orders on intraday market.
            pass
        elif self.strategy['RDM_quantity']=='None':
           #this agent does not participate in the redispatch market
           print('Agent {}:does not participate in redispatch market'.format(self.unique_id))
           pass
        else:
           print("Agent {} makes redispatch bids".format(self.unique_id))

           #first delete all own redispatch orders from previous round from orderbook
           self.model.red_obook.delete_orders(agent_id_orders = self.unique_id)
           #lists per order attribute
           asset_location_lst = []
           agentID_lst = []
           assetID_lst = []
           init_lst = []
           direction_lst = []
           ordertype_lst = []
           qty_lst=[]
           price_lst=[]
           day_lst=[]
           mtu_lst=[]
           delivery_duration =[]

           gate_closure_MTU = self.model.RD_marketoperator.gate_closure_time
           #delivery time lists
           dayindex = list(self.model.schedules_horizon.index.get_level_values(0))[gate_closure_MTU:]
           timeindex = list(self.model.schedules_horizon.index.get_level_values(1))[gate_closure_MTU:]

           #order initiation for up to 1000 agents (shows when order is initiated within one simulation step)
           init = self.step_rank/1000 + self.model.clock.get_MTU()

           #get all assets of that market party
           all_ids = self.assets.index.values
           for i in range(len(all_ids)):
               a = self.assets.loc[all_ids[i],:].item()
               #store current asset schedule (excluding the past) for calculation of dispatch contraints from redispatch
               a.schedule_at_redispatch_bidding= a.schedule.loc[self.model.schedules_horizon.index].copy()
               if self.strategy['RDM_timing']=='instant':
                   #list containing three lists (delivery day, delivery time, delivery duration)
                   #used for block orders i.e. delivery duration > 1 MTU
                   startblocks =[[],[],[]]
                   stopblocks = [[],[],[]]
                   price_start_lst = []
                   price_stop_lst= []
                   if self.strategy['RDM_quantity']=='random':
                       #'random' quantity strategy determines small random quantity
                       qty_up_lst = self.small_random_quantity(a.schedule[['available_up','p_max_t']].loc[
                                    self.model.schedules_horizon.index].fillna(0))
                       qty_down_lst = self.small_random_quantity(a.schedule[['available_down','p_max_t']].loc[
                                    self.model.schedules_horizon.index].fillna(0))
                   elif (self.strategy['RDM_quantity']=='all_operational')|(
                           self.strategy['RDM_quantity']=='all_plus_startstop'):
                       #'all_operational' includes all available capacity, excluding start stop.
                       #'all__plus_start_stop' means all available capacity.
                       qty_up_lst = list(a.schedule['available_up'].loc[
                                    self.model.schedules_horizon.index].fillna(0).astype(int))
                       qty_down_lst = list(a.schedule['available_down'].loc[
                                    self.model.schedules_horizon.index].fillna(0).astype(int))
                   elif self.strategy['RDM_quantity']=='not_offered_plus_startstop':
                       #' not_offered_plus start_stop' means that offered quantities on intra-day market is deducted.
                       #get the position from offered on IDM
                       buy_position, sell_position =self.model.IDM_obook.get_offered_position(associated_asset=a.assetID)
                       #deduct offered position from availble capacity
                       if a.schedule['available_up'].loc[
                                    self.model.schedules_horizon.index].index.isin(sell_position.index).any():
                           qty_up_lst = a.schedule['available_up'].loc[
                                        self.model.schedules_horizon.index].fillna(0).to_frame().join(
                                        -sell_position).sum(axis=1).astype(int).copy().values
                           #correct negative values. Reason is a portfolio dispatch optimization after IDM clearing.
                           qty_up_lst[qty_up_lst < 0] = 0
                           qty_up_lst = list(qty_up_lst)
                       else:
                           #sell_position empty or outside schedule
                           qty_up_lst = list(a.schedule['available_up'].loc[
                                    self.model.schedules_horizon.index].fillna(0).astype(int))
                       if a.schedule['available_down'].loc[
                                    self.model.schedules_horizon.index].index.isin(buy_position.index).any():
                           qty_down_lst = a.schedule['available_down'].loc[
                                            self.model.schedules_horizon.index].fillna(0).to_frame().join(
                                            -buy_position).sum(axis=1).astype(int).copy().values
                           qty_down_lst[qty_down_lst < 0] = 0
                           qty_down_lst = list(qty_down_lst)
                       else:
                           qty_down_lst = list(a.schedule['available_down'].loc[
                                    self.model.schedules_horizon.index].fillna(0).astype(int))

                   else:
                       raise Exception('redispatch quantity strategy not known')
                   if (self.strategy['RDM_quantity']=='all_plus_startstop')|(
                           self.strategy['RDM_quantity']=='not_offered_plus_startstop'):
                       #add startblock and stop blocks for strategies involving start and stop capacity.
                       av_cap = pd.concat([a.schedule.loc[self.model.schedules_horizon.index[gate_closure_MTU:]],
                                    a.constraint_df.loc[self.model.schedules_horizon.index[
                                    gate_closure_MTU:],'upward_commit'].copy()], axis=1)
                       startblocks, stopblocks = self.start_stop_blocks (av_cap, \
                               a.pmin, a.min_up_time, a.min_down_time,a.assetID)
                       #in case the pricing strategy contains no start stop markup, prices are per default srmc
                       if startblocks:
                           price_start_lst = [int(a.srmc)] * len(startblocks[0])
                       if stopblocks:
                           price_stop_lst= [int(a.srmc)] * len(stopblocks[0])

                   #ORDER PRICING

                   #short-run marginal costs are fundamental price to which mark-ups are added
                   price_up_lst = [int(a.srmc)] * len(dayindex)
                   price_down_lst = [int(a.srmc)] * len(dayindex)

                   def add_markups_to_price_list(price_up_lst, price_down_lst, markup_up, markup_down):

                       if len(markup_up)==len(price_up_lst):
                           price_up_lst = [i[0] +i[1] for i in zip(markup_up,price_up_lst)]
                       else:
                           import pdb
                           pdb.set_trace()
                       if len(markup_down)==len(price_down_lst):
                           price_down_lst= [i[0] +i[1] for i in zip(markup_down, price_down_lst)]
                       else:
                           import pdb
                           pdb.set_trace()
                       return (price_up_lst,price_down_lst)

                   if self.strategy['RDM_pricing'] =='srmc':
#                       price_up_lst = [int(a.srmc)] * len(dayindex)
#                       price_down_lst = [int(a.srmc)] * len(dayindex)
                       price_start_lst = [int(a.srmc)] * len(startblocks[0])
                       price_stop_lst= [int(a.srmc)] * len(stopblocks[0])
                   elif (self.strategy['RDM_pricing']=='all_markup')|(
                           self.strategy['RDM_pricing']=='opportunity_markup'):
                       #opportunity markup
                       opportunity_markup_up = self.opportunity_markup(
                               direction='upward', of_quantity = qty_up_lst,
                               asset = a,success_assumption = 'offered_quantity')
                       opportunity_markup_down = self.opportunity_markup(
                               direction='downward', of_quantity = qty_down_lst,
                               asset = a, success_assumption = 'offered_quantity')
                       #mark-ups added to price list
                       price_up_lst, price_down_lst = add_markups_to_price_list(
                               price_up_lst, price_down_lst,
                               opportunity_markup_up[gate_closure_MTU:], opportunity_markup_down[gate_closure_MTU:])
                   if (self.strategy['RDM_pricing']=='all_markup')|(
                           self.strategy['RDM_pricing']=='startstop_markup'):
                      #prices for start and stop blocks
                      start_markup = self.startstop_markup(
                              direction = 'upward', of_quantity = startblocks, asset = a,
                              gct = gate_closure_MTU, partial_call = False)
                      stop_markup = self.startstop_markup(
                             direction = 'downward', of_quantity = stopblocks,
                             asset = a, gct = gate_closure_MTU, partial_call = False)
                      price_start_lst, price_stop_lst = add_markups_to_price_list(
                               [a.srmc] * len (start_markup), [a.srmc] * len (stop_markup),
                              start_markup, stop_markup)

                   if (self.strategy['RDM_pricing']=='all_markup')|(
                           self.strategy['RDM_pricing']=='ramping_markup'):
                      #ramping mark-up
                      ramp_markup_up =self.ramping_markup(direction='upward',
                                                      of_quantity = qty_up_lst,
                                                      asset = a)
                      ramp_markup_down =self.ramping_markup(direction='downward',
                                                      of_quantity = qty_down_lst,
                                                      asset = a)
                      #mark-ups added to price list
                      price_up_lst, price_down_lst = add_markups_to_price_list(
                               price_up_lst, price_down_lst,
                               ramp_markup_up[gate_closure_MTU:], ramp_markup_down[gate_closure_MTU:])
                   if (self.strategy['RDM_pricing']=='all_markup')|((
                           self.strategy['RDM_pricing']=='double_scoring_markup')&(
                                   self.strategy['RDM_quantity']!='not_offered_plus_startstop')):
                      #mark-up for double scoring risk on two markets
                      doublescore_markup_up =self.doublescore_markup(direction='upward',
                                                      of_quantity = qty_up_lst,
                                                      asset = a)
                      doublescore_markup_down =self.doublescore_markup(direction='downward',
                                                      of_quantity = qty_down_lst,
                                                      asset = a)
                      #mark-ups added to price list
                      price_up_lst, price_down_lst = add_markups_to_price_list(
                               price_up_lst, price_down_lst,
                               doublescore_markup_up[gate_closure_MTU:], doublescore_markup_down[gate_closure_MTU:])


                   #length of attribute lists
                   length =len(dayindex) * 2 + len(startblocks[0]) + len(stopblocks[0])
                   #per order attribute a list over all agent assets is built
                   asset_location_lst += [a.location] * length
                   agentID_lst += [self.unique_id]* length
                   assetID_lst += [a.assetID]* length
                   init_lst += [init] * length
                   #note that the sorting of lists is important
                   direction_lst += ['buy'] * len(dayindex) + ['sell'] *len(dayindex) +['sell'] * len(startblocks[0])+['buy'] * len(stopblocks[0])
                   ordertype_lst += ['redispatch_supply']* length
                   delivery_duration += [1] * len(dayindex) * 2 + startblocks[2] + stopblocks[2]
                   qty_lst += qty_down_lst[gate_closure_MTU:] + qty_up_lst[gate_closure_MTU:] + [a.pmin] *(
                           len(startblocks[0]) + len(stopblocks[0]))
                   price_lst += price_down_lst + price_up_lst + price_start_lst + price_stop_lst
                   day_lst += dayindex * 2 + startblocks[0] + stopblocks[0]
                   mtu_lst += timeindex * 2 + startblocks[1] + stopblocks[1]
               else:
                   raise Exception ('redispatch timing strategy not known')

           orders = DataFrame()
           #NOTE: important to have same ranking of columns and values
           columns=['agent_id','associated_asset','delivery_location',
                                  'quantity','price', 'delivery_day','delivery_time',
                                  'order_type','init_time', 'direction', 'delivery_duration']

           values =[agentID_lst,assetID_lst, asset_location_lst,
                               qty_lst, price_lst, day_lst,mtu_lst,
                               ordertype_lst, init_lst, direction_lst, delivery_duration]

           #make dataframe per column to maintain datatype of lists (otherwise seen as objects by pandas)
           for i in range(len(columns)):
                orders[columns[i]]=values[i]
           #remove 0 MW rows
           orders = orders.loc[orders['quantity']!=0].copy()

           if not orders.empty:
               # insert order IDs (must be at second last column because of many dependencies)
               orders.insert(loc = len(orders.columns)-2,column='order_id',
                             value =list(range(self.ordercount, self.ordercount + len(orders))) )
               orders['order_id']=orders['agent_id'] + orders['associated_asset'] + orders['order_id'].astype(str)

               #order count for the order ID
               self.ordercount += len(orders)
               order_message = OrderMessage(orders)
               self.model.red_obook.add_order_message(order_message)


    def place_ID_orders(self):
        """
        Method: determine order quantity and order price and other order attributes.
        Then make an order message and send it to the intra-day order book.

        Note: order messages contain many orders. To reduce computation time, the order messages
        are composed from lists, instead of manipulating DataFrames. This makes
        the sorting of the content of the lists (i.e. the filling of the lists) crucial.
        """
        #check if redipatch is part of simulation task
        if self.model.exodata.sim_task['run_IDM[y/n]']=='n':
            print('Agent {}:no intraday market in simlulation task'.format(self.unique_id))
        elif self.strategy['IDM_quantity']=='None':
           #this agent does not participate in the redispatch market
           print('Agent {}:does not participate in redispatch market'.format(self.unique_id))
           pass
        else:
            #first delete all ID orders from previous round from orderbook
            self.model.IDM_obook.delete_orders(agent_id_orders = self.unique_id)

            print("Agent {} makes ID bids".format(self.unique_id))
            #lists per order attribute
            asset_location_lst = []
            agentID_lst = []
            assetID_lst = []
            init_lst = []
            direction_lst = []
            ordertype_lst = []
            qty_lst=[]
            price_lst=[]
            day_lst=[]
            mtu_lst=[]
            delivery_duration =[]
            gate_closure_MTU = self.model.ID_marketoperator.gate_closure_time
            dayindex = list(self.model.schedules_horizon.index.get_level_values(0))[gate_closure_MTU:]
            timeindex = list(self.model.schedules_horizon.index.get_level_values(1))[gate_closure_MTU:]
            #order initiation for up to 1000 agents (shows when order is initiated within one simulation step)
            init = self.step_rank/1000 +self.model.clock.get_MTU()

            otype = 'intraday_limit_order'
            #get all assets of that market party
            all_ids = self.assets.index.values

            if (self.strategy['IDM_quantity']=='random_plus_cond_startstop')|(
                    self.strategy['IDM_quantity']=='all_plus_cond_startstop'):
                #quantity strategies with conditional start stop capacity considers
                #redispatch activation in previous simulation step. This strategy may be required for IDCONS.
                prev_day, prev_mtu= self.model.clock.calc_timestamp_by_steps(self.model.schedule.steps -1, 0)
                if not self.model.red_obook.redispatch_demand_upward_all_df.empty:
                    #get redispatch demand from previous step
                    red_demand_upward =self.model.red_obook.redispatch_demand_upward_all_df.loc[
                                    self.model.red_obook.redispatch_demand_upward_all_df['offer_daytime']==(prev_day, prev_mtu)].copy()
                    red_demand_upward = red_demand_upward.groupby(by=['delivery_day', 'delivery_time']).first()
                else:
                    red_demand_upward= DataFrame(columns=['quantity'])
                if not self.model.red_obook.redispatch_demand_downward_all_df.empty:
                    red_demand_downward =self.model.red_obook.redispatch_demand_downward_all_df.loc[
                            self.model.red_obook.redispatch_demand_downward_all_df['offer_daytime']==(prev_day, prev_mtu)]
                    red_demand_downward = red_demand_downward.groupby(by=['delivery_day', 'delivery_time']).first()
                else:
                    red_demand_downward= DataFrame(columns=['quantity'])
                #concat upward and downward demand
                red_demand =pd.concat([red_demand_upward['quantity'],red_demand_downward['quantity']], axis=1)
            for i in range(len(all_ids)):
                a = self.assets.loc[all_ids[i],:].item()
                #the following lists are required for blockorders
                startblocks =[[],[],[]]
                stopblocks = [[],[],[]]
                price_start_lst = []
                price_stop_lst= []
                if self.strategy['IDM_timing']=='instant':
                    if (self.strategy['IDM_quantity']=='random')|(
                            self.strategy['IDM_quantity']=='random_plus_cond_startstop'):
                        #'random' quantity strategy determines small random quantity
                        qty_up_lst = self.small_random_quantity(a.schedule[['available_up','p_max_t']].loc[
                                    self.model.schedules_horizon.index].fillna(0))
                        qty_down_lst = self.small_random_quantity(a.schedule[['available_down','p_max_t']].loc[
                                    self.model.schedules_horizon.index].fillna(0))
                    elif (self.strategy['IDM_quantity']=='all_operational')|(
                            self.strategy['IDM_quantity']=='all_plus_cond_startstop'):
                        qty_up_lst = list(a.schedule['available_up'].loc[
                                    self.model.schedules_horizon.index].fillna(0).astype(int))
                        qty_down_lst = list(a.schedule['available_down'].loc[
                                    self.model.schedules_horizon.index].fillna(0).astype(int))
                    else:
                        raise Exception('IDM quantity strategy not known')
                    if (self.strategy['IDM_quantity']=='random_plus_cond_startstop')|(
                            self.strategy['IDM_quantity']=='all_plus_cond_startstop'):
                        #'all_operational' includes all available capacity, excluding start stop.
                        #'all__plus_cond_start_stop' means all available capacity including start stop, in case redispatch was activated in the last round.
                        # This conditional start stop strategy is relevant for IDCONS-style (redispatch via intraday market) market designs.

                        #store current asset schedule (excluding the past) for calculation of dispatch contraints from redispatch
                        a.schedule_at_redispatch_bidding= a.schedule.loc[self.model.schedules_horizon.index].copy()
                        av_cap = pd.concat([a.schedule.loc[self.model.schedules_horizon.index[gate_closure_MTU:]],
                                                           a.constraint_df.loc[self.model.schedules_horizon.index[
                                                                   gate_closure_MTU:],'upward_commit'].copy()], axis=1)

                        if not red_demand.empty:
                            #startblocks are only calculated if demand in previous step is not empty
                            startblocks, stopblocks = self.start_stop_blocks (av_cap, \
                               a.pmin, a.min_up_time, a.min_down_time,a.assetID)
                            #delete blocks that do not overlap with redispatch demand
                            for i in range(len(startblocks[0])-1,-1,-1):
                                #reverse range is used to delete from list without changing index
                                days, mtus = self.model.clock.calc_delivery_period_range(
                                               startblocks[0][i],
                                               startblocks[1][i],
                                               startblocks[2][i])
                                #add start day and MTU to the lists
                                days = [startblocks[0][i]]+days
                                mtus = [startblocks[1][i]]+mtus

                                if red_demand.loc[red_demand.index.isin(list(zip(days,mtus)))].empty:
                                    #remove from lists, because available startblock
                                    #does not overlap with previous redisaptch demand
                                    del startblocks[0][i]
                                    del startblocks[1][i]
                                    del startblocks[2][i]
                            for i in range(len(stopblocks[0])-1,-1,-1):
                                #reverse range is used to delete from list without changing index
                                days, mtus = self.model.clock.calc_delivery_period_range(
                                               stopblocks[0][i],
                                               stopblocks[1][i],
                                               stopblocks[2][i])
                                #add start day and MTU to the lists
                                days = [stopblocks[0][i]]+days
                                mtus = [stopblocks[1][i]]+mtus
                                if red_demand.loc[red_demand.index.isin(list(zip(days,mtus)))].empty:
                                    #remove from lists, because available stopblock
                                    #does not overlap with previous redisaptch demand
                                    del stopblocks[0][i]
                                    del stopblocks[1][i]
                                    del stopblocks[2][i]
                        otype = 'IDCONS_order'

                    #ORDER PRICING
                    #short-run marginal costs are fundamental price to which mark-ups are added
                    price_up_lst = [int(a.srmc)] * len(dayindex)
                    price_down_lst = [int(a.srmc)] * len(dayindex)

                    def add_markups_to_price_list(price_up_lst, price_down_lst, markup_up, markup_down):
                        if len(markup_up)==len(price_up_lst):
                           price_up_lst = [i[0] +i[1] for i in zip(markup_up,price_up_lst)]
                        else:
                            import pdb
                            pdb.set_trace()
                        if len(markup_down)==len(price_down_lst):
                           price_down_lst= [i[0] +i[1] for i in zip(markup_down, price_down_lst)]
                           #aa=[i[0] +i[1] for i in zip(opportunity_markup_down, price_down_lst)]
                        else:
                            import pdb
                            pdb.set_trace()
                        return (price_up_lst, price_down_lst)

                    if self.strategy['IDM_pricing'] == 'srmc+-1':
                        #Pricing strategy with fixed 1 Eur mark-up
                        price_up_lst = [int(a.srmc)+1] * len(dayindex)
                        price_down_lst = [int(a.srmc)-1] * len(dayindex)
                    elif (self.strategy['IDM_pricing'] == 'marginal_orderbook_strategy')|(
                            self.strategy['IDM_pricing'] == 'marg_obook_plus_startstop_plus_partialcall'):
                        #opportunity mark-ups
                        opportunity_markup_up = self.opportunity_markup(
                                direction='upward', of_quantity = qty_up_lst,
                                asset = a,success_assumption = 'offered_quantity')
                        opportunity_markup_down = self.opportunity_markup(
                                direction='downward', of_quantity = qty_down_lst,
                                asset = a, success_assumption = 'offered_quantity')
                        #add opportunity mark-up to fundamental costs (aka indifference price)
                        price_up_lst, price_down_lst = add_markups_to_price_list(
                               price_up_lst, price_down_lst,
                               opportunity_markup_up[gate_closure_MTU:], opportunity_markup_down[gate_closure_MTU:])


                        if (len(price_up_lst) > 0) & (len(price_down_lst) > 0):
                            #prices including opportunity and intraday 'open order book' mark up.
                            #   Note: price list is enlarged, as mark_up method needs to work with schedules_horizon
                            #   (not considering gate closure time)
                            price_up_lst = self.intraday_markup([0]*gate_closure_MTU + price_up_lst, 'sell')
                            price_down_lst = self.intraday_markup([0]*gate_closure_MTU + price_down_lst, 'buy')

                            #slice list to consider gate closure time
                            price_up_lst = price_up_lst[gate_closure_MTU:]
                            price_down_lst = price_down_lst[gate_closure_MTU:]

                        if self.strategy['IDM_pricing'] == 'marg_obook_plus_startstop_plus_partialcall':
                            #start and stop capacity prices for IDCONS include a partial call markup
                            #(assuming that also start stop offers could be partially cleared in line with limit orders)
                            start_markup = self.startstop_markup(
                                    direction = 'upward', of_quantity = startblocks,
                                    asset = a, gct = gate_closure_MTU, partial_call=True)
                            stop_markup = self.startstop_markup(
                                    direction = 'downward', of_quantity = stopblocks,
                                    asset = a, gct = gate_closure_MTU, partial_call=True)
                            price_start_lst, price_stop_lst = add_markups_to_price_list(
                               [a.srmc] * len (start_markup), [a.srmc] * len (stop_markup),
                              start_markup, stop_markup)
                        if self.strategy['IDM_quantity']=='all_plus_cond_startstop':
                            #Ramping markups are only required for strategies with larget ('all') quanities.
                            #Small random quantities do not need a ramping mark-up, as no additional risk is imposed.
                            ramp_markup_up =self.ramping_markup(direction='upward',
                                                          of_quantity = qty_up_lst,
                                                          asset = a)
                            ramp_markup_down =self.ramping_markup(direction='downward',
                                                          of_quantity = qty_down_lst,
                                                          asset = a)
                            price_up_lst, price_down_lst = add_markups_to_price_list(
                               price_up_lst, price_down_lst,
                               ramp_markup_up[gate_closure_MTU:], ramp_markup_down[gate_closure_MTU:])
                    else:
                        raise Exception ('IDM pricing strategy not known')
                    length =len(dayindex) * 2 + len(startblocks[0]) + len(stopblocks[0])
                    #per order attribute a list over all agent assets is built
                    asset_location_lst += [a.location] * length
                    agentID_lst += [self.unique_id]* length
                    assetID_lst += [a.assetID]* length
                    init_lst += [init] * length
                    direction_lst += ['buy'] * len(dayindex) + ['sell'] *len(dayindex) +['sell'] * len(startblocks[0])+['buy'] * len(stopblocks[0])
                    ordertype_lst += [otype]* length
                    delivery_duration += [1] * len(dayindex) * 2 + startblocks[2] + stopblocks[2]
                    qty_lst += qty_down_lst[gate_closure_MTU:] + qty_up_lst[gate_closure_MTU:] + [a.pmin] *(
                            len(startblocks[0]) + len(stopblocks[0]))
                    price_lst += price_down_lst + price_up_lst + price_start_lst + price_stop_lst
                    day_lst += dayindex * 2 + startblocks[0] + stopblocks[0]
                    mtu_lst += timeindex * 2 + startblocks[1] + stopblocks[1]
                else:
                    raise Exception('IDM timing strategy not known')
                #end intra-day orders attributes from available asset capacity

            #place intraday orders for the imbalance position of the agent (i.e. not from asset capacity)
            #this section is actually driven by the imbalance strategy of the agent.
            if (self.trade_schedule['imbalance_position'].fillna(value=0).astype(int)!=0).any():

                imb = DataFrame(columns=['imbalance_position','direction','price'])
                imb['imbalance_position'] = self.trade_schedule['imbalance_position'].fillna(value=0)
                imb = imb.loc[imb['imbalance_position'] !=0].copy()
                #filter out all imbalance of past delivery MTUs inlcuding gate closure time (because it is too late)
                mask= imb.index.isin(self.model.schedules_horizon.iloc[gate_closure_MTU:].index)
                imb = imb.loc[mask].copy()
                if imb.empty:
                    #no orders needed
                    pass
                else:
                    print ('making IDM orders to mitigate imbalances')
                    imb['direction'] = None
                    imb['price'] = None

                    #try to sell for a long imbalance position
                    imb.loc[imb['imbalance_position']>0, 'direction']= 'sell'
                    #try to buy for a short imbalance position
                    imb.loc[imb['imbalance_position']<0, 'direction']= 'buy'

                    #the imbalance risk price is an maximum price a agent would accept for a MTU
                    #imbalance price for buy is positive IB risk price, for sell is negative IB risk price
                    imb.loc[imb['imbalance_position']>0, 'price']= -self.imbalance_risk_price
                    imb.loc[imb['imbalance_position']<0, 'price']= self.imbalance_risk_price
                    if (self.strategy['IBM_quantity']=='random'):
                        #IBM quantity strategy 'random' assumes a patient agent only slowly trading the open positions

                        #make negative (short) positions positive buy quantity values
                        #price is only provided to have dataframe
                        imb.loc[imb['imbalance_position']>0, 'imbalance_position']= self.small_random_quantity(
                                imb.loc[imb['imbalance_position']>0, ['imbalance_position','price']])
                        imb.loc[imb['imbalance_position']<0, 'imbalance_position']= self.small_random_quantity(
                                imb.loc[imb['imbalance_position']<0, ['imbalance_position','price']].abs())
                        qty_lst += list(imb['imbalance_position'].astype(int))
                    elif (self.strategy['IBM_quantity']=='all'):
                        #IBM quantity strategy 'all' assumes an impatient agent, who quickly trades all open positions
                        # make negative (short) positions positive buy quantity values
                        imb['imbalance_position']= imb['imbalance_position'].abs().astype(int)
                        qty_lst += list(imb['imbalance_position'])

                    elif (self.strategy['IBM_quantity']=='impatience_curve'):
                        """ agents with imbalance provide partly market and partly limit orders,
                          depending on the time before delivery"""
                        #impatience curve
                        imp_curve= DataFrame()

                        #this impatient curve is shaped like cumulated ID trade quantity before delivery in NL (2016)
                        #other impatience curves may be provided here. To do: make this a parameter.
                        imp_curve['mtu_before_delivery'] = [8,12,16,20,24,28,32,36,40,2*96]
                        imp_curve['offer_share_market_orders'] =[1,0.85,0.7,0.5,0.4,0.3,0.25,0.2,0.15,0.1]
                        #add new columns for market order (MO) quantity and limit order (LO) quantity
                        imb['MO_vol']=None
                        imb['LO_vol']=None
                        for i in range(len(imp_curve)):
                            #select MTUs per imb_curve bin
                            if i == 0:
                                lt_idx = self.model.schedules_horizon.index.values[:imp_curve['mtu_before_delivery'].iloc[i]]
                            else:
                                lt_idx =  self.model.schedules_horizon.index.values[imp_curve[
                                        'mtu_before_delivery'].iloc[i-1]:imp_curve['mtu_before_delivery'].iloc[i]]
                            #multiply imbalance quantity with respective respective share to be placed as market orders and limit orders
                            imb.loc[imb.index.isin(lt_idx), 'MO_vol']=imb.loc[imb.index.isin(lt_idx), 'imbalance_position'
                                    ] *imp_curve['offer_share_market_orders'].iloc[i]
                            imb.loc[imb.index.isin(lt_idx), 'LO_vol']=imb.loc[imb.index.isin(lt_idx), 'imbalance_position'
                                    ] *(1-imp_curve['offer_share_market_orders'].iloc[i])
                        #make a limit order quantity list (must be in shape of schedules_horizon)
                        imb_LO_vol =self.model.schedules_horizon.copy()
                        #add imbalance quantity to be placed as limit order
                        imb_LO_vol['commit'] = imb_LO_vol['commit'].add(imb['LO_vol'], fill_value=0)
                        #Limit order quantity is reduced to small random strategy (to hide the total open position in the open orderbook)
                        imb_LO_vol['small_random_limit_order']=self.small_random_quantity(
                                imb_LO_vol['commit'].abs().astype(int).to_frame())
                        qty_down_lst =list(imb_LO_vol['small_random_limit_order'].abs().where(imb_LO_vol['commit'] < 0, 0).astype(int))
                        qty_up_lst = list(imb_LO_vol['small_random_limit_order'].where(imb_LO_vol['commit'] > 0, 0).astype(int))
                        #add  to vol attribute lists (attention: market orders are added before limit orders to list)
                        qty_lst += list(imb['MO_vol'].abs().astype(int)) + qty_down_lst[gate_closure_MTU:] + qty_up_lst[gate_closure_MTU:]
                    else:
                        raise Exception('imbalance quantity strategy not known')

                    if self.strategy.loc['IBM_pricing']=='market_order_strategy':
                        #strategy: all open positions are placed as market orders
                        ordertype_lst += ['market_order'] * len(imb)
                        price_lst +=list(imb['price'])
                    elif self.strategy.loc['IBM_pricing']=='marginal_orderbook_strategy':
                        #strategy takes the expected imbalance price as srmc price and applies the intra-day 'open order book' pricing strategy.
                        ordertype_lst += ['intraday_limit_order'] * len(imb)
                        imb['eIBP_long'] = self.intraday_markup(list(self.model.rpt.eIBP['expected_IBP_long']), 'sell')[gate_closure_MTU:]
                        imb['eIBP_short'] = self.intraday_markup(list(self.model.rpt.eIBP['expected_IBP_short']), 'buy')[gate_closure_MTU:]
                        price_lst += list(imb['eIBP_long'].where(imb['direction']=='sell',imb['eIBP_short']))
                    elif self.strategy.loc['IBM_pricing']=='impatience_curve':
                        ordertype_lst += ['market_order'] * len(imb)
                        #the limit orders are placed with intra-day mark-ups,
                        #which take the expected imbalance price as srmc price.
                        #the market orders have the administrative high imbalance price.
                        price_up_lst = self.intraday_markup(list(self.model.rpt.eIBP['expected_IBP_long']), 'sell')
                        price_down_lst = self.intraday_markup(list(self.model.rpt.eIBP['expected_IBP_short']), 'buy')
                        #slice list to consider gate closure time
                        price_up_lst = price_up_lst[gate_closure_MTU:]
                        price_down_lst = price_down_lst[gate_closure_MTU:]
                        #first imbalance risk price is added for market orders (imb['price']), then prices for limit orders are added
                        price_lst +=list(imb['price']) + price_down_lst + price_up_lst
                    else:
                        raise Exception('imbalance pricing strategy not known')

                    #add other order attributes
                    direction_lst += list(imb['direction'])
                    day_lst += list(imb.index.get_level_values(0))
                    mtu_lst += list(imb.index.get_level_values(1))
                    asset_location_lst += ['anywhere'] * len(imb)
                    agentID_lst += [self.unique_id] * len(imb)
                    assetID_lst += ['imbalance'] * len(imb)
                    init_lst += [init] * len(imb)
                    delivery_duration += [1] * len(imb)
                    if (self.strategy['IBM_quantity']=='impatience_curve'):
                        #add limit orders to attribute lists
                        length =len(dayindex) * 2
                        asset_location_lst += ['anywhere'] * length
                        agentID_lst += [self.unique_id]* length
                        assetID_lst += ['imbalance']* length
                        init_lst += [init] * length
                        direction_lst += ['buy'] * len(dayindex) + ['sell'] *len(dayindex)
                        ordertype_lst += ['intraday_limit_order']* length
                        delivery_duration += [1] * length
                        day_lst += dayindex * 2
                        mtu_lst += timeindex * 2

            #make order dataframe from lists
            orders = DataFrame()
            columns=['agent_id','associated_asset','delivery_location',
                     'quantity','price', 'delivery_day','delivery_time',
                     'order_type','init_time', 'direction' , 'delivery_duration']
            values = [agentID_lst,assetID_lst, asset_location_lst,
                      qty_lst, price_lst, day_lst, mtu_lst,
                      ordertype_lst, init_lst, direction_lst, delivery_duration]

            #make dataframe per column to maintain datatype of lists (otherwise seen as objects by pandas)
            for i in range(len(columns)):
                orders[columns[i]]=values[i].copy()
            #remove 0 MW rows
            orders = orders.loc[orders['quantity']!=0].copy()

            if not orders.empty:
                # insert order IDs (must be at second last column because of many dependencies)
                orders.insert(loc = len(orders.columns)-2,column='order_id',
                              value =list(range(self.ordercount, self.ordercount + len(orders))) )
                orders['order_id']=orders['agent_id'] + orders['associated_asset'] + orders['order_id'].astype(str)

                #order count for the order ID
                self.ordercount += len(orders)
                order_message = OrderMessage(orders.copy())
                self.model.IDM_obook.add_order_message(order_message)
                if not (orders['order_type']== 'IDCONS_order').empty:
                    #In case of IDCONS orders these orders also need to be stored in the redispatch orderbook
                    #for redispatch statistics
                    self.model.red_obook.add_order_message(OrderMessage(orders.loc[
                            orders['order_type']== 'IDCONS_order', orders.columns]))

                #make ID bid triggerst immediate ID clearing
                self.model.ID_marketoperator.clear_intraday(for_agent_id =self.unique_id)

    def place_BE_orders(self):
        """
        Method: determine order quantity and order price and other order attributes.
        Then make an order message and send it to the intra-day order book.

        Note: order messages contain many orders. To reduce computation time, the order messages
        are composed from lists, instead of manipulating DataFrames. This makes
        the sorting of the content of the lists (i.e. the filling of the lists) crucial.
        """
        if self.model.exodata.sim_task['run_BEM[y/n]']=='n':
           pass
        else:
            if (self.strategy['BEM_timing']=='at_gate_closure')|(
                    self.strategy['BEM_quantity']=='available_ramp')|(
                            self.strategy['BEM_pricing']=='srmc'):
                print("Agent {} makes BE bids".format(self.unique_id))
                #first delete all redispatch orders from previous round from orderbook
                self.model.BEM_obook.delete_orders(agent_id_orders = self.unique_id)
                asset_location_lst = []
                agentID_lst = []
                assetID_lst =[]
                init_lst = []
                direction_lst = []
                ordertype_lst = []
                qty_lst=[]
                price_lst=[]
                day_lst=[]
                mtu_lst=[]
                delivery_duration =[]
                gate_closure_MTU = self.model.BE_marketoperator.gate_closure_time
                if gate_closure_MTU <len(self.model.schedules_horizon):
                    dayindex = list(self.model.schedules_horizon.index.get_level_values(0))[gate_closure_MTU]
                    timeindex = list(self.model.schedules_horizon.index.get_level_values(1))[gate_closure_MTU]
                else:
                    #gate closure time is out of range of simulation. no bids are made
                    return
                otype = 'FRR_order'
                #init means simulation mtu plus agent step rank. works for up to 100 agents.
                init = self.step_rank/1000 +self.model.clock.get_MTU()
                #get all assets of that market party
                all_ids = self.assets.index.values
                for i in range(len(all_ids)):
                    a = self.assets.loc[all_ids[i],:].item()
                    if  ((a.schedule.loc[(dayindex,timeindex),
                                        'rem_ramp_constr_avail_up'] < 0).any())|((a.schedule.loc[(dayindex,timeindex),
                                        'rem_ramp_constr_avail_down']<0).any()):
                        import pdb
                        pdb.set_trace()
                        raise Exception ('remaining ramp of a generator is negative. This is inconsistent.')

                    #offers remaining ramp
                    qty_up_lst = [a.schedule.loc[(dayindex,timeindex),
                                        'rem_ramp_constr_avail_up'].astype(int)]
                    qty_down_lst = [a.schedule.loc[(dayindex,timeindex),
                                        'rem_ramp_constr_avail_down'].astype(int)]
                    price_up_lst = [int(a.srmc)]
                    price_down_lst = [int(a.srmc)]

                    asset_location_lst += [a.location]*2
                    agentID_lst += [self.unique_id]*2
                    assetID_lst += [a.assetID]*2
                    init_lst += [init]*2
                    direction_lst += ['buy'] + ['sell']
                    ordertype_lst += [otype]* 2
                    delivery_duration += [1] * 2
                    qty_lst += qty_down_lst + qty_up_lst
                    price_lst += price_down_lst + price_up_lst
                    day_lst += [dayindex] * 2
                    mtu_lst += [timeindex] * 2

                #make order dataframe from lists
                orders = DataFrame()
                columns=['agent_id','associated_asset','delivery_location',
                         'quantity','price', 'delivery_day','delivery_time',
                         'order_type','init_time', 'direction' , 'delivery_duration']
                values = [agentID_lst,assetID_lst, asset_location_lst,
                          qty_lst, price_lst, day_lst, mtu_lst,
                          ordertype_lst, init_lst, direction_lst, delivery_duration]

                #make dataframe per column to maintain datatype of lists (otherwise seen as objects by pandas)
                for i in range(len(columns)):
                    orders[columns[i]]=values[i].copy()
                #remove 0 MW rows
                orders = orders.loc[orders['quantity']!=0].copy()

                if not orders.empty:
                    # insert order IDs (must be at second last column because of many dependencies)
                    orders.insert(loc = len(orders.columns)-2,column='order_id',
                                  value =list(range(self.ordercount, self.ordercount + len(orders))) )
                    orders['order_id']=orders['agent_id'] + orders['associated_asset'] + orders['order_id'].astype(str)

                    #order count for the order ID
                    self.ordercount += len(orders)
                    order_message = OrderMessage(orders.copy())
                    self.model.BEM_obook.add_order_message(order_message)
            else:
                raise Exception ('Balancing energy market strategy not known')




    def small_random_quantity(self, av_cap, min_quantity = 5, max_quantity= 20):
        """Method: provides a random quantity, under consideration of the min and max available quantity.
           Timestamps with the same available capacity (min and max) receive the same the random quantity.
           Note:
           - min_quantity and max_quantity define the range of the random quantity
           - No quantity <1 are provided, even when available.
           - Input is a Series, output is a list
           """
        if av_cap.empty:
            return

        av_cap['min_vol'] = av_cap.iloc[:,0].where(av_cap.iloc[:,0]<min_quantity, min_quantity)
        av_cap['max_vol'] = av_cap.iloc[:,0].where(av_cap.iloc[:,0]<max_quantity, max_quantity)
        av_cap['rand_vol'] = int(0)
        #make small available quantity for the 'random quantity' to avoid unused capacity.
        av_cap['rand_vol'] = av_cap['rand_vol'].where(
                av_cap['min_vol']<av_cap['max_vol'],av_cap.iloc[:,0].astype(int).values)

        for available, group in av_cap.loc[(av_cap['rand_vol'] == 0)&(
                av_cap['max_vol'] - av_cap['min_vol'] >= 1)].groupby(
                by = ['min_vol','max_vol']):

            #make a random seed that is linked to the randomness of agent rank
            seed= self.step_rank + self.model.schedule.steps
            av_cap.loc[group.index,'rand_vol'] = np.random.RandomState(seed
                                ).randint(group['min_vol'].iloc[0], group['max_vol'].iloc[0])

        if av_cap['rand_vol'].isnull().any():
            import pdb
            pdb.set_trace()
        return (list(av_cap['rand_vol']))

    def start_stop_blocks (self, av_cap, pmin, min_up_time, min_down_time, assetID):
        """Method: identifies consecutive MTUs of asset schedule at pmin
            or at 0 MW. These MTU-blocks can be used for start-up or shut-down orders, if these periods
            are >= min_up_time (for upward orders) or min_down_time (for downward orders).

            Assumption of this method: if a (MTU-block) period is < 2 * min_up_time or min_down_time
            then the entire period is offered in one order. When >=
            2 * min_up_time or min_down_time, then several blocks with lengths of min_up_time or down_time are selected.
            Note:
            - The method returns two nested lists instead of DataFrames to be consistent with
              list approach of make_bid methods (is faster than append to DF).
            - One list for orders from start-up capacity one list for shut-down capacity.
            - Each list contains lists with delivery day, delivery time and delivery duration.
              [startdelday,startdeltime,startduration], [stopdelday, stopdeltime, stopduration]
            - The method implicitly assumes that the quantity of these start/ stop blocks is pmin
            - It has to be enshured that provided time series of av_cap starts with gate-closure time
              (i.e. timestamps as of which the blocks shall be calculated)
        """

        if (pmin == 0) | (av_cap.empty):
            #no dedicated stop pr start orders needed because the full range can be provided.
            #Also if available capacity is empty.
            #empty nested list return
            return([[],[],[]],[[],[],[]])

        def count_if_value(x,value):
            #x is pandas.rolling object
            amount = np.count_nonzero(np.where(x == value,1,0))
            return(amount)

        #check that no outages and stop commitments lead to new startblocks.
        av_cap['commit'] = av_cap['commit'].where(av_cap['p_max_t']!=0, -1)
        av_cap['commit'] = av_cap['commit'].where(~((av_cap['commit']==pmin)&(av_cap['upward_commit']!=0)), -1)
        av_cap.reset_index(inplace=True)

        #av_cap timeseries has to star with gate-closure time.
        gate_closure_time = (av_cap['delivery_day'].iloc[0], av_cap['delivery_time'].iloc[0])
        #search feasible start blocks (count lengths/duration of a block)
        av_cap['feasible_startblock'] =av_cap['commit'].rolling(window=min_up_time).apply(
                lambda x: count_if_value(x,0))
        #calculate the possible start time of each block
        av_cap['del_time_startblock'] = av_cap.apply(lambda x: self.model.clock.calc_delivery_period_end(
                (x['delivery_day'],x['delivery_time']), -min_up_time+2),axis=1)
        #remove blocks with start before gate-closure time
        av_cap['feasible_startblock'] =av_cap['feasible_startblock'].where(
                av_cap['del_time_startblock']>=gate_closure_time, np.nan)

        #search feasible stop blocks
        av_cap['feasible_stopblock'] =av_cap['commit'].rolling(window=min_down_time).apply(
                lambda x: count_if_value(x,pmin))
        av_cap['del_time_stopblock'] = av_cap.apply(lambda x: self.model.clock.calc_delivery_period_end(
                (x['delivery_day'],x['delivery_time']), -min_down_time+2),axis=1)
        av_cap['feasible_stopblock'] =av_cap['feasible_stopblock'].where(
                av_cap['del_time_startblock']>=gate_closure_time, np.nan)
        i =0
        startdelday=[]
        startdeltime =[]
        startduration =[]
        stopdelday =[]
        stopdeltime =[]
        stopduration =[]

        #block selection
        while i < len(av_cap):
            #selection start-up blocks
            if av_cap['feasible_startblock'].iloc[i] == min_up_time:
                #Subsequent block with duration min_down_time lies outside horizon.
                #Last feasible block of horizon has duration == min_up_time
                if  i + min_up_time-1 >= len(av_cap) -1:
                    #select block with duration min_up_time -1 + (len(av_cap) -i)
                    if av_cap['feasible_startblock'].iloc[len(av_cap)-1] == min_up_time:
                        startduration += [min_up_time-1 + (len(av_cap) -i)]
                        startdelday += [int(av_cap['del_time_startblock'].iloc[i][0])]
                        startdeltime += [int(av_cap['del_time_startblock'].iloc[i][1])]
                        #jump accordingly in while loop
                        i += len(av_cap)-i
                    #Last feasible block of horizon has duration < min_up_time
                    else:
                        ##search last feasible block before av_cap['feasible_startblock'].iloc[i +k]
                        k = 1
                        while av_cap['feasible_startblock'].iloc[i +k]==min_up_time:
                            k += 1
                        startduration += [min_up_time -1 + k]
                        startdelday += [int(av_cap['del_time_startblock'].iloc[i][0])]
                        startdeltime += [int(av_cap['del_time_startblock'].iloc[i][1])]
                        i += k
                #block lies within horizon and has duration == min_up_time
                #and there is a subsequent block == min_up_time
                elif av_cap['feasible_startblock'].iloc[i + min_up_time-1] == min_up_time:
                    #select block with duration min_up_time
                    startduration += [min_up_time]
                    startdelday += [int(av_cap['del_time_startblock'].iloc[i][0])]
                    startdeltime += [int(av_cap['del_time_startblock'].iloc[i][1])]
                    i += min_up_time
                #subsequent block has duration < min up time.
                else:
                    #search last feasible block before av_cap['feasible_startblock'].iloc[i + min_up_time-1]
                    k = 1
                    while av_cap['feasible_startblock'].iloc[i +k]==min_up_time:
                        k += 1
                    #select block with duration min_up_time -1 + k
                    startduration += [min_up_time -1 + k]
                    startdelday += [int(av_cap['del_time_startblock'].iloc[i][0])]
                    startdeltime += [int(av_cap['del_time_startblock'].iloc[i][1])]
                    i += k
            #selection shut-down blocks
            elif av_cap['feasible_stopblock'].iloc[i] == min_down_time:
                if  i + min_down_time-1 >= len(av_cap)-1:
                    if av_cap['feasible_stopblock'].iloc[len(av_cap)-1] == min_down_time:
                        stopduration += [min_down_time -1 + (len(av_cap) -i)]
                        stopdelday += [int(av_cap['del_time_stopblock'].iloc[i][0])]
                        stopdeltime += [int(av_cap['del_time_stopblock'].iloc[i][1])]
                        i += len(av_cap)-i
                    else:
                        k = 1
                        while av_cap['feasible_stopblock'].iloc[i +k] == min_down_time:
                            k += 1
                        stopduration += [min_down_time -1 + k]
                        stopdelday += [int(av_cap['del_time_stopblock'].iloc[i][0])]
                        stopdeltime += [int(av_cap['del_time_stopblock'].iloc[i][1])]
                        i += k
                elif av_cap['feasible_stopblock'].iloc[i + min_down_time-1] == min_down_time:
                    stopduration += [min_down_time]
                    stopdelday += [int(av_cap['del_time_stopblock'].iloc[i][0])]
                    stopdeltime += [int(av_cap['del_time_stopblock'].iloc[i][1])]
                    i += min_down_time
                else:
                    k = 1
                    while av_cap['feasible_stopblock'].iloc[i +k]==min_down_time:
                        k += 1
                    stopduration += [min_down_time -1 + k]
                    stopdelday += [int(av_cap['del_time_stopblock'].iloc[i][0])]
                    stopdeltime += [int(av_cap['del_time_stopblock'].iloc[i][1])]
                    i += k
            else:
                i+=1
        return ([startdelday,startdeltime,startduration], [stopdelday, stopdeltime, stopduration])


    def opportunity_markup(self, direction='upward', of_quantity = None, asset = None,
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
        elif direction == 'downward':
            IBP = 'IB_price_short'

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
            return(list(df['markup']), df)


    def startstop_markup (self, direction='upward', of_quantity = None,
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


    def ramping_markup(self, direction='upward', of_quantity = None, asset = None, unit_test=None):
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

    def doublescore_markup(self, direction='upward', of_quantity = None, asset = None, unit_test= None):
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
        - Assumption: capacity is first placed on the IDM, because of continous trading and instantanous clearing.
                      The capacity that is not (yet) cleared is subsequently also offered for redispatch.
        - Assumption: the risk quantity is determined with a uniform distribution assumption regarding
                      the quantity that is double-scored:
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
        av_cap['markup'] = av_cap['markup'].round().astype(int).copy()
        markup_lst = list(av_cap['markup'])

        if not unit_test:
            return(markup_lst)
        else:
            #return also av_cap in case of unit test.
            return (markup_lst, av_cap)

    def intraday_markup(self, indiff_price_lst, direction,
                        profit_margin_pu = 0.1, raise_margin_pu=0.03):
        """Method: provides a heuristic mark-up for intraday orders to a open order book.
           In case the indifference price (short-run marginal cost + opportunity cost) is extra marginal
           (meaning the order will not be cleared instantanously), the order price is set to the next
           more expensive order plus 1 Euro (in case of buy orders), respectively minus 1 euro (in case of sell order).
           If the indifference price is intra-marginal (will be cleared instantanously), then the indifference
           price plus a standard margin will be offered.
           In case of extra-marginal indifference prices where no competing orders orders exist
           the last cleared price + a raise marging is taken, when higher(sell)/lower(buy) than indifference price.
           The raise margin is a strategic margin to raise the price in a step-wise negotiation via small orders.

           Note:
           profit_margin is in p.u. ( 0.1 means 10% profit mark-up on indifference prices)
           raise_margin_pu (0.02 means 2% markup on the latest highest price)
        """

        #make Series with time index from price list (note: the price_list must be in shape of schedules_horizon)
        indiff_price = Series(indiff_price_lst, index = self.model.schedules_horizon.index)
        price_list = indiff_price.copy()
        price_list.loc[:] = np.nan
        buyorders = self.model.IDM_obook.buyorders[['delivery_day','delivery_time','price']].copy()
        sellorders = self.model.IDM_obook.sellorders[['delivery_day','delivery_time','price']].copy()

        #extra-marginal price setting
        if direction == 'buy':
                for timestamp ,orders_t in buyorders.groupby(by=['delivery_day','delivery_time']):
                   competing_price =orders_t['price'].loc[orders_t['price'] < indiff_price[timestamp]].max()
                   if (~math.isnan(competing_price))&(competing_price + 1 < indiff_price[timestamp]):
                       price_list[timestamp] = competing_price + 1

        elif direction == 'sell':
            for timestamp ,orders_t in sellorders.groupby(by=['delivery_day','delivery_time']):
               competing_price = orders_t['price'].loc[orders_t['price'] > indiff_price[timestamp]].min()
               if (~math.isnan(competing_price))&(competing_price - 1 > indiff_price[timestamp]):
                   price_list[timestamp] = competing_price - 1

        #intra-marginal price setting
        if direction == 'buy':
            for timestamp ,orders_t in sellorders.groupby(by=['delivery_day','delivery_time']):
                if (orders_t['price'] <= indiff_price[timestamp]* (1 - profit_margin_pu)).any():
                    price_list[timestamp] = indiff_price[timestamp]* (1 - profit_margin_pu)
        elif direction == 'sell':
            for timestamp ,orders_t in buyorders.groupby(by=['delivery_day','delivery_time']):
                if (orders_t['price'] >= indiff_price[timestamp] * (1 + profit_margin_pu)).any():
                    price_list[timestamp] = indiff_price[timestamp] * (1 + profit_margin_pu)

        #get previous clearing prices for extra-marginal price setting
        if direction == 'buy':
            prev_highest_cleared_cur_step = self.model.IDM_obook.cleared_buyorders.set_index(['delivery_day', 'delivery_time'])
            prev_highest_cleared_prev_step = self.model.IDM_obook.cleared_buyorders_all_df.set_index(['delivery_day', 'delivery_time'])
            highest ='min'
        elif direction == 'sell':
            prev_highest_cleared_cur_step = self.model.IDM_obook.cleared_sellorders.set_index(['delivery_day', 'delivery_time'])
            prev_highest_cleared_prev_step = self.model.IDM_obook.cleared_sellorders_all_df.set_index(['delivery_day', 'delivery_time'])
            highest ='max'
        prev_highest_cleared_prev_step=prev_highest_cleared_prev_step.loc[
                    prev_highest_cleared_prev_step[
                            'offer_daytime']==self.model.clock.calc_timestamp_by_steps(self.model.schedule.steps-1,0)].copy()

        mask= prev_highest_cleared_prev_step.index.isin(prev_highest_cleared_cur_step.index)
        prev_highest_cleared=pd.concat([prev_highest_cleared_cur_step,
                prev_highest_cleared_prev_step[~mask]])

        if not  prev_highest_cleared.empty:
            #definition of highest price depends on buy or sell direction
            prev_highest_cleared=prev_highest_cleared.reset_index().groupby(['delivery_day', 'delivery_time'])['cleared_price'].agg(highest).reset_index()
            #make index values integers again
            prev_highest_cleared[['delivery_day', 'delivery_time']]=prev_highest_cleared[['delivery_day', 'delivery_time']].astype('int64')
            prev_highest_cleared.set_index(['delivery_day', 'delivery_time'], inplace=True)

        missing_prices = price_list.loc[price_list.isnull().values]
        #use previous prices for missing prices (isnull)
        for i in range(len(missing_prices)):
            day = missing_prices.index.get_level_values(0)[i]
            mtu = missing_prices.index.get_level_values(1)[i]
            if not  prev_highest_cleared.empty:
                if (day,mtu) in prev_highest_cleared.index:
                    last_highest_price = prev_highest_cleared.loc[(day,mtu), 'cleared_price']
                else:
                    last_highest_price = None
            else:
                last_highest_price = None
            if direction == 'sell':
                if not last_highest_price == None:
                    if last_highest_price >= indiff_price[(day, mtu)] *(1 + profit_margin_pu):
                        price_list.loc[(day,mtu)] = last_highest_price *(1 + raise_margin_pu)
                    else:
                        price_list.loc[(day,mtu)] = indiff_price[(day, mtu)] *(1 + profit_margin_pu)
                else:
                    price_list.loc[(day,mtu)] = indiff_price[(day, mtu)] *(1 + profit_margin_pu)
            elif direction == 'buy':
                 if not last_highest_price == None:
                    if last_highest_price <= indiff_price[(day, mtu)] *(1 - profit_margin_pu):
                        price_list.loc[(day,mtu)] = last_highest_price*(1 - raise_margin_pu)
                    else:
                        price_list.loc[(day,mtu)] = indiff_price[(day, mtu)] *(1 - profit_margin_pu)
                 else:
                    price_list.loc[(day,mtu)] = indiff_price[(day, mtu)] *(1 - profit_margin_pu)
        price_list = price_list.round(0).astype(int).copy()
        return (list(price_list.values))




