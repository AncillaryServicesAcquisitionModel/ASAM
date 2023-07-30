# -*- coding: utf-8 -*-
"""
Created on Fri May  5 11:40:02 2017
@author: Samuel Glismann

Grid and System Operator (e.g. TSO or DSO) class for ASAM.
The Grid and Sytstem Operator class could get inherent classes for specific operators, such as DSO' or TSO's.

Currently, the grid and System Operator class contains methods which are typically
executed by a Transmission System Operator (according to EU definitions):
    - check_market_consistency()
    - update_imbalances_and_returns()
The balancing market operation and imbalance settlement is implemented as a Market Operator method in ASAM.

Typical Grid Operator methods are:
    - determine_congestions()
    - redispatch_demand()
The redispatch market/mechansm is implemented in ASAM as a Market Operator method.

Yet, the ancillary service demand (redispatch) is not simulated but provided exogeneously.
"""

from mesa import Agent, Model
from mesa.time import RandomActivation
from random import randrange, choice
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from OrderMessage import *
from Orderbook import *



class GridSystemOperator():
    """The Grid and Sytstem Operator class could get inherent classes for specific
    operators, such as DSO' or TSO's"""

    def __init__(self, unique_id, model):
#        super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.money = 0
        self.model = model
        # used for unique order ID of Grid and System Operator
        self.ordercount = 0
        #redispatch demand does not contain past MTU
        #Notation: positive value means upward demand in area,negative means downward
        self.red_demand = DataFrame(columns = ['delivery_day', 'delivery_time'] + self.model.gridareas)
        self.red_demand.set_index(['delivery_day', 'delivery_time'], inplace = True)
        #procured is cleaned every round
        self.red_procured = DataFrame(columns=['agent_id','associated_asset','delivery_location','quantity','price', 'delivery_day','delivery_time','order_type','init_time', 'order_id', 'direction','matched_order','cleared_quantity','cleared_price','rem_vol', 'due_amount'])
        #return imbalance last round. cleared every round
        self.imb_return = 0
        self.financial_return = model.clock.asset_schedules_horizon() #manipulated every round. All returns where delivery periode>current time remain unchanged
        self.financial_return = self.financial_return.reindex(columns = ['RD_return', 'BE_return','IB_return', 'total_return'])
        self.imbalances = model.clock.asset_schedules_horizon() #manipulated every round. All trade positions where delivery periode>current time remain unchanged
        self.imbalances = self.imbalances.reindex(columns=  ['imbalance_redispatch', 'imbalance_market(realized)','imbalance_market(scheduled)','imbalance_balancing', 'sum_imbalances' ], fill_value= 0)
        self.imbalances.iloc[:]=np.nan
        self.system_transactions = model.clock.asset_schedules_horizon()
        self.system_transactions = self.system_transactions.reindex(columns=['DAM_sell','DAM_buy','IDM_sell', 'IDM_buy','RDM_sell', 'RDM_buy','BEM_sell','BEM_buy'])



    def check_market_consistency(self):
        #this check assumes that the System Operator directly receives all trades from market platforms
        # via so-called 'nomination on behalve'.
        all_trades = self.model.schedules_horizon.copy()
        all_trades= all_trades.add(self.system_transactions,fill_value = 0)
        buy_keys =[]
        sell_keys=[]
        if self.model.exodata.sim_task['save_intermediate_results[y/n]']=='y':
            #enlarge system transactions timeframe to schedule time index
            for obook in self.model.rpt.all_books.keys():
                csell = self.model.rpt.all_books[obook].cleared_sell_sum_quantity.sum(min_count=1)
                cbuy =self.model.rpt.all_books[obook].cleared_buy_sum_quantity.sum(min_count=1)
                all_trades[str(obook)+ '_sell' ] = csell
                all_trades[str(obook)+ '_buy' ] = cbuy
                if (obook =='DAM')|(str(obook) =='IDM'):
                    #store keys of commodity markets to check consistency
                    buy_keys =buy_keys +[str(obook)+ '_buy']
                    sell_keys =sell_keys +[str(obook)+ '_sell']

        elif self.model.exodata.sim_task['save_intermediate_results[y/n]']=='n':
            for obook in self.model.rpt.all_books.keys():
                sells= self.model.rpt.all_books[obook].cleared_sellorders_all_df.groupby(['delivery_day','delivery_time'])[
                        'cleared_quantity'].sum(min_count=1).reset_index()
                if sells.empty:
                    all_trades[str(obook)+ '_sell' ]= np.nan
                else:
                    sells[['delivery_day','delivery_time']]=sells[['delivery_day','delivery_time']]
                    sells.set_index(['delivery_day','delivery_time'], inplace=True)
                    all_trades[str(obook)+ '_sell' ]= sells.sort_index().round(0).astype('int64').copy()

                buys = self.model.rpt.all_books[obook].cleared_buyorders_all_df.groupby(['delivery_day','delivery_time'])[
                    'cleared_quantity'].sum(min_count=1).reset_index()
                if buys.empty:
                    all_trades[str(obook)+ '_buy' ] =np.nan
                else:
                    buys[['delivery_day','delivery_time']]=buys[['delivery_day','delivery_time']]
                    buys.set_index(['delivery_day','delivery_time'], inplace=True)
                    all_trades[str(obook)+ '_buy' ] = buys.sort_index().round(0).astype('int64').copy()

                if (obook =='DAM')|(str(obook) =='IDM'):
                    #store keys of commodity markets to check consistency
                    buy_keys =buy_keys +[str(obook)+ '_buy']
                    sell_keys =sell_keys +[str(obook)+ '_sell']


        #this is a consistency check: if the sum of all buy orders minus the sum of all sell orders != 0, there is an issue
        if ((all_trades[buy_keys].sum(axis=1).fillna(value=0)-all_trades[sell_keys].sum(axis=1).fillna(value=0)).round() != 0).any():
            all_trades['sum_trades'] = all_trades[buy_keys].sum(axis=1).fillna(value=0)-all_trades[sell_keys].sum(axis=1).fillna(value=0)
            self.system_transactions[all_trades.columns] = all_trades.copy()
            import pdb
            pdb.set_trace()
            raise Exception ('inconsistent trades detected: globally, cleared trades do not sum up to zero', all_trades)
        else:
            self.system_transactions = all_trades.copy()


    def update_imbalances_and_returns(self, positions =[]):
        new_imbalances = self.model.schedules_horizon.copy()
        new_imbalances = new_imbalances.add(self.imbalances,fill_value = 0)
        new_returns = self.model.schedules_horizon.copy()
        new_returns = new_returns.add(self.financial_return,fill_value = 0)
        for i in positions:
            new_transactions = DataFrame()
            if i == 'imbalance_redispatch':
                new_transactions = self.red_procured[['delivery_day','delivery_time'
                                                         ,'cleared_quantity','direction', 'due_amount']]
                #reverse pay direction to System Operator convention
                new_transactions['due_amount'] = - new_transactions['due_amount']
                k = 'RD_return'

                #attention: buy and sell orders are the market notation.
                #sell means market party sold, System Operator bought electricity. Buy means market party bought.
                #System Operator can neither produce nor consume electricity.
                #positive redispatch imbalance means System Operator has a long position (because System Operator bought more than it sold electricity)
                new_imbalances[i] =  self.system_transactions['RDM_sell'].fillna(value=0).sub(self.system_transactions['RDM_buy'], fill_value=np.nan)

            elif i == 'imbalance_market(realized)':
                #realized imbalance is only admistered for previous timestamp
                #no new transaction is admistered, as no orders involved
                k = 'IB_return'
                try:
                    timestamp = self.model.IB_marketoperator.cur_imbalance_settlement_period
                except:
                    timestamp = None
                if timestamp == None:
                    #no imbalance settlement has taken place, or no IBM is simulated
                    pass
                else:
                    new_imbalances.loc[timestamp,i] = self.model.IB_marketoperator.imbalances_short + self.model.IB_marketoperator.imbalances_long
                    new_returns.loc[timestamp, k] = self.imb_return
            elif i == 'imbalance_market(scheduled)':
                #scheduled imbalance from market is only admistered for monitoring
                #this is actually not yet firm or settled

                #set former scheduled market imbalance on 0, because it is not added to former scheduled imbalances
                new_imbalances.loc[self.model.schedules_horizon.index,i]=0

                for agent in self.model.schedule.agents:
                    #negative means short position, positive means long position.
                    if (agent.trade_schedule.loc[self.model.schedules_horizon.index ,'imbalance_position']!=0).any():
                        new_imbalances.loc[self.model.schedules_horizon.index,i
                                           ] = new_imbalances.loc[self.model.schedules_horizon.index,i
                                           ].add(agent.trade_schedule.loc[
                                                   self.model.schedules_horizon.index,'imbalance_position'], fill_value=0)
#                #remove scheduled imbalance value from  current (-1 because global counter has advanced after agents step)
                new_imbalances.loc[self.model.schedules_horizon.index[0],i]=np.nan
            else:
                raise Exception('Grid and system operator position to be updated is unknown')
            if new_transactions.empty:
                pass #do nothing. next position.
            else:

                new_transactions[k] = new_transactions['due_amount']
                new_transactions.set_index(['delivery_day','delivery_time'], inplace=True)
                #sum (saldo) of trades from the agent per timestamp
                new_transactions = new_transactions.groupby(level =[0,1]).sum(numeric_only=True)
                #add to 'return' column in self.financial_return
                new_returns[k] = new_returns[k].add(new_transactions[k], fill_value = 0)

        #overwrite imbalances
        self.imbalances = new_imbalances.copy()
        #overwrite self.financial returns
        self.financial_return = new_returns.copy()
        #calculate net imbalance.
        self.imbalances['sum_imbalances'] = self.imbalances[['imbalance_redispatch', 'imbalance_market(realized)','imbalance_market(scheduled)','imbalance_balancing']].sum(axis=1, min_count=1).fillna(value=np.nan)
        #calculate total return.
        self.financial_return['total_return'] = self.financial_return[['RD_return','IB_return', 'BE_return']].sum(axis=1, min_count=1).fillna(value=np.nan)



    def determine_congestions (self):
        """
        This method determines congestions. However, at the moment this means only
        reading exogeneosly provided redispatch demand.

        In future, a load-flow calculation could be added here.
        """

        if self.model.exodata.sim_task['congestions'] == 'exogenious':
            #check if exodatabase provides new congestions for this step
            #current time;
            cur_day = self.model.schedules_horizon.index.get_level_values(0)[0]
            cur_mtu = self.model.schedules_horizon.index.get_level_values(1)[0]
            new_congestion = self.model.exodata.congestions.loc[
                    (self.model.exodata.congestions['identification_day']==cur_day)&(
                            self.model.exodata.congestions['identification_MTU']==cur_mtu)]
            if new_congestion.empty:
                return(None)
            else:
                #make dataframe with areas as columns and horizon as index
                new_red_demand = pd.concat([self.model.schedules_horizon.copy(), DataFrame(columns=(self.model.gridareas))])
                new_red_demand[self.model.gridareas] = 0
                new_red_demand.drop('commit', axis =1, inplace = True)

                #make a addable data frame from identified new congestion.
                #can be multiple in a round
                for i in range(len(new_congestion)):
                    con_DF = new_red_demand.loc[(slice(new_congestion.loc[i,'congestion_start_day'],new_congestion.loc[i,'congestion_end_day']),
                                                        slice(new_congestion.loc[i,'congestion_start_time'],new_congestion.loc[i,'congestion_end_time'])),:].copy()

                    con_DF.loc[:,new_congestion.loc[i,'down_area']]= -new_congestion.loc[i,'redispatch_quantity']
                    con_DF.loc[:,new_congestion.loc[i,'up_area']]= new_congestion.loc[i,'redispatch_quantity']
                    new_red_demand = new_red_demand.add(con_DF, fill_value = 0)

                print('new congestions identified')
                return(new_red_demand)

        elif self.model.exodata.sim_task['congestions'] == 'from_scenario':
            #select all congestions within the schedules horizon with idetification time == current
            if self.model.exodata.congestions is not None:
                new_congestion = self.model.exodata.congestions.loc[
                        self.model.exodata.congestions.index.isin(self.model.schedules_horizon.index.values)]

                new_congestion = new_congestion.loc[(new_congestion['identification_day'] ==
                                self.model.schedules_horizon.index.values[0][0])&(
                        new_congestion['identification_MTU'] ==
                                self.model.schedules_horizon.index.values[0][1])].copy()
                if new_congestion.empty:
                    return(None)
            else:
                return(None)
            #make dataframe with areas as columns and horizon as index
            new_red_demand = pd.concat([self.model.schedules_horizon.copy(), DataFrame(columns=(self.model.gridareas))])
            new_red_demand[self.model.gridareas] = 0
            new_red_demand.drop('commit', axis =1, inplace = True)
            for congestion in Series(list(zip(list(new_congestion['redispatch_areas_down']),list(
                    new_congestion['redispatch_areas_up'])))).unique():
                #Series of a specific congestion
                C_MW = new_congestion.loc[(new_congestion['redispatch_areas_down']==congestion[0])&(
                        new_congestion['redispatch_areas_up']==congestion[1]), 'congestion_MW']
                #down area
                new_red_demand[congestion[0]] = new_red_demand[congestion[0]].add(-C_MW, fill_value = 0)
                #up area
                new_red_demand[congestion[1]] = new_red_demand[congestion[1]].add(C_MW, fill_value = 0)
            print('new congestions identified:')
            print(new_congestion)
            return(new_red_demand)


    def redispatch_demand(self, new_congestion):
        """
        Method combines new redispatch demand with previous redispatch demand (if not yet solved).
        It formulates demand orders.

        input: congestions (list with congestion parameters)
        """
        #check if redisaptch is part of simmulation task
        if self.model.exodata.sim_task['run_RDM[y/n]']=='n':
            print('Grid Operator: no redispatch in simlulation task')
        else:
            #first delete all Grid Operator redispatch orders from previous round from orderbook
            self.model.red_obook.delete_orders(agent_id_orders = self.unique_id)

            #get DF with schedule horizon for new redispatch demand DF
            new_red_demand = pd.concat([self.model.schedules_horizon.copy(), DataFrame(columns=(self.model.gridareas))])
            new_red_demand[self.model.gridareas] = 0
            new_red_demand.drop('commit', axis =1, inplace = True)

            #get redispatch demand from previous round
            new_red_demand = new_red_demand.add(self.red_demand, fill_value = 0)


            # include procured redispatch from previous round
            if self.red_procured.empty:
                #no redispatch procured in previous round
                pass
            else:
                self.red_procured.set_index(['delivery_location','delivery_day','delivery_time'], inplace =True)

                #make buyorders negative
                self.red_procured['cleared_quantity']=self.red_procured['cleared_quantity'].where(
                        self.red_procured['direction']=='sell',-self.red_procured['cleared_quantity'])
                #make again unique values per location, day and time
                shifted = self.red_procured['cleared_quantity'].groupby(level=[0,1,2]).sum()
                #pivot to get the areas as column names
                shifted = shifted.unstack(level = 0)
                #DF with booleans to check if more is procured than needed
                over_proc = new_red_demand.abs().fillna(value=0).sub(shifted.abs(), fill_value = 0) < 0
                remaining_demand = new_red_demand.fillna(value=0).sub(shifted, fill_value = 0)
                #substract procured redispatch from red_demand and fill 0 if proc>demand
                new_red_demand = remaining_demand.where(~over_proc, 0)
                self.red_procured = self.red_procured.iloc[0:0]
                self.red_procured.reset_index(inplace=True)
            #remove demand of delivery MTU that lie in the past
            new_red_demand = new_red_demand.loc[self.model.schedules_horizon.index]

            #add new congestions (if any)
            if new_congestion is None:
                print("no Grid Operator additional congestions identified in this round")
                pass
            else:
                new_red_demand = new_red_demand.add(new_congestion, fill_value = 0)
            self.red_demand = new_red_demand.copy()

            #delete all Grid Operator orders from previous rounds
            self.model.red_obook.delete_orders(agent_id_orders = -1)
            #make the redispatch_demand orders
            order_lst=[]
            make_orders = self.red_demand.loc[(self.red_demand != 0).any(axis=1)]
            make_orders.columns.name = 'delivery_location'
            make_orders = make_orders.stack()
            make_orders.name = 'quantity'
            make_orders = make_orders.reset_index()
            #exclude all areas with 0 demand
            make_orders = make_orders.loc[make_orders['quantity'] != 0].copy()
            make_orders = make_orders.reset_index()
            init = self.model.schedule.steps -1 + 0.0
            for k in range(len(make_orders)):
                order_id = str("GridOperator") + str("_") + str(self.ordercount)
                location = make_orders.loc[k,'delivery_location']
                #note that quantity will be positive in the Grid Operator order (23-08-2018)
                vol = int(make_orders.loc[k,'quantity'])
                if vol > 0:
                    direction = 'sell'
                elif vol < 0:
                    direction = 'buy'
                else:
                    continue
                price = np.nan
                delivery_duration =1
                delday = int(make_orders.loc[k,'delivery_day'])
                deltime = int(make_orders.loc[k,'delivery_time'])
                GridOperatorBid = [self.unique_id,"Network", location,abs(vol),price, delday, deltime,
                          'redispatch_demand', init, order_id, direction, delivery_duration]
                order_lst.append(GridOperatorBid)
                self.ordercount += 1
            if order_lst:
                orders = OrderMessage(order_lst)
                self.model.red_obook.add_order_message(orders)




