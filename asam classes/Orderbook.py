# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 16:48:54 2017
@author: Samuel Glismann

The orderbook contains dataframes with orders.
and some variables used for statistics of report class and visualtion class.

Orderbooks belong to market operator instances.
Methods:
add_order_message()
delete_orders()
update_orders()
remove_matched_orders()
adjust_partial_match_orders()
get_obook_as_multiindex()
get_offered_position()
"""


import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from Time import *
from OrderMessage import *

class Orderbook():
    def __init__(self, model, ob_type=None):
        self.model = model
        #this makes the orderbook aware of the type of market operator it is used for
        self.ob_type = ob_type
        if self.ob_type != None: #Imbalance orderbook not required
            #label lists needed to convert single orders in dataframe collections
            self.offerlabels = ['agent_id','associated_asset','delivery_location',
                                 'quantity','price', 'delivery_day','delivery_time',
                                 'order_type','init_time', 'order_id', 'direction',
                                 'delivery_duration']
            self.transactionlabels = self.offerlabels +  ['matched_order',
                                                          'cleared_quantity','cleared_price','rem_vol','offer_daytime']
            #DataFrame with currently valid orders in orderbook
            self.buyorders = DataFrame(columns=self.offerlabels)
            self.sellorders = DataFrame(columns=self.offerlabels)

            #all orders submitted during entire simulationstep (deleted afterwards, by a reporters method)
            #The difference with the currently valid orders is that ID orders are cleared within one simulation step.
            #This distinction is irrelevant for e.g. redispatch orderbooks, as they are cleared once per round.
            self.buyorders_full_step = DataFrame(columns=self.offerlabels)
            self.sellorders_full_step = DataFrame(columns=self.offerlabels)
            if self.ob_type == 'redispatch':
                self.redispatch_demand_orders_upward = DataFrame(columns=self.offerlabels)
                self.redispatch_demand_orders_downward = DataFrame(columns=self.offerlabels)

            #cleared orders of one round
            self.cleared_sellorders = DataFrame(columns=self.transactionlabels)
            self.cleared_buyorders = DataFrame(columns=self.transactionlabels)

            #Dataframes that store all orders with indication of simulation daytime for later analysis.
            self.buyorders_all_df=DataFrame(columns=self.offerlabels)
            self.cleared_buyorders_all_df=DataFrame(columns=self.transactionlabels)
            self.sellorders_all_df=DataFrame(columns=self.offerlabels)
            self.cleared_sellorders_all_df=DataFrame(columns=self.transactionlabels)
            if self.ob_type =='redispatch':
                self.redispatch_demand_upward_all_df=DataFrame()
                self.redispatch_demand_downward_all_df=DataFrame()


            #the following orderbook reporters use a standard report matrix.
            #this matrix is with (redispatch) or without (others) location
            if self.ob_type == 'redispatch':
                report_index = model.clock.report_location_time_matrix
                if self.model.exodata.sim_task['save_intermediate_results[y/n]'] == 'n':
                    #reduce the large matrix
                    end_date = self.model.clock.calc_timestamp_by_steps(0,self.model.exodata.sim_task['number_steps']-1)
                    report_index = report_index.sort_index(level=[0,1,2]).loc[(slice(None),slice(end_date[0],end_date[0]),slice(end_date[1],end_date[1])),:].copy()
            else:
                report_index = model.clock.report_time_matrix
                if self.model.exodata.sim_task['save_intermediate_results[y/n]'] == 'n':
                    #reduce the large matrix
                    end_date = self.model.clock.calc_timestamp_by_steps(0,self.model.exodata.sim_task['number_steps']-1)
                    report_index = report_index.sort_index(level=[0,1]).loc[(slice(end_date[0],end_date[0]),slice(end_date[1],end_date[1])),:].copy()

            #reporters for the orderbook over time. used by visualizations.
            self.sell_sum_quantity = report_index.copy()
            self.sell_min_price = report_index.copy()
            self.sell_wm_price = report_index.copy()
            self.sell_max_price = report_index.copy()
            self.buy_sum_quantity = report_index.copy()
            self.buy_min_price = report_index.copy()
            self.buy_wm_price = report_index.copy()
            self.buy_max_price = report_index.copy()
            self.cleared_sell_sum_quantity = report_index.copy()
            self.cleared_sell_min_price = report_index.copy()
            self.cleared_sell_wm_price = report_index.copy()
            self.cleared_sell_max_price = report_index.copy()
            self.cleared_sell_number_trades = report_index.copy()
            self.cleared_buy_sum_quantity = report_index.copy()
            self.cleared_buy_min_price = report_index.copy()
            self.cleared_buy_wm_price = report_index.copy()
            self.cleared_buy_max_price = report_index.copy()
            self.cleared_buy_number_trades = report_index.copy()
            if self.ob_type =='redispatch':
                self.redispatch_demand_upward = report_index.copy() #only relevant for redispatch orderbook
                self.redispatch_demand_downward = report_index.copy() #only relevant for redispatch orderbook

            self.rep_dict ={
                    'sell_sum_quantity':self.sell_sum_quantity,
                    'sell_min_price':self.sell_min_price,
                    'sell_wm_price':self.sell_wm_price,
                    'sell_max_price' :self.sell_max_price,
                    'buy_sum_quantity':self.buy_sum_quantity,
                    'buy_min_price': self.buy_min_price,
                    'buy_wm_price': self.buy_wm_price,
                    'buy_max_price': self.buy_max_price,
                        'cleared_sell_sum_quantity':self.cleared_sell_sum_quantity,
                    'cleared_sell_min_price':self.cleared_sell_min_price,
                    'cleared_sell_wm_price':self.cleared_sell_wm_price,
                    'cleared_sell_max_price':self.cleared_sell_max_price,
                    'cleared_sell_number_trades':self.cleared_sell_number_trades,
                    'cleared_buy_sum_quantity':self.cleared_buy_sum_quantity,
                    'cleared_buy_min_price':self.cleared_buy_min_price,
                    'cleared_buy_wm_price':self.cleared_buy_wm_price,
                    'cleared_buy_max_price':self.cleared_buy_max_price,
                    'cleared_buy_number_trades':self.cleared_buy_number_trades,
                    }
            if self.ob_type =='redispatch':
                self.rep_dict['redispatch_demand_upward'] = self.redispatch_demand_upward
                self.rep_dict['redispatch_demand_downward'] = self.redispatch_demand_downward


    def add_order_message(self, order):
        if isinstance(order, OrderMessage):
            new_orders = order.get_as_df()
            if self.buyorders.empty:
                #this ensures that dtypes stay unchanged from OrderMessage dtypes.
                self.buyorders=new_orders.loc[new_orders['direction']=='buy']
            else:
                self.buyorders = self.buyorders.append(new_orders.loc[new_orders['direction']=='buy'],
                                                       ignore_index =True)
            if self.sellorders.empty:
                self.sellorders =new_orders.loc[new_orders['direction']=='sell']
            else:
                self.sellorders = self.sellorders.append(new_orders.loc[new_orders['direction']=='sell'],
                                                   ignore_index =True)

            if (new_orders['order_type']!='redispatch_demand').any():
                #dublicate block orders to have the delivery times correctly represented in reporters
                if new_orders.loc[new_orders['delivery_duration']>1].empty:
                    pass
                else:
                    blocks = new_orders.loc[new_orders['delivery_duration']>1]
                    for i in range(len(blocks)):
                        df = DataFrame(
                                [blocks.iloc[i]] *(blocks['delivery_duration'].iloc[i] - 1))
                        day_lst, mtu_lst = self.model.clock.calc_delivery_period_range(
                                blocks['delivery_day'].iloc[i],
                                blocks['delivery_time'].iloc[i],
                                blocks['delivery_duration'].iloc[i])
                        df['delivery_day'] = day_lst
                        df['delivery_time'] = mtu_lst
                        new_orders = new_orders.append(df, ignore_index = True)
                if self.buyorders_full_step.empty:
                    self.buyorders_full_step=new_orders.loc[new_orders['direction']=='buy']
                else:
                    self.buyorders_full_step = self.buyorders_full_step.append(new_orders.loc[new_orders['direction']=='buy'],
                                                       ignore_index =True)
                if self.sellorders_full_step.empty:
                     self.sellorders_full_step=new_orders.loc[new_orders['direction']=='sell']
                else:
                    self.sellorders_full_step = self.sellorders_full_step.append(new_orders.loc[new_orders['direction']=='sell'],
                                               ignore_index =True)
            else:
                self.redispatch_demand_orders_downward = new_orders.loc[new_orders['direction']=='buy'].copy()
                self.redispatch_demand_orders_upward =  new_orders.loc[new_orders['direction']=='sell'].copy()
        else:
            raise Exception('Orderbook only excepts orders of the OrderMessage() class')


    def delete_orders(self, order_ids=None, agent_id_orders=None):
        if agent_id_orders == None:
            pass
        else:
            if not self.buyorders.empty:
                self.buyorders = self.buyorders.loc[self.buyorders['agent_id'] != agent_id_orders].copy()
            if not self.sellorders.empty:
                self.sellorders = self.sellorders.loc[self.sellorders['agent_id'] != agent_id_orders].copy()

    def get_offered_position(self,associated_asset=None):
        """method provides the sum quantity offered per delivery period of associated with a given asset"""
        if associated_asset == None:
            pass
        else:
            col=['delivery_day','delivery_time','quantity']
            if not self.buyorders.empty:
                buy_position = self.buyorders.loc[self.buyorders['associated_asset'] == associated_asset,
                                                  col].copy().groupby(by=['delivery_day','delivery_time']).sum()
            else:
                buy_position = DataFrame()
            if not self.sellorders.empty:
                sell_position = self.sellorders.loc[self.sellorders['associated_asset'] == associated_asset,
                                                    col].copy().groupby(by=['delivery_day','delivery_time']).sum()
            else:
                sell_position = DataFrame()
        return(buy_position, sell_position)


    def update_orders(self, min_activ_time = None):
        #removes all orders that are not executable anymore
        act_day = self.model.clock.get_day()
        act_time = self.model.clock.get_MTU()

        if not self.buyorders.empty:
            self.buyorders = self.buyorders[(self.buyorders['delivery_day'] > act_day) |
                    ((self.buyorders['delivery_day'] == act_day) & (self.buyorders['delivery_time'] > act_time))]
        if not self.sellorders.empty:
            self.sellorders = self.sellorders[(self.sellorders['delivery_day'] > act_day) |
                ((self.sellorders['delivery_day'] == act_day) & (self.sellorders['delivery_time'] > act_time))]

    def remove_matched_orders(self, orderID_df):
        # make indexe objects from it
        ind = orderID_df.set_index('order_id').index
        ind1 = self.buyorders.set_index('order_id').index
        self.buyorders = self.buyorders[~ind1.isin(ind)]
        ind1 = self.sellorders.set_index('order_id').index
        self.sellorders = self.sellorders[~ind1.isin(ind)]

    def remove_market_orders(self):
        #market orders are fill-or-kill. They are not lasting in the orderbook when not instantanoulsy matched
        self.buyorders = self.buyorders.loc[self.buyorders['order_type']!='market_order'].copy()
        self.sellorders = self.sellorders.loc[self.sellorders['order_type']!='market_order'].copy()

    def adjust_partial_match_orders(self, orderID_df):
        #this index reset is needed for alignment with orderID_df and correct mask
        self.buyorders = self.buyorders.set_index('order_id')
        self.sellorders = self.sellorders.set_index('order_id')
        orderID_df.set_index('order_id', inplace =True)

        mask1 = self.buyorders.index.isin(orderID_df.index)
        mask2 = self.sellorders.index.isin(orderID_df.index)

        #adds new temporary rem_vol column
        self.buyorders= self.buyorders.join(orderID_df['rem_vol'])
        self.sellorders= self.sellorders.join(orderID_df['rem_vol'])

        #overwrites quantity with remaining quantity where ~mask is False
        self.buyorders['quantity']= self.buyorders['quantity'].where(~mask1,self.buyorders['rem_vol'])
        self.sellorders['quantity']= self.sellorders['quantity'].where(~mask2,self.sellorders['rem_vol'])

        self.buyorders.reset_index(inplace = True)
        self.sellorders.reset_index(inplace = True)
        #drop all columns like rem_vol etc. that come from manipulation
        self.buyorders = self.buyorders[self.offerlabels]
        self.sellorders = self.sellorders[self.offerlabels]

    def get_obook_as_multiindex(self, selection= None, incl_location = False):
        if selection == 'buyorders':
             obook_as_multiindex = self.buyorders.copy()
             sort_rule= [False,True]
        elif selection == 'sellorders':
            obook_as_multiindex = self.sellorders.copy()
            sort_rule= [True,True]
        else:
            raise Exception('selection argument is unknown')
        if incl_location == True:
            mi_columns = ['delivery_location','delivery_day','delivery_time']
        elif incl_location == False:
            mi_columns = ['delivery_day','delivery_time']
        obook_as_multiindex = obook_as_multiindex.set_index(mi_columns)
        obook_as_multiindex.sort_index(level = 1, inplace = True)
        obook_as_multiindex.sort_values(by =['price','init_time'], ascending= sort_rule, inplace=True)

        #useful columns for clearing and settlement
        obook_as_multiindex['rem_vol'] = obook_as_multiindex['quantity']
        obook_as_multiindex['cleared_quantity'] = np.NaN
        obook_as_multiindex['matched_order'] = np.NaN
        obook_as_multiindex['cleared_price']= np.NaN
        return (obook_as_multiindex)






