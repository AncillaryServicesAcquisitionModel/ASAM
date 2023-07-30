# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 09:59:29 2017
@author: Samuel Glismann
"""

from mesa import Agent, Model
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from ast import literal_eval



class Reports():
    def __init__(self, model):
        self.model = model
        #all_books dict is used to capture market statistics
        self.all_books ={}

        try:
            self.all_books['BEM']=self.model.BEM_obook
        except AttributeError:
            pass
        try:
            self.all_books['DAM']=self.model.DAM_obook
        except AttributeError:
            pass
        try:
            self.all_books['IDM']=self.model.IDM_obook
        except AttributeError:
            pass
        try:
            self.all_books['RDM']=self.model.red_obook
        except AttributeError:
            pass





        #here it is possible to add simple reports in MESA style
        self.table_reporters = {}
        self.model_reporters = {}
        self.agent_reporters = {"bank_deposit": lambda a: a.money,
                                "step_rank": lambda a: a.step_rank}


        self.prices= self.model.clock.schedule_template.copy()
        self.prices = self.prices.assign(DAP=None,IBP_short=None,IBP_long=None,
                                        control_state=None,IDM_weighted_mean=None,
                                        RDM_upwards_weighted_mean=None, RDM_downwards_weighted_mean=None,
                                        RDM_spread_weighted_mean=None)
        self.prices.drop('commit',axis=1, inplace=True)

        #Df with expected imbalance prices. This is updated every step and has size of schedules_horizon
        self.eIBP = DataFrame()


    def publish_DAM_prices(self,clearing_prices):
        #receives hourly clearing prices from DAM operator and converts them to prices per 15 minute MTU
        #Note: this method needs to be changed when a 15-minute DAM is simulated.
        clearing_prices = pd.concat([clearing_prices,clearing_prices,clearing_prices,clearing_prices])
        clearing_prices.sort_index(inplace=True)
        if (self.model.clock.get_MTU() >= self.model.DA_marketoperator.gate_closure_time) & (
                self.model.schedule.steps == 0):
            #get the right delivery mtu from the schedules horizon
            mtus = list(self.model.schedules_horizon.index.get_level_values(1))
        elif (self.model.clock.get_MTU() < self.model.DA_marketoperator.gate_closure_time) & (
                self.model.schedule.steps == 0):
            mtus = list(range(self.model.clock.get_MTU(),97))
        elif self.model.clock.get_MTU() == self.model.DA_marketoperator.gate_closure_time:
            mtus = list(range(1,97))
        #cut of MTUs of a first hour that lie in the past.
        clearing_prices = clearing_prices.iloc[len(clearing_prices)-len(mtus):]
        clearing_prices['delivery_time'] = mtus
        clearing_prices.drop('delivery_hour', axis = 1, inplace = True)
        clearing_prices.set_index(['delivery_day','delivery_time'], inplace= True)
        #change the bus name from pypsa model. to be generic, the column is copied and the original is dropped
        clearing_prices.rename(columns={clearing_prices.columns[0]:'DAP'},inplace =True)
        self.prices.loc[clearing_prices.index, 'DAP'] = clearing_prices['DAP']

    def publish_BEM_control_state(self,cur_control_state, day, mtu):
        self.prices.loc[(day,mtu),'control_state'] = cur_control_state

    def publish_IBM_prices(self,IBP_short,IBP_long, day, mtu):
        self.prices.loc[(day,mtu),'IBP_short'] = IBP_short
        self.prices.loc[(day,mtu),'IBP_long'] = IBP_long

    def publish_RDM_wm_prices(self):
        #make weighted averages over all simulation steps
        qty_red_up=self.model.red_obook.cleared_sell_sum_quantity
        wm_red_up=(self.model.red_obook.cleared_sell_wm_price * qty_red_up/(qty_red_up.sum(min_count=1))).sum(min_count=1).fillna(value=np.nan)

        qty_red_down=self.model.red_obook.cleared_buy_sum_quantity
        wm_red_down=(self.model.red_obook.cleared_buy_wm_price * qty_red_down/(qty_red_down.sum(min_count=1))).sum(min_count=1).fillna(value=np.nan)
        #Attention, it should be mentioned explicitly that in case one value is missing a 0 is assumed.
        wm_red_spread =wm_red_up.sub(wm_red_down, fill_value=0)

        self.prices['RDM_upwards_weighted_mean'] =wm_red_up
        self.prices['RDM_downwards_weighted_mean'] =wm_red_down
        self.prices['RDM_spread_weighted_mean'] =wm_red_spread

    def publish_IDM_wm_prices(self):
        #make weighted averages over all simulation steps
        qty_IDM = self.model.IDM_obook.cleared_sell_sum_quantity
        wm_IDM=(self.model.IDM_obook.cleared_sell_wm_price * qty_IDM/(qty_IDM.sum(min_count=1))).sum(min_count=1).fillna(value=np.nan)
        self.prices['IDM_weighted_mean'] = wm_IDM

    def get_cleared_prices(self):
        self.publish_RDM_wm_prices()
        self.publish_IDM_wm_prices()
        return(self.prices)

    def update_expected_IBP(self, MTU_of_h_consideration=False):
        """
        The expected imbalance price is used by agents to evaluate risks in their price mark-up calculations.
        The expected imbalance price method uses precalculated expected prices depending on day-ahead prices.
        When MTU_of_h_consideration==True, the method makes a distinction regarding the MTU of the hour.
           For this option the input data (opportunity_cost_db) must be delivered accordingly
           with a column 'PTU_of_an_hour' element {1,2,3,4}

        Note: the expeced IBP dataframe is updated every step to ensure similar length
               with self.model.schedules_horizon

        """
        if not self.prices['DAP'].isnull().all():
            self.eIBP = self.prices['DAP'].loc[self.model.schedules_horizon.index].to_frame()
        else:
            #if there is either no DAM run, or a default run without hourly prices
            self.eIBP = self.model.schedules_horizon
            self.eIBP['DAP'] = 30

        #MTU of hour list needed to get the specific expected prices
        MTU = list(self.eIBP.index.get_level_values(1))
        if MTU_of_h_consideration==True:
            MTU_of_h = []
            for mtu in MTU:
                mtu_of_h = mtu%4
                if mtu_of_h == 0:
                    mtu_of_h=4
                MTU_of_h += [mtu_of_h]
        DAP = list(self.eIBP['DAP'])
        odf = self.model.exodata.opportunity_costs_db
        expected_values_short =[]
        expected_values_long =[]
        for p in range(len(DAP)):
            #expeced value of IBP given a DA bin. The K-value is irrelevant here, but one existing (e.g. 30) must be chosen to get a unique return.
            try:
                if MTU_of_h_consideration==True:
                    value_short =odf.loc[(odf['price_data']=='IB_price_short') & (odf['PTU_of_an_hour']==MTU_of_h[p]) & (
                        odf['DAP_left_bin']<=DAP[p]) & (odf['DAP_right_bin(excl)'] >DAP[p] ) & (
                                odf['K-value'] == 30),'bin_exp_value'].iloc[0]
                else:
                    value_short =odf.loc[(odf['price_data']=='IB_price_short') & (
                        odf['DAP_left_bin']<=DAP[p]) & (odf['DAP_right_bin(excl)'] >DAP[p] ) & (
                                odf['K-value'] == 30),'bin_exp_value'].iloc[0]
            except:
                import pdb
                pdb.set_trace()

            expected_values_short += [int(round(value_short,0))]

            if MTU_of_h_consideration==True:
                value_long =odf.loc[(odf['price_data']=='IB_price_long') & (odf['PTU_of_an_hour']==MTU_of_h[p]) & (
                        odf['DAP_left_bin']<=DAP[p]) & (odf['DAP_right_bin(excl)'] >DAP[p] ) & (
                                odf['K-value'] == 30),'bin_exp_value'].iloc[0]
            else:
                value_long =odf.loc[(odf['price_data']=='IB_price_long') & (
                        odf['DAP_left_bin']<=DAP[p]) & (odf['DAP_right_bin(excl)'] >DAP[p] ) & (
                                odf['K-value'] == 30),'bin_exp_value'].iloc[0]
            expected_values_long += [int(round(value_long,0))]

        self.eIBP['expected_IBP_short'] =expected_values_short
        self.eIBP['expected_IBP_long'] = expected_values_long


    def save_all_orders_per_round(self):
        """
        - method must be applied before save_market_stats()
          because order_df  are deleted per round"""
        day, MTU=   self.model.clock.calc_timestamp_by_steps(self.model.schedule.steps -1, 0)
        key=(day,MTU)
        for obook in self.all_books.keys():
            #save orders and add offer time
            self.all_books[obook].buyorders_full_step['offer_daytime']= [key]*len(self.all_books[obook].buyorders_full_step)
            self.all_books[obook].buyorders_all_df = pd.concat([self.all_books[obook].buyorders_all_df,
                                                               self.all_books[obook].buyorders_full_step], ignore_index=True)
            self.all_books[obook].sellorders_full_step['offer_daytime']= [key]*len(self.all_books[obook].sellorders_full_step)
            self.all_books[obook].sellorders_all_df = pd.concat([self.all_books[obook].sellorders_all_df,
                                                                self.all_books[obook].sellorders_full_step], ignore_index=True)
            self.all_books[obook].cleared_buyorders['offer_daytime']= [key]*len(self.all_books[obook].cleared_buyorders)
            self.all_books[obook].cleared_buyorders_all_df = pd.concat([self.all_books[obook].cleared_buyorders_all_df,
                                                                      self.all_books[obook].cleared_buyorders], ignore_index=True)
            self.all_books[obook].cleared_sellorders['offer_daytime']= [key]*len(self.all_books[obook].cleared_sellorders)
            self.all_books[obook].cleared_sellorders_all_df = pd.concat([self.all_books[obook].cleared_sellorders_all_df,
                                                                        self.all_books[obook].cleared_sellorders], ignore_index=True)
            if obook == 'RDM':
                #considering also redispatch demand orders
                self.all_books[obook].redispatch_demand_orders_upward['offer_daytime']= [key]*len(self.all_books[obook].redispatch_demand_orders_upward)
                self.all_books[obook].redispatch_demand_upward_all_df=pd.concat([self.all_books[obook].redispatch_demand_upward_all_df,
                              self.all_books[obook].redispatch_demand_orders_upward], ignore_index=True)
                self.all_books[obook].redispatch_demand_orders_downward['offer_daytime']= [key]*len(self.all_books[obook].redispatch_demand_orders_downward)
                self.all_books[obook].redispatch_demand_downward_all_df=pd.concat([self.all_books[obook].redispatch_demand_downward_all_df,
                              self.all_books[obook].redispatch_demand_orders_downward], ignore_index=True)

            #delete all df's with orders of that round for orderbook with actually valid orders
            self.all_books[obook].sellorders_full_step =self.all_books[obook].sellorders_full_step.iloc[0:0]
            self.all_books[obook].buyorders_full_step =self.all_books[obook].buyorders_full_step.iloc[0:0]
            self.all_books[obook].cleared_sellorders =self.all_books[obook].cleared_sellorders.iloc[0:0]
            self.all_books[obook].cleared_buyorders =self.all_books[obook].cleared_buyorders.iloc[0:0]
            if obook == 'RDM':
                self.all_books[obook].redispatch_demand_orders_upward=self.all_books[obook].redispatch_demand_orders_upward.iloc[0:0]
                self.all_books[obook].redispatch_demand_orders_downward=self.all_books[obook].redispatch_demand_orders_downward.iloc[0:0]

    def save_market_stats(self, mode='at_end'):
        """ This method stores market statistics regarding offered and cleared quantity and prices.
            It can also store these statistic per simulation step to analyse the intermediate simulation stages.
            The intermediate statistics may also be used by agents.

            In a matrix with delivery time in columns and simulation time (and area incase of redispatch) in index,
            the following indicators are saved per market:
                 - sum of offered and cleared quantity per direction (buy sell)
                 - minimum, maximum and quantity weigtened average prices for offered and cleared orders per direction

        Note:
           - if modues 'at_end' is chosen, the statistics for all markets are only once per simulation calculated.
             This saves time compared to calculation per step with modus'every_round'.
           - sum quantity indicators are sum of (cleared) order MW. Not MWh
           - Truly, this method maybe written more beautifully when the statistics per variable are stored in a multi-index dataframe or nested dictionary.

           """

        #calculate timestamp of previous round
        day, MTU=   self.model.clock.calc_timestamp_by_steps(self.model.schedule.steps -1, 0)

        self.save_all_orders_per_round()
        if (mode =='every_step')|((day==self.model.clock.end_date[0])&(MTU == self.model.clock.end_date[1])):
            print('save market statistics of previous round')

            if mode =='every_step':
                offer_daytimes = [(day,MTU)]
            elif mode =='at_end':
                 offer_daytimes = list(self.model.clock.report_time_matrix.index.values)
            for obook in self.all_books.keys():
                #select index type
                if obook == 'RDM': #has also gridarea in index
                    round_index = (slice(None),slice(day,day),slice(MTU,MTU))
                else:
                    round_index = (day,MTU)

                #Offered sell orders
                if self.all_books[obook].sellorders_all_df.loc[
                            self.all_books[obook].sellorders_all_df['offer_daytime'].isin(offer_daytimes)].empty:
                    pass
                else:
                    if obook =='IDM':
                        #market order prices need to be set to np.nan to avoid distortion of statistic
                        self.all_books[obook].sellorders_all_df.loc[self.all_books[obook].sellorders_all_df
                                       ['order_type'] == 'market_order', 'price'] = np.nan
                    #get statistics of the orders
                    res = self.group_statistic(self.all_books[obook].sellorders_all_df.loc[
                            self.all_books[obook].sellorders_all_df['offer_daytime'].isin(offer_daytimes)], 'quantity', obook)

                    #add these statistics to respective statistic variables of the orderbook
                    if obook=='RDM':
                        included_areas=list(res.index.unique().get_level_values(level=0))
                        res=res.unstack(level=0)
                        for i in range(len(included_areas)):
                            area =included_areas[i]
                            #store results in report matrixes
                            self.all_books[obook].sell_sum_quantity.loc[(area,day,MTU),:] = res.loc[:,('quantity','sum',area)].copy()
                            self.all_books[obook].sell_min_price.loc[(area,day,MTU),:] = res.loc[:,('price','min',area)].copy()
                            self.all_books[obook].sell_wm_price.loc[(area,day,MTU),:] = res.loc[:,('price','weighted_mean',area)].copy()
                            self.all_books[obook].sell_max_price.loc[(area,day,MTU),:] = res.loc[:,('price','max',area)].copy()
                    else:
                        #store results in report matrixes
                        self.all_books[obook].sell_sum_quantity.loc[round_index,res.index] = res[('quantity','sum')].copy()
                        self.all_books[obook].sell_min_price.loc[round_index,res.index] = res[('price','min')].copy()
                        self.all_books[obook].sell_wm_price.loc[round_index,res.index] = res[('price','weighted_mean')].copy()
                        self.all_books[obook].sell_max_price.loc[round_index,res.index] = res[('price','max')].copy()

                #Offered buy orders
                if self.all_books[obook].buyorders_all_df.loc[
                            self.all_books[obook].buyorders_all_df['offer_daytime'].isin(offer_daytimes)].empty:
                    pass
                else:
                    if obook =='IDM':
                        #market order prices of need to be set to np.nan to avoid distortion of statistic
                        self.all_books[obook].buyorders_all_df.loc[self.all_books[obook].buyorders_all_df
                                       ['order_type'] == 'market_order', 'price'] = np.nan
                    #get statistics of the orders
                    res = self.group_statistic(self.all_books[obook].buyorders_all_df.loc[
                            self.all_books[obook].buyorders_all_df['offer_daytime'].isin(offer_daytimes)], 'quantity', obook)
                    #add these statistics to respective statistic variables of the orderbook
                    if obook=='RDM':
                        included_areas=list(res.index.unique().get_level_values(level=0))
                        res=res.unstack(level=0)
                        for i in range(len(included_areas)):
                            area =included_areas[i]
                            #store results in report matrixes
                            self.all_books[obook].buy_sum_quantity.loc[(area,day,MTU),:] = res.loc[:,('quantity','sum',area)].copy()
                            self.all_books[obook].buy_min_price.loc[(area,day,MTU),:] = res.loc[:,('price','min',area)].copy()
                            self.all_books[obook].buy_wm_price.loc[(area,day,MTU),:] = res.loc[:,('price','weighted_mean',area)].copy()
                            self.all_books[obook].buy_max_price.loc[(area,day,MTU),:] = res.loc[:,('price','max',area)].copy()
                    else:
                        #store results in report matrixes
                        self.all_books[obook].buy_sum_quantity.loc[round_index, res.index] = res[('quantity','sum')].copy()
                        self.all_books[obook].buy_min_price.loc[round_index, res.index] = res[('price','min')].copy()
                        self.all_books[obook].buy_wm_price.loc[round_index, res.index] = res[('price','weighted_mean')].copy()
                        self.all_books[obook].buy_max_price.loc[round_index, res.index] = res[('price','max')].copy()

                #Cleared sell orders
                if self.all_books[obook].cleared_sellorders_all_df.loc[
                            self.all_books[obook].cleared_sellorders_all_df['offer_daytime'].isin(offer_daytimes)].empty:
                    pass
                else:
                    res = self.group_statistic(self.all_books[obook].cleared_sellorders_all_df.loc[
                            self.all_books[obook].cleared_sellorders_all_df['offer_daytime'].isin(offer_daytimes)], 'cleared_quantity', obook)
                    if obook=='RDM':
                        included_areas=list(res.index.unique().get_level_values(level=0))
                        res=res.unstack(level=0)
                        for i in range(len(included_areas)):
                            area =included_areas[i]
                            #store results in report matrixes
                            self.all_books[obook].cleared_sell_sum_quantity.loc[(area,day,MTU),:] = res.loc[:,('cleared_quantity','sum',area)].copy()
                            self.all_books[obook].cleared_sell_min_price.loc[(area,day,MTU),:] = res.loc[:,('cleared_price','min',area)].copy()
                            self.all_books[obook].cleared_sell_wm_price.loc[(area,day,MTU),:] = res.loc[:,('cleared_price','weighted_mean',area)].copy()
                            self.all_books[obook].cleared_sell_max_price.loc[(area,day,MTU),:] = res.loc[:,('cleared_price','max',area)].copy()
                    else:
                        #store results in report matrixes
                        self.all_books[obook].cleared_sell_sum_quantity.loc[round_index, res.index] = res[('cleared_quantity','sum')].copy()
                        self.all_books[obook].cleared_sell_min_price.loc[round_index, res.index] = res[('cleared_price','min')].copy()
                        self.all_books[obook].cleared_sell_wm_price.loc[round_index, res.index] = res[('cleared_price','weighted_mean')].copy()
                        self.all_books[obook].cleared_sell_max_price.loc[round_index, res.index] = res[('cleared_price','max')].copy()

                #Clared buy orders
                if self.all_books[obook].cleared_buyorders_all_df.loc[
                            self.all_books[obook].cleared_buyorders_all_df['offer_daytime'].isin(offer_daytimes)].empty:
                    pass
                else:
                    res = self.group_statistic(self.all_books[obook].cleared_buyorders_all_df.loc[
                            self.all_books[obook].cleared_buyorders_all_df['offer_daytime'].isin(offer_daytimes)], 'cleared_quantity', obook)
                    if obook=='RDM':
                        included_areas=list(res.index.unique().get_level_values(level=0))
                        res=res.unstack(level=0)
                        for i in range(len(included_areas)):
                            area =included_areas[i]
                            #store results in report matrixes
                            self.all_books[obook].cleared_buy_sum_quantity.loc[(area,day,MTU),:] = res.loc[:,('cleared_quantity','sum',area)].copy()
                            self.all_books[obook].cleared_buy_min_price.loc[(area,day,MTU),:] = res.loc[:,('cleared_price','min',area)].copy()
                            self.all_books[obook].cleared_buy_wm_price.loc[(area,day,MTU),:] = res.loc[:,('cleared_price','weighted_mean',area)].copy()
                            self.all_books[obook].cleared_buy_max_price.loc[(area,day,MTU),:] = res.loc[:,('cleared_price','max',area)].copy()
                    else:
                        #store results in report matrixes
                        self.all_books[obook].cleared_buy_sum_quantity.loc[round_index, res.index] = res[('cleared_quantity','sum')].copy()
                        self.all_books[obook].cleared_buy_min_price.loc[round_index, res.index] = res[('cleared_price','min')].copy()
                        self.all_books[obook].cleared_buy_wm_price.loc[round_index, res.index] = res[('cleared_price','weighted_mean')].copy()
                        self.all_books[obook].cleared_buy_max_price.loc[round_index, res.index] = res[('cleared_price','max')].copy()
                #add also redispatch demand statistics
                if obook=='RDM':
                    ###round_index = (slice(None),slice(day,day),slice(MTU,MTU))# is in the beginning defined
                    #get redispatch demand order quantity upward
                    if self.all_books[obook].redispatch_demand_upward_all_df.loc[
                            self.all_books[obook].redispatch_demand_upward_all_df['offer_daytime'].isin(offer_daytimes)].empty:
                        pass
                    else:
                        red =  self.group_statistic(self.all_books[obook].redispatch_demand_upward_all_df.loc[
                            self.all_books[obook].redispatch_demand_upward_all_df['offer_daytime'].isin(offer_daytimes)].fillna(value=0), 'quantity', obook)
                        included_areas=list(red.index.unique().get_level_values(level=0))
                        red=red.unstack(level=0)
                        for i in range(len(included_areas)):
                            area =included_areas[i]
                            self.all_books[obook].redispatch_demand_upward.loc[(area,day,MTU),:] = red.loc[:,('quantity','sum',area)].copy()
                    #get redispatch demand order quantity downward
                    if self.all_books[obook].redispatch_demand_downward_all_df.loc[
                            self.all_books[obook].redispatch_demand_downward_all_df['offer_daytime'].isin(offer_daytimes)].empty:
                        pass
                    else:
                        red =  self.group_statistic(self.all_books[obook].redispatch_demand_downward_all_df.loc[
                            self.all_books[obook].redispatch_demand_downward_all_df['offer_daytime'].isin(offer_daytimes)].fillna(value=0), 'quantity', obook)
                        included_areas=list(red.index.unique().get_level_values(level=0))
                        red=red.unstack(level=0)
                        for i in range(len(included_areas)):
                            area =included_areas[i]
                            self.all_books[obook].redispatch_demand_downward.loc[(area,day,MTU),:] = red.loc[:,('quantity','sum',area)].copy()


    def group_statistic(self, data, quantity_type, obook):
        """quantity_type must be string 'quantity' or string 'cleared_quantity'"""
        if quantity_type == 'cleared_quantity':
            price_type = 'cleared_price'
        else:
            price_type = 'price'
        if  obook=='RDM':
            group_lst= ['delivery_location','delivery_day','delivery_time']
        else:
            group_lst= ['delivery_day','delivery_time']
        changed_price=False
        if pd.isnull(data[price_type]).all():
            #to avoid errors with agg() function
            data[price_type]=1
            changed_price=True

        data=data.reset_index()
        grouped = data.groupby(group_lst)
        try:

            wm = lambda x: np.average(x, weights=data.loc[x.index,quantity_type])
        except:
            import pdb
            pdb.set_trace()

        f = {quantity_type: 'sum', price_type:[wm, 'mean', 'max','min']}
        result =grouped.agg(f).rename(columns={'<lambda_0>':'weighted_mean'})

        if changed_price == True:
            #make price results np.nan again, but hold dataframe format
            result[price_type]=np.nan

        result= result.reset_index()
        result[['delivery_day','delivery_time']]=result[['delivery_day','delivery_time']].astype('int64')
        result.set_index(group_lst, inplace=True)
        result.sort_index(inplace = True)
        return(result)


    def get_system_trade(self):
        """sum of cleared orders (buy and sell seperated and as total saldo)
        for all deliverytimes. Result shows all trades up to the step of method application"""
        all_trades=DataFrame()
        buy_keys =[]
        sell_keys=[]
        for obook in self.all_books.keys():
            csell = self.all_books[obook].cleared_sell_sum_quantity.sum()
            cbuy =self.all_books[obook].cleared_buy_sum_quantity.sum()
            all_trades[str(obook)+ '_sell' ] = csell
            all_trades[str(obook)+ '_buy' ] = cbuy
            if (str(obook) =='DAM')|(str(obook) =='IDM'):
                #store keys of commodity markets to check consistency
                buy_keys =buy_keys +[str(obook)+ '_buy']
                sell_keys =sell_keys +[str(obook)+ '_sell']

        all_trades['sum_trades'] = all_trades[buy_keys].sum(axis=1)-all_trades[sell_keys].sum(axis=1)
        return (all_trades)

    def get_all_trade_schedules(self):
        #latest trade schedules per agent in a dataframe
        all_trade_schedules = DataFrame()
        for agent in self.model.schedule.agents:
            df=agent.trade_schedule.copy()
            df.drop(['commit','total_dispatch'], axis=1, inplace=True)
            df.columns=pd.MultiIndex.from_product([[agent.unique_id],df.columns])
            all_trade_schedules = pd.concat([all_trade_schedules,df], axis=1)
        return (all_trade_schedules)

    def get_all_returns(self):
        #latest returbs per agent in a dataframe
        all_returns = DataFrame()
        for agent in self.model.schedule.agents:
            df=agent.financial_return.copy()
            df.drop(['commit'], axis=1, inplace=True)
            df.columns=pd.MultiIndex.from_product([[agent.unique_id],df.columns])
            all_returns = pd.concat([all_returns,df], axis=1)
        return (all_returns)


    def get_system_dispatch(self):
        #latest dispatch schedules per agent in a dataframe
        all_scheds = {}
        for agent in self.model.schedule.agents:
            for asset in agent.assets['object']:
                name =  asset.assetID
                all_scheds[name]=asset.schedule['commit']
        dispatch_df = DataFrame.from_dict(all_scheds)
        dispatch_df['sum_dispatch'] = dispatch_df.sum(axis=1)
        return (dispatch_df)

    def get_system_cost(self):
        """ critics: intraday return is counted twice, while redispatch and DA are not"""
        all_return = {}
        all_dispatch_cost = {}
        for agent in self.model.schedule.agents:
            all_return[agent.unique_id] = agent.financial_return['total_return']
            all_dispatch_cost[agent.unique_id] = agent.financial_return['total_dispatch_costs']
        return_df =  DataFrame.from_dict(all_return)
        return_df['sum_return'] = return_df.sum(axis = 1)
        dcost_df =  DataFrame.from_dict(all_dispatch_cost)
        dcost_df['sum_dispatch_cost'] = dcost_df.sum(axis = 1)
        df = pd.concat([return_df['sum_return'],dcost_df['sum_dispatch_cost']], axis =1)
        df['producer_surplus'] = df['sum_return'] + df['sum_dispatch_cost']
        return (df)


    def redispatch_PI(self, norm_values_to =None):
        """
        Method provides the performance indicators 'over-procurement',
        'under-procurement' and 'redispatch imbalance' per delivery timestamp
        Note that the values are in MW, not MWh"""

        #get a dataframe with cleared redispatch and imbalances for the past up to horizon end.
        r_pi = pd.concat([self.model.aGridAndSystemOperator.imbalances['imbalance_redispatch'],
                          self.model.aGridAndSystemOperator.system_transactions[['RDM_buy', 'RDM_sell']]],axis=1)
        r_pi['redispatch_demand'] = np.nan

        #Get all identified congestions up to this simulation step
        cong = self.model.exodata.congestions.reset_index().set_index(['identification_day','identification_MTU']).sort_index().loc[
                    :self.model.schedules_horizon.index[0]]
        #ensure that congestions before starttime are not considered
        cong=cong.loc[(self.model.exodata.sim_task[ 'start_day'],self.model.exodata.sim_task[ 'start_MTU']):].reset_index()


        if self.model.exodata.sim_task['congestions'] == 'exogenious':
            if cong.empty:
                pass
            else:
                for i in range(len(cong)):
                    #there can be overlapping congestions
                    a_cong= r_pi.loc[(slice(cong.loc[i,'congestion_start_day'],cong.loc[
                            i,'congestion_end_day']), slice(cong.loc[i,'congestion_start_time'
                                                  ], cong.loc[i,'congestion_end_time'
                                                  ])), 'redispatch_demand']
                    a_cong = a_cong.add(cong.loc[i,'redispatch_quantity'], fill_value=0)
                    try:
                        r_pi.loc[a_cong.index,'redispatch_demand'] = r_pi.loc[a_cong.index,'redispatch_demand'].add(a_cong, fill_value =0)
                    except:
                        #a_cong not in the index horizon of r_pi
                        pass
        elif self.model.exodata.sim_task['congestions'] == 'from_scenario':
            if cong.empty:
                pass
            else:
                cong.set_index(['delivery_day','delivery_time'], inplace=True)
                r_pi.loc[cong.index,'redispatch_demand'] = cong['congestion_MW']
        else:
            raise Exception('redispatch_PI doesnt now sim_task congestion paramameter')

        r_pi['residual_demand_downwards'] = r_pi['RDM_buy']- r_pi['redispatch_demand']
        r_pi['residual_demand_upwards'] = r_pi['RDM_sell']- r_pi['redispatch_demand']
        r_pi['overproc_downwards'] = r_pi['residual_demand_downwards'].where(r_pi['residual_demand_downwards']>0,np.nan)
        r_pi['underproc_downwards'] = -r_pi['residual_demand_downwards'].where(r_pi['residual_demand_downwards']<0,np.nan)
        r_pi['overproc_upwards'] = r_pi['residual_demand_upwards'].where(r_pi['residual_demand_upwards']>0,np.nan)
        r_pi['underproc_upwards'] = -r_pi['residual_demand_upwards'].where(r_pi['residual_demand_upwards']<0,np.nan)
        r_pi['redispatch_solved'] =pd.concat([r_pi['redispatch_demand'].fillna(value=0).where(r_pi['RDM_buy']>=r_pi['redispatch_demand'].fillna(value=0), r_pi['RDM_buy']
            ), r_pi['redispatch_demand'].fillna(value=0).where(r_pi['RDM_sell']>=r_pi['redispatch_demand'].fillna(value=0), r_pi['RDM_sell'])], axis=1).sum(axis=1, min_count=1)/2
        """to be added when needed"""
        #sum overprocurement and underprocument relative to redisaptch demand, load, peak,load
        return(r_pi)

    def redispatch_supply_demand_ratio(self):
        #Please consult ASAM documentation for rational behind demand supply ratio
        if self.model.red_obook.redispatch_demand_upward_all_df.empty:
            print('redispatch_supply_demand_ratio determined as demand df is empty')
            return (None)
        else:
            #quantity mean supply/demand ratio per offer time
            demand_up =self.model.red_obook.redispatch_demand_upward_all_df.groupby(by=[
                'delivery_day','delivery_time','delivery_location','offer_daytime']).sum(numeric_only=False)['quantity']
            demand_down =self.model.red_obook.redispatch_demand_downward_all_df.groupby(
                by=['delivery_day','delivery_time','delivery_location','offer_daytime']).sum(numeric_only=False)['quantity']

            supply_up =self.model.red_obook.sellorders_all_df.groupby(by=[
                'delivery_day','delivery_time','delivery_location','offer_daytime']).sum(numeric_only=False)['quantity']
            supply_down =self.model.red_obook.buyorders_all_df.groupby(by=[
                'delivery_day','delivery_time','delivery_location','offer_daytime']).sum(numeric_only=False)['quantity']

            #remove redispatch demand for past delivery times
            demand_up= demand_up.where(demand_up.reset_index().set_index(
                    ['delivery_day','delivery_time']
                    ).index.values >=demand_up.reset_index()['offer_daytime'].values,np.nan)
            demand_up =demand_up.to_frame().join(supply_up.to_frame(), lsuffix='_demand',rsuffix='_supply').copy()

            demand_down= demand_down.where(demand_down.reset_index().set_index(
                    ['delivery_day','delivery_time']
                    ).index.values >=demand_down.reset_index()['offer_daytime'].values,np.nan)
            demand_down =demand_down.to_frame().join(supply_down.to_frame(), lsuffix='_demand',rsuffix='_supply').copy()

            #calculate ratio
            demand_up['s_d_ratio'] =demand_up['quantity_supply'].fillna(value=0).values/demand_up['quantity_demand'].values
            demand_down['s_d_ratio'] =demand_down['quantity_supply'].fillna(value=0).values/demand_down['quantity_demand'].values
            r_pi=Series(dtype='float64')
            #make mean values
            r_pi['av_s_d_ratio_up'] = demand_up['s_d_ratio'].mean()
            r_pi['av_s_d_ratio_down'] = demand_down['s_d_ratio'].mean()
            return(r_pi)


    def interdependence_indicators(self, quantity_indicator='sum'):
        """ Method
        calculates statistics across offer time and delivery time on various markets.

        Note:
            quantity results can be confusing, as e.g. average cleared quantity might
            be larger than average offered quantity.

            Total quantity (sum), on the other hand, does not entail the 'averaging effects',
            but can also be confusing because of large differences in trading periods
            and resulting large differences in offered quantity. For both indicators
            it is important to notice that cleared quantity influence the offered quantity.
        """

        indicators= DataFrame(index =list(self.all_books.keys()), columns=[
                'qty_av_av_sell','qty_av_av_buy','qty_av_av_sell_cleared','qty_av_av_buy_cleared',
                'price_med_wav_sell','price_med_wav_buy','price_med_wav_sell_cleared','price_med_wav_buy_cleared',
                'return', 'return [%]'])

        for obook in self.all_books.keys():
            #calculate the average quantity offered per round, normed over all delivery periods
            if not self.all_books[obook].buyorders_all_df.empty:
                qty_buy= self.all_books[obook].buyorders_all_df.groupby(by=['delivery_day', 'delivery_time','offer_daytime'])['quantity'].sum()
                indicators.loc[obook, 'qty_av_av_buy'] = (qty_buy.unstack(level=[0,1])).mean().mean()
                indicators.loc[obook, 'qty_total_buy_MWh'] = qty_buy.sum(min_count=1)/4
            if not self.all_books[obook].sellorders_all_df.empty:
                qty_sell= self.all_books[obook].sellorders_all_df.groupby(by=['delivery_day', 'delivery_time','offer_daytime'])['quantity'].sum()
                indicators.loc[obook, 'qty_av_av_sell'] = (qty_sell.unstack(level=[0,1])).mean().mean()
                indicators.loc[obook, 'qty_total_sell_MWh'] =qty_sell.sum(min_count=1)/4
            if not self.all_books[obook].cleared_buyorders_all_df.empty:
                qty_buy_cleared= self.all_books[obook].cleared_buyorders_all_df.groupby(by=['delivery_day', 'delivery_time','offer_daytime'])['cleared_quantity'].sum()
                indicators.loc[obook, 'qty_av_av_buy_cleared'] = (qty_buy_cleared.unstack(level=[0,1])).mean().mean()
                indicators.loc[obook, 'qty_total_buy_cleared_MWh'] =qty_buy_cleared.sum(min_count=1)/4
            if not self.all_books[obook].cleared_sellorders_all_df.empty:
                qty_sell_cleared=self.all_books[obook].cleared_sellorders_all_df.groupby(by=['delivery_day', 'delivery_time','offer_daytime'])['cleared_quantity'].sum()
                indicators.loc[obook, 'qty_av_av_sell_cleared'] = (qty_sell_cleared.unstack(level=[0,1])).mean().mean()
                indicators.loc[obook, 'qty_total_sell_cleared_MWh'] =qty_sell_cleared.sum(min_count=1)/4

            qty_sell=self.all_books[obook].sell_sum_quantity
            qty_buy=self.all_books[obook].buy_sum_quantity
            qty_sell_cleared=self.all_books[obook].cleared_sell_sum_quantity
            qty_buy_cleared=self.all_books[obook].cleared_buy_sum_quantity

            indicators.loc[obook, 'price_med_wav_sell'] = (self.all_books[obook].sell_wm_price * qty_sell/(
                    qty_sell.sum(min_count=1))).sum(min_count=1).median()
            indicators.loc[obook, 'price_med_wav_buy'] = (self.all_books[obook].buy_wm_price * qty_buy/(
                    qty_buy.sum(min_count=1))).sum(min_count=1).median()
            indicators.loc[obook, 'price_med_wav_sell_cleared'] = (self.all_books[obook].cleared_sell_wm_price * qty_sell_cleared/(
                    qty_sell_cleared.sum(min_count=1))).sum(min_count=1).median()
            indicators.loc[obook, 'price_med_wav_buy_cleared'] = (self.all_books[obook].cleared_buy_wm_price * qty_buy_cleared/(
                    qty_buy_cleared.sum(min_count=1))).sum(min_count=1).median()

            indicators.loc[obook,'price_med_wav_spread'] =  indicators.loc[obook,
                          'price_med_wav_sell_cleared'] - indicators.loc[obook, 'price_med_wav_buy_cleared']

        indicators['total_qty_cleared [%]'] =((indicators[['qty_total_sell_cleared_MWh',
                  'qty_total_buy_cleared_MWh']].sum(axis=1,min_count=1))/(
            indicators[['qty_total_sell_cleared_MWh','qty_total_buy_cleared_MWh']
                       ].sum(min_count=1).sum(min_count=1))).fillna(value=np.nan) *100

        all_returns =DataFrame(index=['DA_return','ID_return','RD_return', 'BE_return','IB_return'])
        all_profit_loss =DataFrame(index=['total_dispatch_costs', 'profit'])

        for agent in self.model.schedule.agents:
            all_returns= pd.concat([all_returns,agent.financial_return[[
                     'DA_return','ID_return','RD_return', 'BE_return','IB_return']].sum(min_count=1)], axis=1)
            all_profit_loss=pd.concat([all_profit_loss,agent.financial_return[
                ['total_dispatch_costs', 'profit']].sum()],axis=1)

        all_profit_loss=all_profit_loss.sum(axis=1)
        all_profit_loss['system_operations_cost'] = self.model.aGridAndSystemOperator.financial_return['total_return'].sum()
        all_profit_loss.rename({'profit': 'market_profit', 'total_dispatch_costs':'total_dispatch_cost'}, inplace=True)
        all_profit_loss['cost_of_electricity'] =all_profit_loss['total_dispatch_cost'] -all_profit_loss['market_profit']

        all_returns=all_returns.abs().sum(min_count=1,axis=1).fillna(value=np.nan)
        #to avoid double counting of IDM returns (sellers and buyers)
        all_returns['ID_return'] =all_returns['ID_return']/2
        if 'IDM' in indicators.index:
            indicators.loc['IDM', 'return'] = all_returns['ID_return']
        if 'DAM' in indicators.index:
            indicators.loc['DAM', 'return'] = all_returns['DA_return']
        if 'RDM' in indicators.index:
            indicators.loc['RDM', 'return'] = all_returns['RD_return']
        if 'BEM' in indicators.index:
            indicators.loc['BEM', 'return'] = all_returns['BE_return']
        indicators['return [%]'] =indicators['return']/indicators['return'].sum(min_count=1
                                                                                )*100
        return(indicators, all_profit_loss)

    def final_keyfigures(self):
        """report overview of key indicators of a full simulation run.
        This is the place to add new key indicators. However, they shoud be calculated in seperate methods"""
        indicators, allprofitloss =self.interdependence_indicators()
        index_lst=[]
        value_lst=[]
        unit_lst=[]

        index_lst+=list(self.model.aGridAndSystemOperator.system_transactions.sum().index)
        value_lst+=list((self.model.aGridAndSystemOperator.system_transactions.sum(min_count=1)/4
                         ).fillna(value=np.nan).values)
        unit_lst+=['MWh','MWh','MWh','MWh','MWh','MWh','MWh','MWh','MWh']

        index_lst+=list(self.model.aGridAndSystemOperator.imbalances.sum().index)
        value_lst+=list((self.model.aGridAndSystemOperator.imbalances.sum(min_count=1)/4
                         ).fillna(value=np.nan).values)
        unit_lst+=['MWh','MWh','MWh','MWh','MWh','MWh']

        index_lst+=list(self.redispatch_PI().sum().index)
        value_lst+=list((self.redispatch_PI().sum(min_count=1)/4).fillna(value=np.nan).values)
        unit_lst+=['MWh','MWh','MWh','MWh','MWh','MWh','MWh','MWh','MWh','MWh','MWh']

        if not self.redispatch_supply_demand_ratio() is None:
            index_lst+=list(self.redispatch_supply_demand_ratio().index)
            value_lst+=list(self.redispatch_supply_demand_ratio().values)
            unit_lst+=['p.u. of redispatch demand upwards','p.u. of redispatch demand downwards']

        index_lst+=list(self.model.aGridAndSystemOperator.financial_return.sum().index)
        value_lst+=list(self.model.aGridAndSystemOperator.financial_return.sum(min_count=1
                                                                               ).fillna(value=np.nan).values)
        unit_lst+=['€','€','€','€','€']

        index_lst+=list(allprofitloss.index)
        value_lst+=list(allprofitloss.values)
        unit_lst+=['€','€','€','€']


        if (self.model.exodata.DA_residual_load is not None):
            #DA_residual_load  may be none if a flat static profile is chosen.
            index_lst+=list(self.model.exodata.DA_residual_load.set_index(
                    ['delivery_day','delivery_time']).loc[
                    self.model.aGridAndSystemOperator.financial_return.index,['load_DA_cor','residual_load_DA']].sum().index)
            value_lst+=list((self.model.exodata.DA_residual_load.set_index(
                    ['delivery_day','delivery_time']).loc[
                    self.model.aGridAndSystemOperator.financial_return.index,['load_DA_cor','residual_load_DA']].sum()/4).values)
            unit_lst+=['MWh','MWh']
        else:
            index_lst+=['load_DA_cor','residual_load_DA']
            value_lst+=[np.nan, np.nan]
            unit_lst+=['MWh','MWh']

        if (len(index_lst)!=len(value_lst))|(len(index_lst)!=len(unit_lst)):
            raise Exception('final_keyfigures has issues with units.')

        keyfigures = DataFrame([value_lst,unit_lst], index=['value', 'unit']).T
        keyfigures.index=index_lst
        keyfigures.index.name='indicator'

        #remove duplicates
        keyfigures=keyfigures.groupby(keyfigures.index).first()

        return (keyfigures)

    def mark_ups_analysis(self, mode='cleared'):
        """
        Method calculates the total mark-up included in orders and adds it to orderbook.
        If the mark-up is positive, the order price is 'more expensive'.
        If the mark-up is negative the price is 'less expensive', i.e. it is a mark-down.

        Note:
        -  mode determines if mark-up is added for 'offered', 'cleared',or 'all' orders.
        -  SRMC are considered as fundamental cost."""
        print('start mark-up analyses (added to orders dataframe). This can take some minutes')
        #get the srmc per offer
        all_assets=self.model.exodata.get_all_assets()
        def get_srmc_from_asset(asset_id):
            if asset_id in all_assets['asset_id'].values:
                return  all_assets.loc[all_assets['asset_id']==asset_id].iloc[0]['srmc']
            else:
                return np.nan

        for obook in self.all_books.keys():
            print(obook)
            if (mode=='offered')|(mode=='all'):

                self.all_books[obook].sellorders_all_df['mark-up']= self.all_books[
                        obook].sellorders_all_df['price'].sub(self.all_books[
                                obook].sellorders_all_df['associated_asset'].apply(
                    lambda x: get_srmc_from_asset(x)))
                #mark-ups for buy orders need to be reversed
                self.all_books[obook].buyorders_all_df['mark-up']=  - self.all_books[
                        obook].buyorders_all_df['price'].sub(self.all_books[
                                obook].buyorders_all_df['associated_asset'].apply(
                    lambda x: get_srmc_from_asset(x)))
            if (mode=='cleared')|(mode=='all'):

                self.all_books[obook].cleared_sellorders_all_df['mark-up']= self.all_books[
                        obook].cleared_sellorders_all_df['price'].sub(self.all_books[
                                obook].cleared_sellorders_all_df['associated_asset'].apply(
                    lambda x: get_srmc_from_asset(x)))
                #mark-ups for buy orders need to be reversed
                self.all_books[obook].cleared_buyorders_all_df['mark-up']= - self.all_books[
                        obook].cleared_buyorders_all_df['price'].sub(self.all_books[
                                obook].cleared_buyorders_all_df['associated_asset'].apply(
                    lambda x: get_srmc_from_asset(x)))












