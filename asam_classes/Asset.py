# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 10:26:57 2017
@author: Samuel Glismann
Asset class of ASAM

Assets are owned by market party agents.
Asset attributes largely correspond to PyPSA attributes (see PyPSA for more explanation).

Assets have furthermore an asset (dispatch) schedule as well as a constraint dataframe.
The constraint dataframe is used by PyPSA for dispatch optimization and by the agent to determine available capacity.

Methods:
    - init()
    - calc_dispatch_constraints(self, this_round_redispatch)
    - get_as_df()

"""
import pandas as pd
from pandas import Series, DataFrame
import numpy as np


class Asset():
    def __init__(self, model, assetname = None, pmax = 1, pmin = 0, location = None,
                 srmc = 0, assetowner=None, ramp_limit_up =None, ramp_limit_down = None,
                 min_up_time =0, min_down_time = 0, start_up_cost =0, shut_down_cost =0,
                 ramp_limit_start_up =1, ramp_limit_shut_down=1):

        self.model = model
        self.assetID = assetname
        self.pmax = pmax #MW
        self.pmin = pmin #MW
        self.location = location #area name
        self.srmc = int(round(srmc,0)) #EUR/MW
        self.assetowner = assetowner
        self.ramp_limit_up = ramp_limit_up #p.u. of pmax: max active power increase from one MTU to next
        self.ramp_limit_down = ramp_limit_down #p.u. of pmax: max active power increase from one MTU to next
        self.ramp_limit_start_up = ramp_limit_start_up #p.u. of pmax: max active power increase from one MTU to next at start up
        self.ramp_limit_shut_down = ramp_limit_shut_down #p.u. of pmax: max active power increase from one MTU to next at shut down
        self.min_up_time = min_up_time #minimum number of subsequent MTU commited
        self.min_down_time = min_down_time #minimum number of subsequent MTU nont commited
        self.start_up_cost = int(round(start_up_cost,0)) #EUR
        self.shut_down_cost = int(round(shut_down_cost,0)) #EUR

        # DataFrame for initial schedules before day-ahead results or redispatch results are available
        init_sched = model.clock.asset_schedules_horizon()
        init_sched['commit'] = 0

        #asset dispatch schedule (including the past, i.e. commitment of past simulation steps)
        self.schedule = DataFrame(columns =['commit','p_max_t','p_min_t'])
        self.schedule['commit'] =init_sched['commit']
        self.schedule['available_up'] = self.pmax
        #initially (before first agent.step) the asset is set to 0.
        self.schedule['available_down'] = 0

        #this variable logs the dispatch schedule during the redispatch bidding.
        #it is the baseline for dispatch limits during the schedule horizon (i.e. past dispatch excluded)
        self.schedule_at_redispatch_bidding = init_sched
        #cumulative contraints per round: previous contraints are taken into account.
        #values of past MTU are excluded
        self.constraint_df =  init_sched
        self.constraint_df['upward_commit'] = 0
        self.constraint_df['downward_commit'] = 0
        self.constraint_df['dispatch_limit'] = 0
        self.constraint_df['previous_dispatch'] = 0
        self.constraint_df['p_max_t'] = self.pmax

        #planned unavailability from scenario, in case the asset name is listed with unavailabilities in pmax p.u.
        self.planned_unavailability = DataFrame()
        if isinstance(self.model.exodata.DA_residual_load, DataFrame):
            if (self.model.exodata.DA_residual_load.columns.isin([self.assetID]).any()):
                self.planned_unavailability = self.model.exodata.DA_residual_load[[
                        'delivery_day','delivery_time','delivery_hour',self.assetID]]
                #from p.u. to MW
                self.planned_unavailability[self.assetID]=self.planned_unavailability[self.assetID]*self.pmax
                self.planned_unavailability.set_index(['delivery_day','delivery_time'], inplace=True)
                #rename unavailability
                self.planned_unavailability.rename(columns={self.assetID : 'p_max_t'}, inplace=True)
                #adjust p_max_t and p_min_t, because otherwise it can happen that pmax <pmin
                self.planned_unavailability['p_max_t'] =self.planned_unavailability['p_max_t'].where(
                        self.planned_unavailability['p_max_t']>=self.pmin, 0)
                self.planned_unavailability['p_min_t'] = Series([self.pmin]*len(self.planned_unavailability),
                                  index=self.planned_unavailability.index).where(self.planned_unavailability[
                                          'p_max_t'] >= self.pmin, 0)

                #adjust constraint_df based on planned unavailability
                self.constraint_df['p_max_t'] = self.planned_unavailability.loc[
                        self.constraint_df.index,'p_max_t'].values
                self.constraint_df['p_min_t'] = self.planned_unavailability.loc[
                        self.constraint_df.index,'p_min_t'].values

    def calc_dispatch_constraints(self, this_round_redispatch):

        """
        Method: administers redispatch transactions in asset constraint dataframe.
        Input: agent.set_asset_commit_constraints () executes this method and
        provides the aggregated redispatch values as dataframe

        In case an asset is associated with a redispatch transaction,
        additional constraints are applicable to the dispatch optimization.

        In case of an upward redispatch transaction, the asset is bound to a
        dispatch above the last dispatch schedule + upward redispatch quantity .
        In case of a downward redispatch, the asset is bound to a dispatch
        below the last disaptch schedule - downward redispatch quantity
        """
        print('calculate dispatch contraints of ',self.assetID)
        #forget the past mtu and add new horizon mtu to constraint_df
        self.constraint_df = self.constraint_df.loc[this_round_redispatch.index]
        #add p_max_t for new schedule horizon
        if self.planned_unavailability.empty:
            self.constraint_df['p_max_t']= self.pmax
            self.constraint_df['p_min_t'] = self.pmin
        else:
            self.constraint_df['p_max_t'] = self.planned_unavailability.loc[
                    this_round_redispatch.index, 'p_max_t'].values
            self.constraint_df['p_min_t'] = self.planned_unavailability.loc[
                    this_round_redispatch.index, 'p_min_t'].values

        self.constraint_df[['commit','upward_commit','downward_commit']] = self.constraint_df[
                ['commit','upward_commit','downward_commit']].fillna(value=0)

        if (this_round_redispatch['commit'] != 0).any():
            #get new redispatch commit (and overwrite previous redispatch commit
            #(previous redispatch is now stored in limit dispatch and asset limits)
            self.constraint_df['commit'] = this_round_redispatch['commit']

            #store all redispatch commitments per direction
            self.constraint_df['upward_commit'] = self.constraint_df[
                    'upward_commit'].fillna(value=0) + self.constraint_df['commit'].where(
                    self.constraint_df['commit']>0, 0)
            self.constraint_df['downward_commit'] = self.constraint_df[
                    'downward_commit'].fillna(value=0) + self.constraint_df['commit'].where(
                    self.constraint_df['commit']<0,0)

            #get dispatch schedule from bidding moment (as reference point)
            self.constraint_df['previous_dispatch'] = self.schedule_at_redispatch_bidding[
                    'commit'].loc[self.model.schedules_horizon.index]
            #because the previous dispatch respected the previous asset constraints
            #there is no need for superposition of the dispatch limit.
            #example: asset dispatch schedule at 90 MW. New redispatch commit +10 MW,
            #          it means that constraint_df allowed for 10 MW sell bids.
            #          no need to check if there was downward redispatch in previous rounds
            self.constraint_df['dispatch_limit'] = (self.constraint_df['previous_dispatch'
                              ] + self.constraint_df['commit']).round(0).astype(int)


            if (self.model.RD_marketoperator.rules['order_types'] == 'limit_block')|(
                    self.model.RD_marketoperator.rules['order_types'] == 'limit_ISP')|(
                            self.model.RD_marketoperator.rules['order_types'] == 'IDCONS_orders'):
                #check dispatch_limit feasibility
                unfeasible = self.constraint_df.loc[(self.constraint_df['dispatch_limit']>self.pmax)|
                        (self.constraint_df['dispatch_limit']<0)]
            elif (self.model.RD_marketoperator.rules['order_types'] == 'all_or_none_block')|(
                    self.model.RD_marketoperator.rules['order_types'] == 'all_or_none_ISP'):
                """ for all-or-none ordertypes the dispatch limit may not lie below pmin"""
                #check dispatch_limit feasibility
                unfeasible = self.constraint_df.loc[(self.constraint_df['dispatch_limit']>self.pmax)|(
                        (self.constraint_df['dispatch_limit']<self.pmin)&(
                                self.constraint_df['dispatch_limit']!=0)&(
                                        self.constraint_df['downward_commit']!=0))]
            else:
                raise Exception ('market rule regarding order type not known to asset class')
            if not unfeasible.empty:
                print('UNFEASIBLE dispatch limit    ',self.assetID)
                stamp=str(self.model.schedules_horizon.index[0][0])+'_'+str(self.model.schedules_horizon.index[0][1])
                writer = pd.ExcelWriter(r'results\unfeasible'+stamp+'.xlsx', engine='xlsxwriter')
                unfeasible.to_excel(writer, sheet_name= 'unfeasible')
                self.constraint_df.to_excel(writer, sheet_name= 'constraint')
                raise Exception ('infeasible redispatch constraints for assets')

        #Use dispatch limit to set p_max_t and p_min_t contraints
        self.constraint_df['p_max_t'] = self.constraint_df['dispatch_limit'].where(
                (self.constraint_df['downward_commit'] != 0), self.constraint_df['p_max_t'])
        self.constraint_df['p_max_t'] = self.constraint_df['p_max_t'].where(
                (self.constraint_df['p_max_t'] >= self.constraint_df['p_min_t']), 0)
        self.constraint_df['p_min_t'] = self.constraint_df['dispatch_limit'].where(
                ((self.constraint_df['upward_commit'] != 0)&(
                        self.constraint_df['dispatch_limit']>self.constraint_df['p_min_t'])),
                        self.constraint_df['p_min_t'])


    def get_as_df(self):
        names = ["agent_id", "asset_id", "pmax", "pmin", "location", "srmc"]
        df = DataFrame ([[self.assetowner, self.assetID, self.pmax, self.pmin, self.location, self.srmc]], columns = names)
        return (df)






