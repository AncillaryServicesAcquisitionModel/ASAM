# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 12:25:08 2018
@author: Samuel Glismann

Visualization methods for ASAM

Methods:
    show_asset_schedules(self,header=False)
    show_asset_schedules(self,header=False)
    show_trade_per_agent(self, only_agent =None,header=False)
    show_dispatch_per_agent(self,only_agent =None,header=False)
    show_return_per_agent(self, only_agent =None,header=False)
    show_demand_supply_IDM(self, simulation_daytime, delivery_day, delivery_time,header=False)
    show_system_balance(self,header=False)
    show_redispatch_PI(self,header=False)
    show_redispatch_summary(self,header=False)
    show_cleared_prices(self,header=False)
    show_cost_distribution(self, header=False)


"""
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.ticker import FuncFormatter, MaxNLocator


class Visualizations():
    def __init__(self, model):
        self.model = model
        #directory to store figures
        self.dir = self.model.exodata.output_path
        #Use simulation name for figure names
        self.sname = self.model.exodata.sim_name +'_'

    def show_asset_schedules(self,header=False):
        """
        Method: all asset schedules are plotted in one figure
        """
        print('----plot asset schedules in percentage of Pmax')
        #get the right time for the graph
        day, MTU=   self.model.clock.calc_timestamp_by_steps(self.model.schedule.steps -1, 0)
        all_scheds ={}
        for agent in self.model.schedule.agents:
            for asset in agent.assets['object']:
                name = agent.unique_id + str('_') + asset.assetID
                all_scheds[name]=asset.schedule['commit']/asset.pmax*100

        fig = plt.figure(figsize=(7,5));
        ax = fig.add_subplot(1,1,1)
        i = 1
        key_lst=sorted(all_scheds.keys())
        for key in key_lst:
            #linestyle jumps from '-' to '--' with i jumping from 1 to -1
            linestyle = ["-",'-',"--"]
            df = all_scheds[key]
            titl = 'asset_schedules_at_timestamp Day_{0}_MTU_{1}'.format(day,MTU)
            df.plot(ax = ax,grid = True, style=linestyle[i], subplots = False,
                            rot=45,legend=False, title =titl,
                            label = key)
            i =i*(-1)

        ax.set_ylim(bottom=0, top = 102)
        # Shrink current axis by 30%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.ylabel('% of Pmax')

        strFile ='results/'+titl+'.png'
        fig.savefig(strFile)   # save the figure to file
        plt.close(fig)

    def show_trade_per_agent(self, only_agent =None,header=False):
        """
        Method: all trade schedules are plotted per market party
        """
        print('-----plot show_trade_per_agent')
        #get the right time for the graph
        day, MTU=   self.model.clock.calc_timestamp_by_steps(self.model.schedule.steps -1, 0)

        #stacked trade positions as areas and stacked disaptch as lines per agent
        all_scheds ={}
        ymin_ag =[]
        ymax_ag =[]
        if not only_agent:
            #all agents are plotted
            for agent in self.model.schedule.agents:
                all_scheds[agent.unique_id]=agent.trade_schedule.copy()/1000
                ymin_ag += [min(agent.trade_schedule.min().min()/1000, 0)]
                ymax_ag += [max(agent.trade_schedule.max().max()/1000, 0)]

        else:
            agent = self.model.MP_dict[only_agent]
            #only one agent is plotted
            all_scheds[agent.unique_id]=agent.trade_schedule.copy()/1000
            ymin_ag += [min(agent.trade_schedule.min().min()/1000, 0)]
            ymax_ag += [max(agent.trade_schedule.max().max()/1000, 0)]

        ymin = min(ymin_ag)
        ymax = max(ymax_ag)
        if (ymax != 0):
                    ymax=ymax + 0.1*abs(ymax)
        else:
            ymax=20
        if (ymin !=0 ):
            ymin=ymin - 0.1*abs(ymin)
        else:
            ymin=-20

        key_lst=sorted(all_scheds.keys())
        if len(key_lst) ==1:
            fwidth = 7
            fhight = 6
            number_rows = 1
            number_col = 1
        elif len(key_lst)< 5:
            fwidth = 8
            fhight = 8
            number_rows = 2
            number_col = 2
        else:
            fwidth = 10
            fhight = 11.5
            number_rows = round((len(key_lst)+len(key_lst)%4)/4)
            number_col = 4

        fig = plt.figure(figsize=(fwidth,fhight));
        suptitl='Trade positions at day {} MTU {}'.format(day,MTU)
        if header==True:
            plt.suptitle(suptitl)
        for i in range(len(key_lst)):
            if i == 0:
                choice = True
            else:
                choice = False
            ax = fig.add_subplot(number_rows, number_col, i+1)
            df = all_scheds[key_lst[i]].copy()
            #name change needed for adequaat mask operation below
            df.rename(columns={'forecast_error':'FE'}, inplace=True)
            # Shrink current axis by 30%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width, box.height])

            mask=df[['DA_position','ID_position','RD_position', 'FE']]>=0
            df[['DAM_position','IDM_position','RDM_position','forecast_error']]=df[['DA_position','ID_position','RD_position','FE']].where(mask, np.nan)
            df[['DAM_neg','IDM_neg','RDM_neg','neg_FE']]=df[['DA_position','ID_position','RD_position','FE']].where(~mask, np.nan)

            titl = 'agent {0}'.format(key_lst[i])
            df[['DAM_position','IDM_position','RDM_position','forecast_error']].plot.area(ax = ax,
                            stacked=True, grid = True, subplots = False,
                            legend=False, title =titl, color=[
                                  'mediumaquamarine','cornflowerblue','mediumpurple','orange'])
#                                    'cornflowerblue','mediumpurple','orange'])
            df[['DAM_neg','IDM_neg','RDM_neg','neg_FE']].plot.area(ax = ax,
                            stacked=True, grid = True, subplots = False,
                            legend=False, title =titl, color=[
                                    'mediumaquamarine','cornflowerblue','mediumpurple','orange'])
#                                    'cornflowerblue','mediumpurple','orange'])
            df[['total_trade','total_dispatch','imbalance_position']].plot(ax = ax,
                grid = True, subplots = False, style="--",
                legend=False, title =titl, color=['darkgreen','darkblue','red'])
            if choice==True:
                handels, labels =  ax.get_legend_handles_labels()
            ax.set_ylim(bottom=ymin, top = ymax)
            if (i >=12):
                plt.xlabel('delivery_day, delivery_MTU')
            else:
                plt.xlabel('')
            if (i %4==0):
                plt.ylabel('GW')

        if len(key_lst)>= 5:
            plt.subplots_adjust(top=0.89,
                                bottom=0.04,
                                left=0.09,
                                right=0.97,
                                hspace=0.29,
                                wspace=0.315)
            fig.legend(handels[:7],labels[:7],loc='center', bbox_to_anchor=(0.5, 0.935), ncol =4)


        elif len(key_lst)== 1:
            plt.subplots_adjust(top=0.8,
                                bottom=0.045,
                                left=0.115,
                                right=0.96,
                                hspace=0.29,
                                wspace=0.315)
            fig.legend(handels[:7],labels[:7],loc='center', bbox_to_anchor=(0.5, 0.90), ncol =4)

        else:
            plt.subplots_adjust(
                            top=0.86,
                            bottom=0.04,
                            left=0.09,
                            right=0.97,
                            hspace=0.29,
                            wspace=0.315)
            fig.legend(handels[:7],labels[:7],loc='center', bbox_to_anchor=(0.5, 0.925), ncol =4)


        stamp=str(datetime.now().replace(microsecond=0))
        stamp=stamp.replace('.','')
        stamp=stamp.replace(':','_')
        if only_agent:
            suptitl =suptitl +'_'+agent.unique_id
        fig.savefig(self.dir+self.sname+suptitl+' '+stamp+'.png')   # save the figure to file
        plt.close(fig)

    def show_dispatch_per_agent(self,only_agent =None,header=False):

        """
        Method: all trade schedules are plotted per market party

        Input: with only_agent (string agent name), a single agent can be plotted
        """

        #get the right time for the graph
        day, MTU=   self.model.clock.calc_timestamp_by_steps(self.model.schedule.steps -1, 0)
        print('-----plot show_dispatch_per_agent')
        def format_fn(tick_val, tick_pos):
            """local function to set ticks on x axes"""
            if int(tick_val) in xs:
                return labels[int(tick_val)]
            else:
                return ''
        asset_owners_set={}

        fig = plt.figure(figsize=(10,10));
        suptitl='Asset dispatch at day {} MTU {}'.format(day,MTU)
        if header==True:
            plt.suptitle(suptitl)
        for agent in self.model.schedule.agents:
            all_scheds ={}
            for asset in agent.assets['object']:
                name =  asset.assetID
                all_scheds[name]=asset.schedule['commit']/asset.pmax*100
            asset_owners_set[agent.unique_id]=all_scheds.copy()
        key_lst=sorted(asset_owners_set.keys())
        for i in range(len(key_lst)):
            ax = fig.add_subplot(2,2,i+1)
            asset_lst=sorted(asset_owners_set[key_lst[i]].keys())
            for k in range(len(asset_lst)):
                df = asset_owners_set[key_lst[i]][asset_lst[k]]
                titl = 'Dispatch of agent {0}'.format(key_lst[i],day,MTU)
                xs=range(len(df))
                labels = df.index.values
                ax.xaxis.set_major_formatter(FuncFormatter(format_fn))
                ax.plot(df.values, drawstyle ='steps-pre', label=asset_lst[k])
                plt.title(titl)
                ax.legend(loc="best")
                ax.grid(True)
                ax.set_ylim(bottom=0, top = 105)
                if (i ==2)|(i==3):
                    plt.xlabel('delivery_day, delivery_MTU')
                if (i ==0)|(i==2):
                    plt.ylabel('% of Pmax')

        stamp=str(datetime.now().replace(microsecond=0))
        stamp=stamp.replace('.','')
        stamp=stamp.replace(':','_')
        fig.savefig(self.dir+self.sname+suptitl+' '+stamp+'.png')   # save the figure to file
        plt.close(fig)


    def show_demand_supply_IDM(self, simulation_daytime, delivery_day, delivery_time,header=False):
        """
        Method: plot supply-demand curve of intra-day market
        Input:
            simulation_daytime: tuple of simulation day and simulation time (MTU). e.g. (3,1)
            delivery_day: integer day
            delivery time: integer MTU"""

        print('-----plot show_demand_supply_IDM')
        sellorders = self.model.IDM_obook.sellorders_all_df.set_index('offer_daytime')
        buyorders= self.model.IDM_obook.buyorders_all_df.set_index('offer_daytime')
        allsupply= sellorders.loc[sellorders.index.isin([simulation_daytime])].reset_index().set_index(['delivery_day','delivery_time'])
        supply = allsupply.loc[allsupply.index.isin([(delivery_day, delivery_time)])].sort_values('price').reset_index()[
                                  ['price', 'quantity', 'direction']]
        supply['MW']=supply['quantity'].cumsum()
        alldemand= buyorders.loc[buyorders.index.isin([simulation_daytime])].reset_index().set_index(['delivery_day','delivery_time'])
        demand = alldemand.loc[alldemand.index.isin([(delivery_day, delivery_time)])].sort_values('price', ascending=False).reset_index()[
                                  ['price', 'quantity', 'direction']]
        demand['MW']=demand['quantity'].cumsum()
        #make multicolumn dataframe from supply and demand
        demand_supply_df =pd.concat([demand,supply], axis=0).set_index('direction', append=True)
        demand_supply_df= demand_supply_df.unstack(1).swaplevel(0,1,1).sort_index(1)
        demand_supply_df.sort_index(axis=1, level=[0,1],inplace=True)

        if demand_supply_df.empty:
            print('demand_supply plot for delivery daytime ({},{}) during simulation daytime {}, neither supply nor demand'.format(
                    delivery_day, delivery_time,simulation_daytime))
            return
        #make dataframe that can be plot
        end = demand_supply_df.loc[:,(slice(None),slice('MW','MW'))].max().max()+1
        x= np.linspace(0,end,end+1, endpoint=True)

        df= DataFrame(columns=['x','NaN','sell_price', 'buy_price'])
        df['x']=x
        df.set_index('x', inplace=True)

        try:
            df['sell_price']=demand_supply_df[[('sell','price'),('sell','MW')]].set_index([('sell','MW')])
            df['sell_price']=df['sell_price'].bfill()
        except:
            df['sell_price']=np.nan
        try:
            df['buy_price']=demand_supply_df[[('buy','price'),('buy','MW')]].set_index([('buy','MW')])
            df['buy_price']=df['buy_price'].bfill()
        except:
            df['buy_price']=np.nan
        fig = plt.figure(figsize=(8,5));
        plt.step(x=df.index, y=df['sell_price'],where='pre', label=['supply'])
        plt.step(x=df.index, y=df['buy_price'],where='pre', label=['demand'])
        if header==True:
            titl='simulation daytime {0}, delivery daytime ({1}, {2})'.format(simulation_daytime,delivery_day, delivery_time)
        else:
            titl=''
        plt.title(titl)
        plt.legend(loc="best")
        plt.grid(True)
        plt.xlabel('MW')
        plt.ylabel("Eur/MW")
        plt.tight_layout()
        stamp=str(datetime.now().replace(microsecond=0))
        stamp=stamp.replace('.','')
        stamp=stamp.replace(':','_')
        fig.savefig(self.dir+self.sname+'demand_supply_curve'+str(simulation_daytime)+'_('+str(delivery_day)+'_'+str(delivery_time)+')i_'+stamp+'.png')   # save the figure to file
        plt.close(fig)

    def show_system_balance(self,header=False):
        """
        Method: plot of system balance and aggregated transactions per market
        """
        print('-----plot show_system_balance')
        def format_fn(tick_val, tick_pos):
            """local function to set ticks on x axes"""
            if int(tick_val) in xs:
                return labels[int(tick_val)]
            else:
                return ''

        #get the right time for the graph
        day, MTU=   self.model.clock.calc_timestamp_by_steps(self.model.schedule.steps -1, 0)
        ymin=0
        ymax=0
        sell_labels = []
        buy_labels = []
        shown_labels = []
        #to enable plots with a subset of simulated markets, lables need to be selected
        if self.model.exodata.sim_task['run_DAM[y/n]'][0] =='y':
            sell_labels += ['DAM_sell']
            buy_labels += ['DAM_buy']
            shown_labels += ['DAM']
        if self.model.exodata.sim_task['run_IDM[y/n]'][0] =='y':
            sell_labels += ['IDM_sell']
            buy_labels += ['IDM_buy']
            shown_labels += ['IDM']
        if self.model.exodata.sim_task['run_RDM[y/n]'][0] =='y':
            sell_labels += ['RDM_sell']
            buy_labels += ['RDM_buy']
            shown_labels += ['RDM']
        if self.model.exodata.sim_task['run_BEM[y/n]'][0] =='y':
            sell_labels += ['BEM_sell']
            buy_labels += ['BEM_buy']
            shown_labels += ['BEM']
        df = pd.concat([self.model.rpt.get_system_dispatch(),
                        self.model.aGridAndSystemOperator.system_transactions.loc[:self.model.schedules_horizon.index[-1]],
                        self.model.aGridAndSystemOperator.imbalances.loc[:self.model.schedules_horizon.index[-1]]], axis=1)

        #make sell negative according to convention
        df[sell_labels]=-1*df[sell_labels]
        #for simple plotting (only one legend sell and buy)
        df[shown_labels]=df[sell_labels]

        ymin=min(df[sell_labels].sum(axis=1).min().min(),df['sum_dispatch'].min().min(), ymin)

        ymax=max(df[buy_labels].sum(axis=1).max().max(),df['sum_dispatch'].max().max(), ymax)

        if (ymax>0)|(ymax<0):
                    ymax=int(ymax + 0.1*abs(ymax))
        else:
            ymax=20
        if (ymin>0)|(ymin<0):
            ymin=int(ymin - 0.1*abs(ymin))
        else:
            ymin=-20
        fig = plt.figure(figsize=(8,5));
        ax = fig.add_subplot(1,1,1)

        # Shrink current axis by 30%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width* 0.6, box.height])
        titl = 'System trade and dispatch at day {0},MTU {1}'.format(day, MTU)
        if header==True:
           plt.title(titl)

        df[shown_labels].plot.area(ax=ax,
                            stacked=True, grid = True, subplots = False,
                            legend=True, color=['mediumaquamarine','cornflowerblue','mediumpurple','orange'])

        df[buy_labels].plot.area(ax=ax, stacked=True,
                grid = True, subplots = False,
                legend=False, color=['mediumaquamarine','cornflowerblue','mediumpurple','orange'])
        df[['sum_dispatch','imbalance_redispatch','imbalance_market(scheduled)','imbalance_market(realized)']].plot(
                ax=ax,color=['darkblue','darkviolet','orange','red'],stacked=False,
                grid = True, subplots = False,
                legend=True,  style="--")

        handels, labels = handles, labels = ax.get_legend_handles_labels()

        plt.legend(handels[:8],labels[:8],loc='center left', bbox_to_anchor=(1.00, 0.5))

        ax.set_ylim(bottom=ymin, top = ymax)
        plt.ylabel('MW')
        plt.xlabel('delivery day, MTU')

        stamp=str(datetime.now().replace(microsecond=0))
        stamp=stamp.replace('.','')
        stamp=stamp.replace(':','_')
        fig.savefig(self.dir+self.sname+titl+' '+stamp+'.png')   # save the figure to file
        plt.close(fig)



    def show_return_per_agent(self, only_agent =None,header=False):
        print('-----plot show_return_per_agent')
        #get the right time for the graph
        day, MTU=   self.model.clock.calc_timestamp_by_steps(self.model.schedule.steps -1, 0)
        #to enable plots with a subset of simulated markets, lables need to be selected
        sell_labels = []
        buy_labels = []
        shown_labels = []
        #to enable plots with a subset of simulated markets, lables need to be selected
        if self.model.exodata.sim_task['run_IDM[y/n]'][0] =='y':
            sell_labels += ['DAM_sell']
            buy_labels += ['DAM_buy']
            shown_labels += ['DAM_return']
        if self.model.exodata.sim_task['run_IDM[y/n]'][0] =='y':
            sell_labels += ['IDM_sell']
            buy_labels += ['IDM_buy']
            shown_labels += ['IDM_return']
        if self.model.exodata.sim_task['run_RDM[y/n]'][0] =='y':
            sell_labels += ['RDM_sell']
            buy_labels += ['RDM_buy']
            shown_labels += ['RDM_return']
        if self.model.exodata.sim_task['run_BEM[y/n]'][0] =='y':
            sell_labels += ['BEM_sell']
            buy_labels += ['BEM_buy']
            shown_labels += ['BEM_return']

        all_scheds ={}
        ymin_ag =[]
        ymax_ag =[]
        if not only_agent:
            #all agents are plotted
            for agent in self.model.schedule.agents:
                all_scheds[agent.unique_id]=agent.financial_return.copy()/1000
                ymin_ag += [min(agent.financial_return.min().min()/1000, 0)]
                ymax_ag += [max(agent.financial_return.max().max()/1000, 0)]

        else:
            agent = self.model.MP_dict[only_agent]
            all_scheds[agent.unique_id]=agent.financial_return.copy()/1000
            ymin_ag += [min(agent.financial_return.min().min()/1000, 0)]
            ymax_ag += [max(agent.financial_return.max().max()/1000, 0)]

        ymin = min(ymin_ag)
        ymax = max(ymax_ag)
        if (ymax != 0):
                    ymax=ymax + 0.1*abs(ymax)
        else:
            ymax=20
        if (ymin !=0 ):
            ymin=ymin - 0.1*abs(ymin)
        else:
            ymin=-20

        key_lst=sorted(all_scheds.keys())
        if len(key_lst) ==1:
            fwidth = 7
            fhight = 6
            number_rows = 1
            number_col = 1
        elif len(key_lst)< 5:
            fwidth = 8
            fhight = 8
            number_rows = 2
            number_col = 2
        else:
            fwidth = 10
            fhight = 11.5
            number_rows = round((len(key_lst)+len(key_lst)%4)/4)
            number_col = 4


        fig = plt.figure(figsize=(fwidth,fhight));
        suptitl='Financial returns at day {} MTU {}'.format(day,MTU)
        if header==True:
            plt.suptitle(suptitl)
        for i in range(len(key_lst)):
            if i == 0:
                choice = True
            else:
                choice = False
            ax = fig.add_subplot(number_rows,number_col,i+1)
            df = all_scheds[key_lst[i]]
            mask=df[['DA_return','ID_return','RD_return', 'IB_return']]>=0
            df[['DAM_return','IDM_return','RDM_return','IBM_return']] = df[['DA_return','ID_return','RD_return', 'IB_return']].where(mask, np.nan)
#            import pdb
#            pdb.set_trace()
            df[['DAM_neg','IDM_neg','RDM_neg', 'IBM_neg']] = df[['DA_return','ID_return','RD_return','IB_return']].where(~mask, np.nan)

            titl = 'agent {0}'.format(key_lst[i])

            df[['DAM_return','IDM_return','RDM_return', 'IBM_return']].plot.area(ax = ax,
                            stacked=True, grid = True, subplots = False,
                            legend=False, title =titl, color=[
                                    'mediumaquamarine','cornflowerblue','mediumpurple', 'orange'])
            df[['DAM_neg','IDM_neg','RDM_neg', 'IBM_neg']].plot.area( ax = ax,
                            stacked=True, grid = True, subplots = False,
                            legend=False, title =titl, color=[
                                    'mediumaquamarine','cornflowerblue','mediumpurple','orange'])
            df[['total_return','total_dispatch_costs', 'profit']].plot( ax = ax,
                grid = True, subplots = False, style=["--", "--",'--'],
                legend=False, title =titl, color=['darkgreen','darkblue','red'])
            ax.set_ylim(bottom=ymin, top = ymax)
            if choice==True:
                handels, labels =  ax.get_legend_handles_labels()
            ax.set_ylim(bottom=ymin, top = ymax)
            if (i >=12):
                plt.xlabel('delivery_day, delivery_MTU')
            else:
                plt.xlabel('')
            if (i %4==0):
                plt.ylabel('thousand €')

        if len(key_lst)>= 5:
            plt.subplots_adjust(top=0.89,
                                bottom=0.04,
                                left=0.09,
                                right=0.97,
                                hspace=0.29,
                                wspace=0.315)
            fig.legend(handels[:7],labels[:7],loc='center', bbox_to_anchor=(0.5, 0.935), ncol =4)


        elif len(key_lst)== 1:
            plt.subplots_adjust(top=0.8,
                                bottom=0.045,
                                left=0.115,
                                right=0.96,
                                hspace=0.29,
                                wspace=0.315)
            fig.legend(handels[:7],labels[:7],loc='center', bbox_to_anchor=(0.5, 0.90), ncol =4)

        else:
            plt.subplots_adjust(
                            top=0.86,
                            bottom=0.04,
                            left=0.09,
                            right=0.97,
                            hspace=0.29,
                            wspace=0.315)
            fig.legend(handels[:7],labels[:7],loc='center', bbox_to_anchor=(0.5, 0.925), ncol =4)


        stamp=str(datetime.now().replace(microsecond=0))
        stamp=stamp.replace('.','')
        stamp=stamp.replace(':','_')
        if only_agent:
            suptitl =suptitl +'_'+agent.unique_id
        fig.savefig(self.dir+self.sname+suptitl+' '+stamp+'.png')   # save the figure to file
        plt.close(fig)


    def show_redispatch_PI(self,header=False):
        """
        Method: plot of redispatch performance indicators over time
        (see report method redispatch_PI() for more information on content)"""

        print('-----plot show_redispatch_PI')
        def format_fn(tick_val, tick_pos):
            """local function to set ticks on x axes"""
            if int(tick_val) in xs:
                return labels[int(tick_val)]
            else:
                return ''
        #get the right time for the graph
        day, MTU=   self.model.clock.calc_timestamp_by_steps(self.model.schedule.steps -1, 0)
        fig = plt.figure(figsize=(7,7));
        ax = fig.add_subplot(1,1,1)

        titl = 'Redispatch Performance Indicators at day {0},MTU {1}'.format(day, MTU)

        df = self.model.rpt.redispatch_PI()[['residual_demand_downwards','residual_demand_upwards',
                                 'imbalance_redispatch']]
        if df is None:
            print('redispatch_PI() is None. No plot available')
            return
        else:
            xs=range(len(df))
            labels = df.index.values
            ax.xaxis.set_major_formatter(FuncFormatter(format_fn))

            lineObjects=ax.plot(df.values, drawstyle ='steps-pre', ls='--')

            colors =['darkorange', 'darkred','darkviolet']
            for i in range(len(lineObjects)):
                lineObjects[i].set_color(colors[i])

            # Shrink current axis by 30%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0*(1.6), box.width, box.height])
            ax.grid(True)
            plt.legend(lineObjects, ['residual_demand_downwards','residual_demand_upwards',
                                     'imbalance_redispatch'],loc='center left',
                bbox_to_anchor=(0, -0.15),fancybox=True, shadow=False)

            plt.ylabel('MW (positive residual demand means over-procurement)')
            plt.xlabel('delivery day, MTU')
            stamp=str(datetime.now().replace(microsecond=0))
            stamp=stamp.replace('.','')
            stamp=stamp.replace(':','')
            fig.savefig(self.dir+self.sname+titl+' '+stamp+'.png')   # save the figure to file
            plt.close(fig)

    def show_redispatch_summary(self,header=False):

        """
        Method: plot of redispatch performance indicators summary (bar plot)
        (see report method redispatch_PI() for more information on content)"""
        fig = plt.figure(figsize=(6,6));
        if header==True:
            plt.suptitle('Redispatch Key Performance Indicators')
        key_figures_df = self.model.rpt.redispatch_PI().sum()
        indis =['redispatch_demand', 'redispatch_solved','overproc_downwards', 'underproc_downwards',
               'overproc_upwards', 'underproc_upwards','imbalance_redispatch', 'imbalance_market']
        ymax =key_figures_df.loc[indis].max().max()/1000
        ymin =key_figures_df.loc[indis].min().min()/1000
        if (ymin <0)&(ymin >=-1):
            ymin = -1

        ax = fig.add_subplot(1,1,1)
        (key_figures_df.loc[indis]/1000).plot(
                            ax=ax, kind='bar', legend=False, grid=True, rot=90)
        ax.set_ylim(bottom=ymin *1.02,top=ymax*1.02)

        plt.ylabel('GWh')
        plt.subplots_adjust(top=0.94,
                            bottom=0.285,
                            left=0.125,
                            right=0.79,
                            hspace=0.2,
                            wspace=0.2)
        stamp=str(datetime.now().replace(microsecond=0))
        stamp=stamp.replace('.','')
        stamp=stamp.replace(':','_')
        fig.savefig(self.dir+'redispatch kpi summary'+stamp+'.png')   # save the figure to file
        plt.close(fig)


    def show_cleared_prices(self,header=False):
        """
        Method: plot cleared prices of markets.
        Note: 'RDM_spread_weighted_mean' name is changed to 'RDM_weighted_mean'"""
        print('-----plot show_cleared_prices')
        def format_fn(tick_val, tick_pos):
            """local function to set ticks on x axes"""
            if int(tick_val) in xs:
                return labels[int(tick_val)]
            else:
                return ''

        prices= self.model.rpt.get_cleared_prices()

        fig = plt.figure(figsize=(7,5));
        ax = fig.add_subplot(1,1,1)
        xs=range(len(prices))
        labels = prices.index.values
        ax.xaxis.set_major_formatter(FuncFormatter(format_fn))

        ax.plot(prices['IDM_weighted_mean'].values, drawstyle ='steps-pre', label='IDM_weighted_mean',alpha=0.8)
        ax.plot(prices['RDM_spread_weighted_mean'].values, drawstyle ='steps-pre', label='RDM_weighted_mean',alpha=0.8)
        ax.plot(prices['DAP'].values, drawstyle ='steps-pre', label='DAM',alpha=0.8)
        ax.plot(prices['IBP_short'].values, drawstyle ='steps-pre', label='IBP_short', alpha=0.3)
        ax.plot(prices['IBP_long'].values, drawstyle ='steps-pre', label='IBP_long', alpha=0.3)

        # Shrink current axis by 30%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.grid(True)
        plt.ylabel('€/MWh')
        plt.xlabel('delivery day, MTU')
        if header==True:
            plt.title('Cleared prices')

        stamp=str(datetime.now().replace(microsecond=0))
        stamp=stamp.replace('.','')
        stamp=stamp.replace(':','_')
        fig.savefig(self.dir+self.sname+'Cleared prices_'+stamp+'.png')   # save the figure to file
        plt.close(fig)


    def show_cost_distribution(self, header=False):
        """
        Method: plot various costs and profits
        (for more information on parameters see report method final_keyfigures().

        """
        ## COSTS overview of the simulation
        fig = plt.figure(figsize=(6,6));
        costs= ['total_dispatch_cost', 'market_profit', 'system_operations_cost','cost_of_electricity']

        keyfigures = self.model.rpt.final_keyfigures()

        #If possible the values are 'normalized' with the corrected day-ahead load (+export, - import)
        if keyfigures.loc['load_DA_cor','value']>0:
            keyfigures.loc[costs, 'value'] = keyfigures.loc[costs, 'value'].div(keyfigures.loc['load_DA_cor','value'])
            y_label='€ / net_DA_load_MWh'
            suptitl='Cost and profit per MWh day-ahead load (+exp. -imp.)'
        elif keyfigures.loc['residual_load_DA','value']>0:
            keyfigures.loc[costs, 'value'] = keyfigures.loc[costs, 'value'].div(keyfigures.loc['residual_load_DA','value'])
            y_label='€ / net_DA_residual_load_MWh'
            suptitl='Cost and profit per MWh day-ahead load (+exp. -imp.)'
        else:
            y_label='€'
            suptitl='Cost and profit'
        if header==True:
            plt.suptitle(suptitl)

        ymax=keyfigures.loc[costs, 'value'].abs().max().max()
        ymin = 0
        offset = max(abs(ymin)*0.05, abs(ymax)*0.05)

        ymin=ymin -offset
        ymax=ymax +offset

        ax = fig.add_subplot(1,1,1)

        keyfigures.loc[costs, 'value'].abs().plot(ax=ax, kind='bar', legend=False, grid=True, rot=90)

        ax.set_ylim(bottom=ymin,top=ymax)
        plt.ylabel(y_label)
        plt.xlabel('')
        plt.subplots_adjust(top=0.94,
                    bottom=0.305,
                    left=0.125,
                    right=0.79,
                    hspace=0.2,
                    wspace=0.2)
        stamp=str(datetime.now().replace(microsecond=0))
        stamp=stamp.replace('.','')
        stamp=stamp.replace(':','_')
        fig.savefig(self.dir+'profitloss '+stamp+'.png')   # save the figure to file
        plt.close(fig)


