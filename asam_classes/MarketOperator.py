# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 16:18:25 2017
@author: Samuel Glismann

MarketOperator

Market operators have an orderbook class.
The market rules per market have the variables
 - gate_opening_time
 - gate_closure_time
 - acquisition_method
 - pricing_method
 - order_types
 - provider_accreditation

Market opperators generally have matching and clearing methods.
EPEX definition of 'clearing': Financial and physical settlement of transactions.
https://www.epexspot.com/en/glossary#l3

Imbalance settlement operators has no matching method, as no orders are used to determine the outcome.

Please consult the ASAM documentation for more background about the clearing algorithms.
"""
from mesa import Agent, Model
from random import randrange, choice
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import pypsa
from pypsa.opt import l_constraint, l_objective, LExpression, LConstraint

class MarketOperator ():
    def __init__(self, model, orderbook, market_rules):
        self.obook = orderbook
        self.model = model
        #series with rules for the markets
        self.rules = market_rules

class MO_intraday(MarketOperator):
    def __init__(self, model, orderbook, market_rules):
        MarketOperator.__init__(self, model, orderbook, market_rules)
        if self.rules.loc['gate_opening_time']=='D-1, MTU 56':
            #gate opening time is a number of MTU of a day (not rolling)
            self.gate_opening_time = 56
        else:
            raise Exception('inraday gate opening time value not known')
        if self.rules.loc['gate_closure_time']=='deliveryMTU-1':
            #attention: this gate- closure-time is an RELATIVE MTU of until the first delivery MTU (in opposite to single auctions)
            #+1 because the gate closes at the beginnig of this relative MTU.
            self.gate_closure_time = 1 +1
        else:
            raise Exception('inraday gate opening time value not known')

    def match_intraday_orders(self, for_agent_id=None):
        """clear every round all ID orders of the same type (limit orders/all
                or nothing orders, same duration)[continuous double auction]"""
        if ((self.rules['acquisition_method']== 'continous')&(
                (self.rules['order_types']== 'limit_and_market')|(
                self.rules['order_types']== 'limit_market_IDCONS'))&(
                       self.rules['pricing_method']== 'best_price')):
            #note: rank of labels is important in this method.
            labels_in_rank = self.obook.offerlabels + ['rem_vol', 'cleared_quantity','cleared_price', 'matched_order']
            print("...: matching intraday orders")
            # get sorted sell and buy orders
            sellorders = self.obook.get_obook_as_multiindex(selection = 'sellorders', incl_location = False).sort_index(axis=1)
            buyorders = self.obook.get_obook_as_multiindex(selection = 'buyorders',incl_location = False).sort_index(axis=1)
            #block orders are not matched on (only as IDCONS)
            buyorders = buyorders.loc[buyorders['delivery_duration'] == 1].copy()
            sellorders = sellorders.loc[sellorders['delivery_duration'] == 1].copy()

            #extract new orders of the agent that triggered the clearing process
            agent_sells = sellorders.loc[sellorders['agent_id'] == for_agent_id].copy()
            agent_sells.sort_index(inplace = True)
            agent_buys = buyorders.loc[buyorders['agent_id'] == for_agent_id].copy()
            agent_buys.sort_index(inplace = True)

            #remove the agent orders from the book for clearing (increases speed)
            sellorders = sellorders.loc[sellorders['agent_id'] != for_agent_id].copy()
            buyorders = buyorders.loc[buyorders['agent_id'] != for_agent_id].copy()
            #filter on delivery date-time start
            list_of_matches=[]

            for delivery_days, m_orders in sellorders.groupby(level=[0,1]):
                if (m_orders.empty)|(agent_buys.empty):
                    agent_orders=DataFrame()
                else:
                    try:
                        agent_orders = agent_buys.loc[agent_buys.index.isin(m_orders.index)]
                    except:
                        import pdb
                        pdb.set_trace()
                for i in range(len(agent_orders)):
                    #remove orders with remaining quantity 0
                    orders = m_orders.loc[(m_orders['rem_vol'] > 0)&(m_orders['price'] < agent_orders['price'].iloc[i])]
                    #end loop if no orders left for matching
                    if len(orders)==0:
                        break
                    #agent order as list
                    a_order = agent_orders.reset_index()[labels_in_rank].iloc[i].values
                    for k in range(len(orders)):
                        #order as list
                        order = orders.reset_index()[labels_in_rank].iloc[k].values
                        #Order columns are ranked as followed with labls_in_rank:
                        #agent_id,associated_asset,delivery_location,quantity,price,delivery_day,delivery_time,
                        #order_type,init_time,order_id,direction, delivery_duration
                        #rem_vol,cleared_quantity,cleared_price,matched_order
                        rem_qty_orders = order[12]
                        rem_qty_agent_orders = a_order[12]
                        match_vol = min(rem_qty_orders,rem_qty_agent_orders)
                        new_rem_qty_orders= rem_qty_orders - match_vol
                        new_rem_qty_agent_orders= rem_qty_agent_orders - match_vol
                        cleared_price = order[4]
                        matched_id_orders = a_order[9]
                        matched_id_agent_orders = order[9]
                        order[12]= new_rem_qty_orders
                        order[13] = match_vol
                        order[15] = matched_id_orders
                        order[14] = cleared_price
                        a_order[12]= new_rem_qty_agent_orders
                        a_order[13] = match_vol
                        a_order[15] = matched_id_agent_orders
                        a_order[14] = cleared_price
                        list_of_matches.append(a_order.copy())
                        list_of_matches.append(order.copy())

                        #ensure that in the next loop the new remaining value is used
                        m_orders['rem_vol'].loc[m_orders['order_id'] == order[9]] = new_rem_qty_orders
                        if new_rem_qty_agent_orders == 0:
                            break

            for delivery_days, m_orders in buyorders.groupby(level=[0,1]):
                if (m_orders.empty)|(agent_sells.empty):
                    agent_orders=DataFrame()
                else:
                    try:
                        agent_orders = agent_sells.loc[agent_sells.index.isin(m_orders.index)]
                    except:
                        import pdb
                        pdb.set_trace()
                for i in range(len(agent_orders)):
                    #remove orders with remaining quantity 0
                    orders = m_orders.loc[(m_orders['rem_vol'] > 0)&(m_orders['price'] > agent_orders['price'].iloc[i])]
                    #end loop if no orders left for matching
                    if len(orders)==0:
                        break
                    #agent order as list
                    a_order = agent_orders.reset_index()[labels_in_rank].iloc[i].values
                    for k in range(len(orders)):
                        #order as list
                        order = orders.reset_index()[labels_in_rank].iloc[k].values
                        #Order columns are ranked as followed with labls_in_rank:
                        #agent_id0,associated_asset1,delivery_location2,quantity3,price4,delivery_day5,delivery_time6,
                        #order_type7,init_time8,order_id9,direction10, delivery_duration11
                        #rem_vol12,cleared_quantity13,cleared_price14,matched_order15
                        rem_qty_orders = order[12]
                        rem_qty_agent_orders = a_order[12]
                        match_vol = min(rem_qty_orders,rem_qty_agent_orders)
                        new_rem_qty_orders= rem_qty_orders - match_vol
                        new_rem_qty_agent_orders= rem_qty_agent_orders - match_vol
                        cleared_price = order[4]
                        matched_id_orders = a_order[9]
                        matched_id_agent_orders = order[9]
                        order[12]= new_rem_qty_orders
                        order[13] = match_vol
                        order[15] = matched_id_orders
                        order[14] = cleared_price
                        a_order[12]= new_rem_qty_agent_orders
                        a_order[13] = match_vol
                        a_order[15] = matched_id_agent_orders
                        a_order[14] = cleared_price
                        list_of_matches.append(a_order.copy())
                        list_of_matches.append(order.copy())

                        #ensure that in next loop the new remaining value is used
                        m_orders['rem_vol'].loc[m_orders['order_id'] == order[9]] = new_rem_qty_orders
                        if new_rem_qty_agent_orders == 0:
                            break

            if list_of_matches: #list is not empty
                #make DF again
                matched = DataFrame(list_of_matches, columns = labels_in_rank)
                #remove fully matched orders from actual orderbook
                full_match = matched.loc[matched['rem_vol']==0].copy()
                self.obook.remove_matched_orders(full_match)
                #adjust order quantities in orderbook which are partially matched
                part_match = matched.loc[matched['rem_vol']!=0].copy()
                part_match = part_match.loc[~part_match['order_id'].isin(full_match['order_id'])]
                #drop all orders which are several times partially cleared.
                #only keep the last, because this is by definition the smallest order
                part_match.drop_duplicates(subset = 'order_id', keep = 'last', inplace = True)
                self.obook.adjust_partial_match_orders(part_match)
            else:
                matched = DataFrame()
            #remove all market orders
            self.obook.remove_market_orders()
            return (matched)
        else:
            raise Exception('IDM clearing_type, ordertypes, pricing method combination not known')

    def clear_intraday(self, for_agent_id=None):
        self.obook.update_orders()
        invoice = self.match_intraday_orders(for_agent_id=for_agent_id)

        if invoice.empty:
            print("no intraday orders cleared")
        else:
            #dublicate orders with a delivery duration > 1 to ensure correct administration in reports and settlement
            if invoice.loc[invoice['delivery_duration']>1].empty:
                pass
            else:
                blocks = invoice.loc[invoice['delivery_duration']>1]
                for i in range(len(blocks)):
                    df = DataFrame(
                            [blocks.iloc[i]] *(blocks['delivery_duration'].iloc[i] - 1))
                    day_lst, mtu_lst = self.model.clock.calc_delivery_period_range(
                            blocks['delivery_day'].iloc[i],
                            blocks['delivery_time'].iloc[i],
                            blocks['delivery_duration'].iloc[i])
                    df['delivery_day'] = day_lst
                    df['delivery_time'] = mtu_lst
                    invoice = invoice.append(df, ignore_index = True)
            print('setteling matched intraday trades')
            #report to cleared orderbook
            self.obook.cleared_sellorders = self.obook.cleared_sellorders.append(invoice.loc[invoice['direction']== 'sell'])
            self.obook.cleared_buyorders = self.obook.cleared_buyorders.append(invoice.loc[invoice['direction']== 'buy'])
            #calculate money sum due (devide by 4 as very product is 15 minutes= 1 MTU and prices are EUR/MWh)
            invoice['due_amount']= invoice['cleared_quantity'] * invoice['cleared_price']/4
            #make buy due amounts negative
            invoice['due_amount']= invoice['due_amount'].where(invoice['direction']=='sell',-1*invoice['due_amount'])
            for agent in self.model.schedule.agents:
                agent.accepted_ID_orders = agent.accepted_ID_orders.append(invoice.loc[invoice['agent_id'] == agent.unique_id])
                transactions = invoice['due_amount'].loc[invoice['agent_id'] == agent.unique_id].sum()
                agent.money += transactions

class MO_redispatch(MarketOperator):
    def __init__(self, model, orderbook, market_rules):
        MarketOperator.__init__(self, model, orderbook, market_rules)
        if self.rules.loc['gate_opening_time']=='D-1, MTU 56':
            #attention: this gate-opening time is an absolute MTU of a day
            self.gate_opening_time = 56
        else:
            raise Exception('redispatch gate opening time value not known')
        if self.rules.loc['gate_closure_time']=='deliveryMTU-1':
            #attention: this gate- closure-time is an RELATIVE MTU of until the first delivery MTU (in contrast to single auctions)
            #+1 because the gate closes at the beginnig of this relative MTU.
            self.gate_closure_time = 1 + 1
        elif self.rules.loc['gate_closure_time']=='deliveryMTU-2':
            self.gate_closure_time = 2 +1
        elif self.rules.loc['gate_closure_time']=='deliveryMTU-3':
            self.gate_closure_time = 3 +1
        elif self.rules.loc['gate_closure_time']=='deliveryMTU-4':
            self.gate_closure_time = 4 +1
        else:
            raise Exception('redispatch gate closure time value not known')

        if (self.rules.loc['acquisition_method']=='cont_RDM_thinf')|(
                self.rules.loc['acquisition_method']=='cont_RDM_th0')|(
                         self.rules.loc['acquisition_method']=='cont_RDM_th50')|(
                                  self.rules.loc['acquisition_method']=='cont_RDM_th5'):
            #'cont_RDM_th0' is an abbrevation for continuous redispatch market/mechanism
            # with a threshold for 'equilibrium constraint' for upward and downward quantity of 0 MW.
            # the other abbrevations work accordingly.
            #PyPSA model is used for matching.

            #Value of lost load used for slack generators capturing under-procurement
            Voll= 10000
            self.commit_model = pypsa.Network()
            for a in self.model.gridareas:
                self.commit_model.add("Bus", a +'_up', carrier = 'DC')
                self.commit_model.add("Bus", a +'_down', carrier = 'DC')
                self.commit_model.add("Load", 'up_demand_' + a, bus = a + '_up')
                self.commit_model.add("Load",'down_demand_' + a, bus = a + '_down')
                #add a generator that captures under-procurement downward redispatch
                self.commit_model.add("Generator",a+'short_position_down',bus=a +'_down',
                       committable=True,
                       p_min_pu= 0,
                       marginal_cost = Voll,
                       p_nom=10000)
                #add a storage unit to capture over-procurement in downward direction
                self.commit_model.add("Generator",a+'long_position_down',bus=a +'_down',
                                      committable = True,
                                      p_min_pu= 1,
                                      p_max_pu= 0,
                                      p_nom=-10000,
                                      marginal_cost= -1001)
                #add a storage unit to capture over-procurement in upward direction
                self.commit_model.add("Generator",a+'long_position_up',bus=a +'_up',
                                      committable = True,
                                      p_min_pu= 1,
                                      p_max_pu= 0,
                                      p_nom=-10000,
                                      marginal_cost= -1001)
                self.commit_model.add("Generator",a+'short_position_up',bus=a +'_up',
                       committable=True,
                       p_min_pu= 0,
                       marginal_cost = Voll,
                       p_nom=10000)

            #these placeholders are used for a lopf constraint in case there are no orders.
            self.commit_model.add("Generator",'placeholder_gen_down' ,bus=a +'_down',
                   committable=True,
                   p_min_pu= 0,
                   marginal_cost = Voll,
                   p_nom=0)
            self.commit_model.add("Generator",'placeholder_gen_up' ,bus=a +'_up',
                   committable=True,
                   p_min_pu= 0,
                   marginal_cost = Voll,
                   p_nom=0)
        if self.rules.loc['acquisition_method']=='cont_RDM_thinf':
            # there is no constraint regarding the quantity equilibrium of upward and downward redispatch actions per MTU
            self.imb_threshold = np.inf
        elif self.rules.loc['acquisition_method']=='cont_RDM_th0':
            # there is constraint regarding the quantity equilibrium of upward and downward redispatch actions per MTU
            self.imb_threshold = 0
        elif self.rules.loc['acquisition_method']=='cont_RDM_th50':
            # there is a constraint regarding the quantity equilibrium of upward and downward redispatch actions per MTU
            self.imb_threshold = 50
        elif self.rules.loc['acquisition_method']=='cont_RDM_th5':
            # there is a constraintregarding the quantity equilibrium of upward and downward redispatch actions per MTU
            self.imb_threshold = 5
        else:
            self.imb_threshold = np.inf #default

    def match_redispatch_orders (self):
        if (self.rules['acquisition_method']== 'cont_RDM_thinf')|(
                self.rules['acquisition_method']== 'cont_RDM_th0')|(
                         self.rules.loc['acquisition_method']=='cont_RDM_th5')|(
                                  self.rules.loc['acquisition_method']=='cont_RDM_th50'):
            #matched orders df
            matched = DataFrame(columns = self.obook.offerlabels +['due_amount'])
            print("matching redispatch orders with pypsa")
            #taking a copy of the pypsa model for redispatch ensures that no previous orders are included
            commit_model = self.commit_model.copy()

            def redispatch_solution_constraints(network, snapshots):
                #this local method has two local constraint methods for pypsa opf
                def block_constraint(network, snapshots):
                    """Block orders have p_max_pu = 0 for all periods outside their block.
                    the snapshots within the block have p_max_pu = order quantity.
                    This block however should be dispatched with constant quantity within this block.
                    Therefore, a set of constraints is given to the solver where
                    gen_p_t1 ==gen_p_t-1, gen_p_t2 ==gen_p_t1 ...see PyPSA github
                    for more information on structure of additional constraints.

                    Unfortunately, this way of adding contraints works only for PyPSA 0.13.1 or older.
                    To-do: translate this method to new PyPSA versions. Help needed.
                    """
                    #get all snapshots of the block (p_max!=0)
                    block_sn={}
                    for gen in network.generators.index:
                       try:
                           block_sn[gen] = network.generators_t.p_max_pu.loc[network.generators_t.p_max_pu[gen] !=0].index
                       except:
                           block_sn[gen] = None
                    constant_block ={}
                    for gens_i, gen in enumerate(network.generators.index):

                        if block_sn[gen] is not None:
                            affected_sns =  block_sn[gen]
                            affected_sns_shifted =[affected_sns[-1]] + list(affected_sns[:-1])
                            for i in range(len(affected_sns)):
                                lhs = LExpression([(1,network.model.generator_p[gen, affected_sns[i]])])
                                rhs = LExpression([(1,network.model.generator_p[gen, affected_sns_shifted[i]])])
                                constant_block[gen, affected_sns[i]]= LConstraint(lhs,"==",rhs)

                    affected_generators = [k for k, v in block_sn.items() if v is not None]
                    gen_sns_index =[]
                    for gen in affected_generators:
                        for sn in block_sn[gen]:
                            gen_sns_index +=[(gen, sn)]
                    #dictionary of LContraints is given to pypsa.opt.l_constraint (set)
                    l_constraint(network.model, "block_constraint", constant_block,
                                  gen_sns_index )
                def balance_threshold (network, snapshots):
                    #all orders up and down need to be selected
                    gen_up = list(network.generators.loc[network.generators.index.isin(
                            list(upsupply['order_id']))].index)
                    gen_down = list(network.generators.loc[network.generators.index.isin(
                            list(downsupply['order_id']))].index)
                    #solver needs at least one variable. Therefore these 0MW placeholders are used,
                    #in case there are no orders in a direction
                    if not gen_up:
                       gen_up =['placeholder_gen_up']
                    if not gen_down:
                       gen_down =['placeholder_gen_down']

                    imb_upper={}
                    imb_lower={}
                    for sn in snapshots:
                        rhs = LExpression([(1,sum(network.model.generator_p[gen,sn] for gen in gen_up))]) +  self.imb_threshold
                        lhs =  LExpression([(1,sum(network.model.generator_p[gen,sn] for gen in gen_down))])
                        imb_upper[sn]= LConstraint(lhs,"<=",rhs)

                        lhs = LExpression([(1,sum(network.model.generator_p[gen,sn] for gen in gen_up))])
                        rhs =  LExpression([(1,sum(network.model.generator_p[gen,sn] for gen in gen_down))]) +  self.imb_threshold
                        imb_lower[sn]= LConstraint(lhs,"<=",rhs)
                    l_constraint(network.model, "imbalance_constraint_upper", imb_upper, list(snapshots))
                    l_constraint(network.model, "imbalance_constraint_lower", imb_lower, list(snapshots))
                #execute both constraint methods
                if (self.rules['order_types']== 'limit_ISP')|(
                    self.rules['order_types']== 'limit_block')|(
                           self.rules['order_types']== 'IDCONS_orders'):
                    block_constraint(network, snapshots)
                balance_threshold (network, snapshots)

            #set the minimum order clearing
            if (self.rules['order_types']== 'all_or_none_ISP')|(
                    self.rules['order_types']== 'all_or_none_block'):
                pmin = 1 #because all-or-none
            elif (self.rules['order_types']== 'limit_ISP')|(
                    self.rules['order_types']== 'limit_block')|(
                           self.rules['order_types']== 'IDCONS_orders') :
                pmin = 0 #because orders are limit orders (partial call possible)

            if not self.rules['order_types']== 'IDCONS_orders':
                #get all available orders for redispatch
                buyorders = self.obook.get_obook_as_multiindex(selection='buyorders', incl_location = True)
                sellorders = self.obook.get_obook_as_multiindex(selection='sellorders', incl_location = True)

                #exclude all block orders in case only orders per ISP are allowed
                if (self.rules['order_types']== 'all_or_none_ISP')|(
                        self.rules['order_types']== 'limit_ISP'):
                    #exclude ordertypes block-orderes (delivery duration > 1)
                    buyorders = buyorders.loc[buyorders['delivery_duration'] == 1].copy()
                    sellorders = sellorders.loc[sellorders['delivery_duration'] == 1].copy()

                downdemand = buyorders.loc[buyorders['order_type'] == 'redispatch_demand']
                updemand = sellorders.loc[sellorders['order_type'] == 'redispatch_demand']
                downsupply = buyorders.loc[buyorders['order_type'] == 'redispatch_supply']
                upsupply = sellorders.loc[sellorders['order_type'] == 'redispatch_supply']
            else:
                #in case of IDCONS, the orders are retrieved from the intraday orderbook (not from the redispatch orderbook)
                downsupply = self.model.IDM_obook.get_obook_as_multiindex(selection='buyorders', incl_location = True)
                upsupply = self.model.IDM_obook.get_obook_as_multiindex(selection='sellorders', incl_location = True)
                #filter on IDCONS_orders
                downsupply= downsupply.loc[downsupply['order_type']=='IDCONS_order'].copy()
                upsupply= upsupply.loc[upsupply['order_type']=='IDCONS_order'].copy()
                #get the redispatch demand orders from the redispatch orderbook
                downdemand = self.obook.get_obook_as_multiindex(selection='buyorders', incl_location = True)
                updemand = self.obook.get_obook_as_multiindex(selection='sellorders', incl_location = True)
                #ensure that only redispatch demand orders are involved
                downdemand =downdemand.loc[downdemand['order_type'] == 'redispatch_demand'].copy()
                updemand =updemand.loc[updemand['order_type'] == 'redispatch_demand'].copy()

            if (updemand.empty & downdemand.empty) |(upsupply.empty & downsupply.empty):
                #no need to calculate anything
                return (DataFrame())

            upsupply.reset_index(inplace=True)
            downsupply.reset_index(inplace=True)
            #calculate end delivery mtu of orders
            upsupply['end_delivery_mtu'] =  upsupply.apply(lambda x: (x['delivery_location'],)
                    + self.model.clock.calc_delivery_period_end((x['delivery_day'],x['delivery_time']
                    ), x['delivery_duration']),axis=1)
            downsupply['end_delivery_mtu'] =  downsupply.apply(lambda x:(x['delivery_location'],)
                    +  self.model.clock.calc_delivery_period_end((x['delivery_day'],x['delivery_time']
                    ), x['delivery_duration']),axis=1)

            upsupply = upsupply.set_index(['delivery_location', 'delivery_day', 'delivery_time'])
            downsupply = downsupply.set_index(['delivery_location', 'delivery_day', 'delivery_time'])
            updemand = updemand.reset_index().set_index(['delivery_location', 'delivery_day', 'delivery_time'])
            downdemand = downdemand.reset_index().set_index(['delivery_location', 'delivery_day', 'delivery_time'])

            ##filter supply orders to keep only supply with demand-overlapping delivery periods
            upsupply = upsupply[(upsupply.index.isin(updemand.index))|(
                    upsupply['end_delivery_mtu'].isin(updemand.index))]
            downsupply = downsupply[(downsupply.index.isin(downdemand.index))|(
                    downsupply['end_delivery_mtu'].isin(downdemand.index))]

            upsupply=upsupply.reset_index()
            downsupply=downsupply.reset_index()

            if (updemand.empty & downdemand.empty) |(upsupply.empty & downsupply.empty):
                #no need to calculate anything
                return (DataFrame())
            #add snapshots for the calculation
            indx = list(self.model.schedules_horizon.index.values)
            snap = DataFrame(index=self.model.schedules_horizon.index)
            snap = snap.reset_index()
            snap['strIndex']=snap['delivery_day'].map(str)+str('_')+snap['delivery_time'].map(str)
            commit_model.set_snapshots(snap['strIndex'])

            #all remaining supply orders are added as generators to the pypsa model
            for i in range(len(upsupply)):
                order = upsupply.iloc[i]
                commit_model.add('Generator',order['order_id'] ,bus = order['delivery_location'] + '_up',
                committable = True,
                #pmin is either 0 (limit orders) or 1 (all-or-none orders)
                p_min_pu = pmin,
                min_up_time = order['delivery_duration'],
                marginal_cost = order['price'],
                p_nom = order['quantity'])
                #use schedules horizon to make delivery period
                delivery_period= DataFrame(columns= ['pmax_pu'], index=self.model.schedules_horizon.index)
                delivery_period['pmax_pu'] = 0
                #get index value from the list of schedules_horizon index of delivery period start
                start = indx.index((order['delivery_day'], order['delivery_time']))
                delivery_duration= order['delivery_duration']
                end = start + delivery_duration
                delivery_period.loc[indx[int(start):int(end)], 'pmax_pu'] = 1
                commit_model.generators_t.p_max_pu[order['order_id']]=list(delivery_period['pmax_pu']).copy()

            for i in range(len(downsupply)):
                order = downsupply.iloc[i]
                commit_model.add('Generator',order['order_id'] ,bus=order['delivery_location'] + '_down',
                committable = True,
                p_min_pu = pmin,
                min_up_time = order['delivery_duration'],
                #make price negative to consider that downward are buy orders
                #(provider pays when price is positive)
                marginal_cost = -order['price'],
                p_nom = order['quantity'])
                #use schedules horizon to make delivery period
                delivery_period= DataFrame(columns= ['pmax_pu'], index=self.model.schedules_horizon.index)
                delivery_period['pmax_pu'] = 0
                #get index value from the list of schedules_horizon index of delivery period start
                start = indx.index((order['delivery_day'], order['delivery_time']))
                delivery_duration= order['delivery_duration']
                end = start + delivery_duration
                delivery_period.loc[indx[int(start):int(end)], 'pmax_pu'] = 1
                commit_model.generators_t.p_max_pu[order['order_id']]=list(delivery_period['pmax_pu']).copy()

            #prepare redispatch demand per area.
            downdemand_per_area = pd.concat([DataFrame(index=self.model.schedules_horizon.index), downdemand['quantity'].unstack(level=0)], axis=1)
            updemand_per_area = pd.concat([DataFrame(index=self.model.schedules_horizon.index), updemand['quantity'].unstack(level=0)],axis =1)

            for area in downdemand_per_area.columns:
                commit_model.loads_t.p_set['down_demand_'+area] = list(downdemand_per_area[area])
            for area in updemand_per_area.columns:
                commit_model.loads_t.p_set['up_demand_'+ area] = list(updemand_per_area[area])
            commit_model.loads_t.p_set.fillna(value=0, inplace=True)
            #run PyPSA
            commit_model.lopf(commit_model.snapshots, solver_name= self.model.exodata.solver_name,
                                  extra_functionality = redispatch_solution_constraints, free_memory={'pypsa'})
            generators = commit_model.generators_t.p
            generators = generators.loc[:,(generators>0).any(axis=0)].copy()
            #get cleared quantity per order. Mean instead of sum, because of possible block orders
            cleared_quantity= {}
            for c in generators.columns:
                cleared_quantity[c] = generators[c].loc[generators[c]>0].mean()

            ##select matched orders based on order_id list
            matched = downsupply.loc[downsupply['order_id'].isin(list(cleared_quantity.keys()))]
            matched = pd.concat([matched,upsupply.loc[upsupply['order_id'].isin(list(cleared_quantity.keys()))]])

            for oid in list(cleared_quantity.keys()):
                matched['cleared_quantity'].loc[matched['order_id'] == oid] = cleared_quantity[oid]

            #calculate remaining quantity of limit orders
            matched['rem_vol'] = matched['quantity'] - matched['cleared_quantity']
            #pay as bid
            matched['cleared_price'] = matched['price']

            if self.rules['order_types']== 'IDCONS_orders':
                #remove fully matched IDCONS orders from ID obook
                full_match = matched.loc[matched['rem_vol']==0].copy()
                self.model.IDM_obook.remove_matched_orders(full_match)
                #adjust order quantities in orderbook which are partially matched
                part_match = matched.loc[matched['rem_vol']!=0].copy()
                part_match = part_match.loc[~part_match['order_id'].isin(full_match['order_id'])]
                #drop all orders which are several times partially cleared.
                #only keep the last, because this is the smallest by definition
                part_match.drop_duplicates(subset = 'order_id', keep = 'last', inplace = True)
                self.model.IDM_obook.adjust_partial_match_orders(part_match)

            #remove matched supply orders from all order list
            self.obook.remove_matched_orders(matched)
            return (matched)

        else:
            raise Exception('redispatch clearing type - ordertype combination not known')

    def clear_redispatch(self):
        self.obook.update_orders()
        invoice = self.match_redispatch_orders()
        invoice['due_amount'] = None

        if invoice.empty:
            print("no redispatch orders cleared")
        else:
            #dublicate orders with a delivery duration > 1 to ensure correct administration in reports and settlement
            if invoice.loc[invoice['delivery_duration']>1].empty:
                pass
            else:
                blocks = invoice.loc[invoice['delivery_duration']>1]
                for i in range(len(blocks)):
                    df = DataFrame(
                            [blocks.iloc[i]] * int(blocks['delivery_duration'].iloc[i] - 1))
                    day_lst, mtu_lst = self.model.clock.calc_delivery_period_range(
                            blocks['delivery_day'].iloc[i],
                            blocks['delivery_time'].iloc[i],
                            blocks['delivery_duration'].iloc[i])
                    df['delivery_day'] = day_lst
                    df['delivery_time'] = mtu_lst
                    invoice = invoice.append(df, ignore_index = True)

            #report to cleared orderbook
            self.obook.cleared_sellorders = self.obook.cleared_sellorders.append(invoice.loc[invoice['direction']== 'sell'])
            self.obook.cleared_buyorders = self.obook.cleared_buyorders.append(invoice.loc[invoice['direction']== 'buy'])
            if self.rules['pricing_method']== 'pay_as_bid':
                invoice['due_amount']= invoice['cleared_quantity'] * invoice['cleared_price']/4
                #make buy due amounts negative
                invoice['due_amount']= invoice['due_amount'].where(invoice['direction']=='sell',-1*invoice['due_amount'])
            else:
               raise Exception('redispatch pricing method not known')
            for agent in self.model.schedule.agents:
                transaction = invoice.loc[invoice['agent_id'] == agent.unique_id, 'due_amount'].sum()
                agent.accepted_red_orders = invoice[invoice['agent_id'] == agent.unique_id]
                agent.money += transaction
                print("agent {} gets {} Euro for redispatch".format(agent.unique_id, transaction))
                self.model.aGridAndSystemOperator.money -= transaction
            #provide matched orders to aGridAndSystemOperator for redispatch demand determination in next round
            self.model.aGridAndSystemOperator.red_procured = invoice.copy()


class MO_dayahead(MarketOperator):
    def __init__(self, model, orderbook, market_rules):
        MarketOperator.__init__(self, model, orderbook, market_rules)
        #for initial asset status approximation
        self.test_init_dispatch=DataFrame()
        if self.rules.loc['gate_opening_time']=='D-1, MTU 44':
            #attention: this gate-opening time is an absolute MTU of a day
            self.gate_opening_time = 44
        else:
            raise Exception('DA gate opening time value not known')
        if self.rules.loc['gate_closure_time']=='D-1, MTU 45':
            #attention: this gate closure-time is ALSO an absolute MTU of a day (in contrast to continous markets)
            self.gate_closure_time = 45
        else:
            raise Exception('DA gate opening time value not known')
        if self.rules['acquisition_method']== 'single_hourly_auction':
            #initiate pypsa model for optimal unit commitment.
            #Note that this method assumes a single-sided auction (inelastic demand) instead of a double-sided auction (with elastic demand/ demand response)
            #value of lost load
            Voll= 10000
            self.commit_model = pypsa.Network()
            self.commit_model.add("Bus","bus")
            self.commit_model.add("Load","load",bus="bus")
            #add a generator that captures the unfeasible demand (open long position)
            self.commit_model.add("Generator",'short_position' ,bus="bus",
                   committable=True,
                   p_min_pu= 0,
                   marginal_cost = Voll,
                   p_nom=10000)
            #add a storage unit to capture unfeasible demand (open short positions)
            self.commit_model.add("Generator",'long_position',bus="bus",
                   committable = True,
                   p_min_pu= 1,
                   p_max_pu= 0,
                   p_nom=-10000,
                   marginal_cost= -Voll)
            all_assets = self.model.exodata.asset_portfolios
            for i in range(len(all_assets)):
                self.commit_model.add("Generator",all_assets.loc[i,'asset_name'] ,bus="bus",
                       committable = True,
                       p_min_pu = all_assets.loc[i,'pmin']/all_assets.loc[i,'pmax'],
                       marginal_cost = all_assets.loc[i,'srmc'],
                       p_nom=all_assets.loc[i,'pmax']
                       )
                #intertemp constraints devided by 4 because it is from mtu to hourly.
                self.commit_model.generators.start_up_cost[all_assets.loc[i,'asset_name']] = all_assets.loc[i,'start_up_cost']/4
                self.commit_model.generators.shut_down_cost[all_assets.loc[i,'asset_name']] = all_assets.loc[i,'shut_down_cost']/4
                #Other intertemp constraints not considered in DA, as tests showed that the results are not improving.
                #    self.commit_model.generators.min_up_time[all_assets.loc[i,'asset_name']] = all_assets.loc[i,'min_up_time']/4
                #    self.commit_model.generators.min_down_time[all_assets.loc[i,'asset_name']] = all_assets.loc[i,'min_down_time']/4
                #    self.commit_model.generators.ramp_limit_down[all_assets.loc[i,'asset_name']] = all_assets.loc[i,'ramp_limit_down']/4
                #    self.commit_model.generators.ramp_limit_up[all_assets.loc[i,'asset_name']] = all_assets.loc[i,'ramp_limit_up']/4

    def match_dayahead_orders(self):
        if (self.rules['acquisition_method']== 'exo_default')|(self.rules['acquisition_method']== 'exo_imbalance_case'):
            return(None) #no clearing needed
        if self.rules['acquisition_method']== 'single_hourly_auction':
            #make hourly snapshots. The MTU of the day and the step of the simulation determines which snapshots are used.
            if (self.model.clock.get_MTU() >= self.gate_closure_time) & (self.model.schedule.steps == 0)&(
                    self.model.schedules_horizon.index.get_level_values(0)[-1] >= self.model.clock.get_day()+1):
                #horizon includes next day and all hours of remaining current day
                day = [self.model.clock.get_day()]*(24 - self.model.clock.get_hour() + 1) + [
                        self.model.clock.get_day() + 1] * 24
                hours = list (range(self.model.clock.get_hour(),25)) + list(range(1,25))
            elif (self.model.clock.get_MTU() < self.gate_closure_time) & (self.model.schedule.steps == 0):
                #horizon includes only current day
                day = [self.model.clock.get_day()]*(24 - self.model.clock.get_hour() + 1)
                hours = list (range(self.model.clock.get_hour(),25))
            elif (self.model.schedule.steps == 0)&(
                    self.model.schedules_horizon.index.get_level_values(0)[-1] < self.model.clock.get_day()+1):
                #when first step but horizon only includes current day
                day = [self.model.clock.get_day()]*(24 - self.model.clock.get_hour() + 1)
                hours = list(range(self.model.clock.get_hour(),25))
            elif (self.model.clock.get_MTU() == self.gate_closure_time)&(
                    self.model.schedules_horizon.index.get_level_values(0)[-1] >= self.model.clock.get_day()+1):
                #when DA GCT and next day is in the horizon of the simulation
                day = [self.model.clock.get_day()+1]*(24)
                hours = list(range(1,25))
            else:
                #horizon shows that simulation is at end. No additional DA auction
                return (DataFrame(), DataFrame())
            snap= DataFrame(columns =['delivery_day','delivery_hour'])
            snap['delivery_day']=day
            snap['delivery_hour']=hours
            snap['strIndex'] = snap['delivery_day'].map(str)+str('_')+snap['delivery_hour'].map(str)
            #get the residual load fom exo-data class
            resload_lst = self.model.exodata.get_DA_resload(snap, mode = self.model.exodata.sim_task['residual_load_scenario'])
            if not resload_lst:
                print("no residual load values in exo database for DA auction horizon")
                print("auction not executed")
                return(DataFrame(), DataFrame())
            elif len(resload_lst) < len(snap):
                print("not sufficient residual load values in exo database for the complete DA auction horizon. Auction horizon reduced to data")
                snap = snap.iloc[:len(resload_lst)]
            self.commit_model.set_snapshots(snap['strIndex'])
            self.commit_model.loads_t.p_set['load']=  resload_lst

            #consider unavailabilities of assets:
            indx = snap.set_index(['delivery_day','delivery_hour']).index
            for agent in self.model.schedule.agents:
                for asset in agent.assets['object']:
                    if not asset.planned_unavailability.empty:
                        #convert unavailabilities in from mtu to mean per hour
                        unav_h = (asset.planned_unavailability.reset_index().groupby(by=['delivery_day','delivery_hour']).mean()/asset.pmax).copy()
                        self.commit_model.generators_t.p_max_pu[asset.assetID] = unav_h.loc[indx,'p_max_t'].values.copy()
                        self.commit_model.generators_t.p_min_pu[asset.assetID] = unav_h.loc[indx,'p_min_t'].values.copy()

            #daytime of last dispatch for initial generator status determination
            if not self.model.rpt.prices['DAP'].empty:
                init_time= self.model.rpt.prices['DAP'].index[-1]
            else:
                # a test run is needed to determine initial dispatch
                init_time= None
                try:
                    #default is status 1 for start
                    self.commit_model.generators.initial_status = 1
                    self.commit_model.lopf(self.commit_model.snapshots, solver_name= self.model.exodata.solver_name, free_memory={'pypsa'})
                    self.test_init_dispatch = self.commit_model.generators_t.p.copy().iloc[0]
                    self.test_init_dispatch.loc[self.test_init_dispatch > 0] = 1
                    self.test_init_dispatch.loc[self.test_init_dispatch == 0] = 0

                except:
                    import pdb
                    pdb.set_trace()
            for agent in self.model.schedule.agents:
                for asset in agent.assets['object']:
                    #ensure that the initial status taken into account
                    if init_time:
                        last_dispatch = asset.schedule.loc[init_time, 'commit']
                        if last_dispatch > 0:
                            self.commit_model.generators.initial_status[asset.assetID] = 1
                        else:
                            self.commit_model.generators.initial_status[asset.assetID] = 0
                    else:
                        self.commit_model.generators.initial_status[asset.assetID]= self.test_init_dispatch[asset.assetID]

            try:
                self.commit_model.lopf(self.commit_model.snapshots, solver_name= self.model.exodata.solver_name, free_memory={'pypsa'})
            except:
                import pdb
                pdb.set_trace()
            opt_dispatch = self.commit_model.generators_t.p.copy()
            clearing_prices= opt_dispatch.copy()
            opt_dispatch['DA_residual_load'] = self.commit_model.loads_t.p.copy()
            #convert index back again from PyPSA style to ASAM style.
            opt_dispatch.index = opt_dispatch.index.str.split(pat='_', expand =True)
            opt_dispatch.index.set_names(['delivery_day','delivery_hour'], inplace=True)
            #make inters from index
            opt_dispatch.reset_index(inplace=True)
            opt_dispatch[['delivery_day','delivery_hour']] = opt_dispatch[['delivery_day','delivery_hour']].astype('int64')
            #determine clearing price (highest cost (order price) of dispatched generator)
            costs= self.commit_model.generators['marginal_cost']
            for generator in costs.index:
                clearing_prices[generator] =clearing_prices[generator].where(
                        clearing_prices[generator]==0, costs[generator])
            clearing_prices['clearing_price'] = clearing_prices.max(axis=1)
            #drop all other columns
            clearing_prices = clearing_prices['clearing_price']
            #convert index again
            clearing_prices.index =  clearing_prices.index.str.split(pat='_', expand =True)
            clearing_prices.index.set_names(['delivery_day','delivery_hour'], inplace=True)
            #make inters from index
            clearing_prices=clearing_prices.reset_index()
            clearing_prices[['delivery_day','delivery_hour']] = clearing_prices[['delivery_day','delivery_hour']].astype('int64')

            #this reporting method converts the hourly prices to MTU prices
            self.model.rpt.publish_DAM_prices(clearing_prices)
            if (opt_dispatch['short_position']>0).any():
                print('DA dispatch with adequacy issues short')
                print(opt_dispatch['short_position'])
                import pdb
                pdb.set_trace()
            if (opt_dispatch['long_position']>0).any():
                print('DA dispatch with adequacy issues long')
                print(opt_dispatch['long_position'])
                import pdb
                pdb.set_trace()

            return (opt_dispatch, clearing_prices)

        else:
            raise Exception ('DA celaring type not known')

    def clear_dayahead(self):
        if self.rules['acquisition_method']== 'single_hourly_auction':
            #Note: this method does not include the possibility to buy electricity on DAM.
            #Only producing assets considered on DA (i.e. single-sided auction instead of double-sided auction)
            if (self.model.clock.get_MTU() == self.gate_closure_time)|(
                    self.model.schedule.steps == 0):
                print('clear and settle Day ahed market with a single doublesided auction (based on PyPSA)')

                matches, clearing_prices = self.match_dayahead_orders()
                if (matches.empty) & (clearing_prices.empty):
                    #no auction results. No settlement needed
                    return
                #remove generators with 0 dispatch
                matches = matches.loc[:,(matches > 0).any()]
                #add clearing price to matches df
                matches['cleared_price'] = clearing_prices.iloc[:,2].values
                #enlarge the matches to get results per mtu of 15 minutes
                matches = pd.concat([matches,matches,matches,matches])
                matches.sort_index(inplace=True)
                if (self.model.clock.get_MTU() >= self.gate_closure_time) & (
                        self.model.schedule.steps == 0):
                    #get the right delivery mtu from the schedules horizon
                    mtus = list(self.model.schedules_horizon.index.get_level_values(1))
                elif  (self.model.clock.get_MTU() < self.gate_closure_time) & (
                        self.model.schedule.steps == 0):
                    mtus = list(range(self.model.clock.get_MTU(),97))
                elif self.model.clock.get_MTU() == self.gate_closure_time:
                    mtus = list(range(1,97))
                #cut of MTUs of a first hour that lies in the past.
                matches = matches.iloc[len(matches)-len(mtus):]
                matches['delivery_time'] = mtus
                matches.drop('delivery_hour', axis=1, inplace = True)
                #make artificially cleared DA orders
                for agent in self.model.schedule.agents:
                    assets = list(self.model.exodata.asset_portfolios.loc[
                            self.model.exodata.asset_portfolios['asset_owner'] == agent.unique_id,'asset_name'])
                    for i in assets:
                        DA_cl_orders = DataFrame()
                        DA_cl_orders[['delivery_day', 'delivery_time']] =  matches[['delivery_day', 'delivery_time']]
                        DA_cl_orders['associated_asset'] = i
                        DA_cl_orders['direction'] = 'sell'
                        DA_cl_orders['agent_id'] = agent.unique_id
                        DA_cl_orders['order_type'] = 'limit order'
                        #hourly auctions (4mtu), but we split them for the administration in 15 minute mtus
                        DA_cl_orders['delivery_duration'] = 1
                        if i in matches.columns:
                            #divide cleared order quantity by 4 to cop with the split from hour to mtu
                            DA_cl_orders['cleared_quantity'] = matches[i]
                            DA_cl_orders['cleared_price']= matches['cleared_price']
                            #right order format. missing columsn are nan. which is not an issue.
                            DA_cl_orders = DA_cl_orders.loc[:,['agent_id','associated_asset',
                                                               'delivery_location','cleared_quantity',
                                                               'cleared_price', 'delivery_day','delivery_time',
                                                               'order_type','init_time', 'order_id', 'direction', 'delivery_duration']]
                            #cleared price divided by 4 because its a price per MWh and we split the hours to 15 minutes mtu
                            DA_cl_orders['due_amount']= DA_cl_orders['cleared_quantity'] * DA_cl_orders['cleared_price']/4
                            agent.money += DA_cl_orders['due_amount'].sum()
                            agent.accepted_DA_orders = pd.concat([agent.accepted_DA_orders,
                                                                  DA_cl_orders])
                            #add orders to reporter
                            self.obook.cleared_sellorders = self.obook.cleared_sellorders.append(DA_cl_orders)
                            #also add the orders to the artificial sellorderbook
                            DA_cl_orders = DA_cl_orders.drop(['cleared_quantity','cleared_price', 'due_amount'], axis=1).copy()
                        DA_cl_orders['quantity']= agent.assets.loc[i].item().pmax
                        DA_cl_orders['price']= agent.assets.loc[i].item().srmc
                        DA_cl_orders = DA_cl_orders.loc[:,['agent_id','associated_asset',
                                                           'delivery_location','quantity',
                                                           'price', 'delivery_day','delivery_time',
                                                           'order_type','init_time', 'order_id', 'direction', 'delivery_duration']]
                        #add orders to reporter
                        self.obook.sellorders_full_step = self.obook.sellorders_full_step.append(DA_cl_orders)


                #administration of 'Central_DA_residual_load_entity' for DA buy orders
                #this is a simplification.
                DA_cl_orders = DataFrame()
                DA_cl_orders['cleared_quantity'] = matches['DA_residual_load']
                DA_cl_orders[['delivery_day', 'delivery_time','cleared_price']] =  matches[['delivery_day', 'delivery_time','cleared_price']]
                DA_cl_orders['associated_asset'] = 'DA_residual_load'
                DA_cl_orders['direction'] = 'buy'
                DA_cl_orders['agent_id'] = 'Central_DA_residual_load_entity'
                DA_cl_orders['order_type'] = 'limit order'

                #right order format. Missing columns are nan, which is not an issue.
                DA_cl_orders = DA_cl_orders.loc[:,['agent_id','associated_asset','delivery_location','cleared_quantity','cleared_price', 'delivery_day','delivery_time','order_type','init_time', 'order_id', 'direction']]
                #cleared price divided by 4 because its a price per MWh and we split the hours to 15 minutes mtu
                DA_cl_orders['due_amount']= - DA_cl_orders['cleared_quantity'] * DA_cl_orders['cleared_price']/4
                #add orders to reporter

                self.obook.cleared_buyorders = self.obook.cleared_buyorders.append(DA_cl_orders)

                DA_cl_orders = DA_cl_orders.drop(['cleared_quantity','cleared_price', 'due_amount'], axis=1).copy()
                DA_cl_orders['quantity']= matches['DA_residual_load']
                DA_cl_orders['price']= np.nan
                DA_cl_orders = DA_cl_orders.loc[:,['agent_id','associated_asset',
                                                           'delivery_location','quantity',
                                                           'price', 'delivery_day','delivery_time',
                                                           'order_type','init_time', 'order_id', 'direction', 'delivery_duration']]


                #add orders to reporter
                self.obook.buyorders_full_step = self.obook.buyorders_full_step.append(DA_cl_orders)
                self.obook.buyorders_full_step[['delivery_day', 'delivery_time','quantity']] = self.obook.buyorders_full_step[['delivery_day', 'delivery_time','quantity']].astype('int64')
            else:
                #no DA auction in this round
                pass
        else:
            raise Exception ('DA clearing type for settlement not known')


class MO_balancing_energy(MarketOperator):
    def __init__(self, model, orderbook, market_rules):
        MarketOperator.__init__(self, model, orderbook, market_rules)
        if self.rules['acquisition_method']=='control_states_only':
            """balancing energy market is simulted based on probabilies of the FRR control state (Dutch style)"""
            self.control_state= None
        else:
            raise Exception('BEM clearing type not implemented: ', self.rules['acquisition_method'])
        if self.rules.loc['gate_opening_time']=='D-1, MTU 56':
            #gate opening time is an absolute MTU of a day.
            self.gate_opening_time = 56
        else:
            raise Exception('BEM gate opening time value not known')
        if self.rules.loc['gate_closure_time']=='deliveryMTU-2':
            #attention: this gate- closure-time is an RELATIVE MTU of until the first delivery MTU (in contrastt to single auctions)
            #+1 because the gate closes at the beginnig of this relative MTU.
            self.gate_closure_time = 2 +1
        else:
            raise Exception('BEM gate opening time value not known')

    def balancing_energy_clearing(self):
        """currently only determination of control state is implemented"""
        print('balancing energy clearing of this MTU')

        if self.rules['acquisition_method']=='control_states_only':
            #orders need to updated anyways, even when not cleared.
            self.obook.update_orders()
        else:
            pass

    def determine_control_state(self):
        """determine the FRR control state based on exogenious probablities per mtu of the day"""
        #balancing of this step (step counter has already proceeded, so calculate mtu back)
        day, MTU=self.model.clock.calc_timestamp_by_steps(self.model.schedule.steps -1, 0)
        #seed to make randomness controlable over various simulations
        if self.model.IB_marketoperator.rules['pricing_method']!='exogenious':
            #Uses the last step rank number of a fixed agent
            seed = self.model.MP_dict[list(self.model.MP_dict.keys())[0]].step_rank + MTU + day

            probabilities =self.model.exodata.control_state_probabilities.loc[
                    self.model.exodata.control_state_probabilities['MTU']== MTU,['control_state','probability']]
            #get random sample of control state, given the probabilities per state of MTU of day
            self.control_state = np.random.RandomState(seed).choice(probabilities.iloc[:,0], p=probabilities.iloc[:,1])
        else:
            #control state obtained from exogenious data
            self.control_state = self.model.exodata.IBP_exo_prices.loc[(self.model.exodata.IBP_exo_prices['delivery_day']==day)&(
                                                                   self.model.exodata.IBP_exo_prices['delivery_time']==MTU),
                                                                   'control_state'].iloc[0]
        self.model.rpt.publish_BEM_control_state(self.control_state, day, MTU)


class MO_imbalance(MarketOperator):
    def __init__(self, model, orderbook, market_rules):
        MarketOperator.__init__(self, model, orderbook, market_rules)
        #Imbalance price of current MTU
        self.IBP_long = None
        self.IBP_short = None
        #sum of all market imbalances of current MTU
        self.imbalances_long = None
        self.imbalances_short = None
        #current MTU tuple
        self.cur_imbalance_settlement_period = None

        if self.rules.loc['gate_opening_time']=='deliveryMTU':
            self.gate_opening_time = 0
        else:
            raise Exception('BEM gate opening time value not known')
        if self.rules.loc['gate_closure_time']=='deliveryMTU':
            self.gate_closure_time = 0
        else:
            raise Exception('BEM gate opening time value not known')


        if isinstance(self.rules['pricing_method'], int):
            #a integer value as imbalance pricing method means that a fixed penalty is payed for any imbalance
            self.IBP_fixed = self.rules['pricing_method']
        elif (self.rules['pricing_method']=='Dutch_IB_pricing'):
            if self.model.exodata.IBP_kde_pdfs.empty:
                raise Exception ('IBP_pdf_sample method for the imbalance market requires IBP_kde_pdf in exodata')
            else:
                DAP_bin_left = self.model.exodata.IBP_kde_pdfs.index.get_level_values('[DAP_left_bin').unique().tolist()
                DAP_bin_right = self.model.exodata.IBP_kde_pdfs.index.get_level_values('DAP_right_bin)').unique().tolist()
                self.DAP_bins =DataFrame({'DAP_bin_left': DAP_bin_left, 'DAP_bin_right': DAP_bin_right})
        elif self.rules['pricing_method']=='exogenious':
            #imbalance prices are taken from exodatabase
            pass
        else:
            raise Exception('the imbalance pricing method not known')

    def imbalance_clearing(self):
        print('imbalance clearing')

        if isinstance(self.rules['pricing_method'], int):
            print('         ...with fixed price',self.rules['pricing_method'])
            #fixed price to be payed for every imbalance (independent from direction)
            self.IBP_short = self.rules['pricing_method']
            self.IBP_long = self.rules['pricing_method']
            day, MTU=self.model.clock.calc_timestamp_by_steps(self.model.schedule.steps -1, 0)
        elif self.rules['pricing_method']=='Dutch_IB_pricing':
            print('imbalance clearing of this MTU (dutch style)')
            if self.model.exodata.sim_task['run_BEM[y/n]']!='y':
                raise Exception ('for the imbalance pricing method Dutch_IB_pricing a BE market operator is needed but Simulation task states not to run BEM')
            #clearing of this step mtu (step counter has already proceeded, so calculate mtu back)
            day, MTU=self.model.clock.calc_timestamp_by_steps(self.model.schedule.steps -1, 0)
            try:
                DAP = self.model.rpt.prices.loc[(day,MTU), 'DAP']
            except:
                if self.model.rpt.prices['DAP'].isnull().all():
                    print('no day-ahead prices available. Possibly because DAM is not run.')
                    print('Default DA price of 30 EUR/MWh used for (Dutch) imbalance clearing')
                    DAP= 30
                else:
                    DAP = self.model.rpt.prices.loc[(day,MTU), 'DAP']
            #get corresponding DAP bin egdes
            DAP_bin =self.DAP_bins.loc[(self.DAP_bins['DAP_bin_left']<=DAP)&(
                    self.DAP_bins['DAP_bin_right']>DAP)]

            #Dutch imbalance pricing method (status 28-03-2019)
            if self.model.BE_marketoperator.control_state == 1:
                #get scipy.stats kde object from dataframe
                pdf = self.model.exodata.IBP_kde_pdfs['pdf'].loc[('IB_price_short',
                                                            DAP_bin.iloc[0,0],
                                                            DAP_bin.iloc[0,1])]
                #get sample from kde pdf. returns nested array. therefore [0][0]
                BEP_up = round(pdf.resample()[0][0])
                self.IBP_short = + BEP_up
                self.IBP_long = BEP_up
            elif self.model.BE_marketoperator.control_state == -1:
                #get scipy.stats kde object from dataframe
                pdf = self.model.exodata.IBP_kde_pdfs['pdf'].loc[('IB_price_long',
                                                            DAP_bin.iloc[0,0],
                                                            DAP_bin.iloc[0,1])]
                #get sample from kde pdf. returns nested array. therefore [0][0]
                BEP_down = round(pdf.resample()[0][0])
                self.IBP_short = BEP_down
                self.IBP_long = BEP_down
            elif self.model.BE_marketoperator.control_state == 0:
                #simplification: DAP as mid price
                self.IBP_short =  DAP
                self.IBP_long =  DAP
            elif self.model.BE_marketoperator.control_state ==2:
                #Attention: prices in FRR control state 2 are drawn independendly from each other.
                #This is a simplification.
                #get scipy.stats kde object from dataframe
                pdf_up = self.model.exodata.IBP_kde_pdfs['pdf'].loc[('IB_price_short',
                                                            DAP_bin.iloc[0,0],
                                                            DAP_bin.iloc[0,1])]
                pdf_down = self.model.exodata.IBP_kde_pdfs['pdf'].loc[('IB_price_long',
                                                            DAP_bin.iloc[0,0],
                                                            DAP_bin.iloc[0,1])]
                #get sample from kde pdf. returns nested array. therefore [0][0]
                BEP_up = round(pdf_up.resample()[0][0])
                BEP_down= round(pdf_down.resample()[0][0])

                #reverse pricing (simplification DAP instead of mid price)
                if (BEP_up < DAP)|(BEP_down > DAP):
                    self.IBP_short = DAP
                    self.IBP_long = DAP
                else:
                    self.IBP_short = BEP_up
                    self.IBP_long = BEP_down
            else:
                raise Exception('imblance clearing requires balancing energy control state of 1,-1, 0 or 2')

        elif self.rules['pricing_method']=='exogenious':
            """Imbalance prices are taken from the exogenious database. This makes sense when comparing various
               simulations, in order to control randomness"""
            #Clearing of this step mtu (step counter has already proceeded, so calculate mtu back)
            day, MTU=self.model.clock.calc_timestamp_by_steps(self.model.schedule.steps -1, 0)

            self.IBP_short = self.model.exodata.IBP_exo_prices.loc[(self.model.exodata.IBP_exo_prices['delivery_day']==day)&(
                                                                   self.model.exodata.IBP_exo_prices['delivery_time']==MTU),
                                                                   'IBP_short'].iloc[0]
            self.IBP_long = self.model.exodata.IBP_exo_prices.loc[(self.model.exodata.IBP_exo_prices['delivery_day']==day)&(
                                                                   self.model.exodata.IBP_exo_prices['delivery_time']==MTU),
                                                                   'IBP_long'].iloc[0]
        self.model.rpt.publish_IBM_prices(self.IBP_short, self.IBP_long, day, MTU)

    def imbalance_settlement(self):
       #step counter has already proceeded, so calculate back, to settle current imbalance
        day, MTU=   self.model.clock.calc_timestamp_by_steps(self.model.schedule.steps -1, 0)
        #sum all imbalances
        self.imbalances_long = 0
        self.imbalances_short = 0
        self.financial_return_ = 0
        self.cur_imbalance_settlement_period = (day, MTU)

        for agent in self.model.schedule.agents:
            #negative means short position, positive means long position.
            imbalance_quantity = agent.trade_schedule.loc[(day,MTU),'imbalance_position']
            if imbalance_quantity < 0.0:
                #always if IBP is positive market party pays system operator
                #short position
                IBP = self.IBP_short
                self.imbalances_short += imbalance_quantity
            elif imbalance_quantity > 0.0:
                #always if IBP is positive, market party receives from system operator
                IBP = self.IBP_long
                self.imbalances_long += imbalance_quantity
            else:
                IBP = 0
            agent.financial_return.loc[(day,MTU),'IB_return'] = imbalance_quantity * IBP / 4
            agent.money += imbalance_quantity * IBP / 4
        #payment inversed for System Operator
        self.model.aGridAndSystemOperator.imb_return = - (
                self.imbalances_long*self.IBP_long + self.imbalances_short *self.IBP_short)/ 4


    def imbalance_clearing_4MTU(self):
        """please note that this alternative method of distinguishing the MTU of an hour
        will soon have less value, because EU cross-border trading is moving towards
        15-minute MTU. However, up to that moment this method may in some cases be useful.
        To use it, the input (IBP_kde_pdfs) need to have the MTU of the hour in the index."""
        print('imbalance clearing')

        if isinstance(self.rules['pricing_method'], int):
            print('         ...with fixed price',self.rules['pricing_method'])
            #fixed price to be payed for every imbalance (independent from direction)
            self.IBP_short = self.rules['pricing_method']
            self.IBP_long = self.rules['pricing_method']
            day, MTU=self.model.clock.calc_timestamp_by_steps(self.model.schedule.steps -1, 0)
        elif self.rules['pricing_method']=='Dutch_IB_pricing':
            print('imbalance clearing of this MTU (dutch style)')
            if self.model.exodata.sim_task['run_BEM[y/n]']!='y':
                raise Exception ('for the imbalance pricing method Dutch_IB_pricing a BE market operator is needed but Simulation task states not to run BEM')
            #TODO: last step: calculate to end of horizon.
            #Clearing of this step mtu (step counter has already proceeded, so calculate mtu back)
            day, MTU=self.model.clock.calc_timestamp_by_steps(self.model.schedule.steps -1, 0)
            try:
                DAP = self.model.rpt.prices.loc[(day,MTU), 'DAP']
            except:
                if self.model.rpt.prices['DAP'].isnull().all():
                    print('no day-ahead prices available. Possibly because DAM is not run.')
                    print('Default DA price of 30 EUR/MWh used for (Dutch) imbalance clearing')
                    DAP= 30
                else:
                    #raise error of this:
                    DAP = self.model.rpt.prices.loc[(day,MTU), 'DAP']
            #get corresponding DAP bin egdes
            DAP_bin =self.DAP_bins.loc[(self.DAP_bins['DAP_bin_left']<=DAP)&(
                    self.DAP_bins['DAP_bin_right']>DAP)]
            #determine which MTU of an hour the current MTU is (1,2,3 or 4)
            MTU_of_h = MTU%4
            if MTU_of_h == 0:
                MTU_of_h = 4
            #Dutch imbalance pricing method (28-03-2019)
            if self.model.BE_marketoperator.control_state == 1:
                #get scipy.stats kde object from dataframe
                pdf = self.model.exodata.IBP_kde_pdfs['pdf'].loc[('IB_price_short',
                                                            MTU_of_h,DAP_bin.iloc[0,0],
                                                            DAP_bin.iloc[0,1])]
                #get sample from kde pdf. returns nested array. therefore [0][0]
                BEP_up = round(pdf.resample()[0][0])
                self.IBP_short = + BEP_up
                self.IBP_long = BEP_up
            elif self.model.BE_marketoperator.control_state == -1:
                #get scipy.stats kde object from dataframe
                pdf = self.model.exodata.IBP_kde_pdfs['pdf'].loc[('IB_price_long',
                                                            MTU_of_h,DAP_bin.iloc[0,0],
                                                            DAP_bin.iloc[0,1])]
                #get sample from kde pdf. returns nested array. therefore [0][0]
                BEP_down = round(pdf.resample()[0][0])
                self.IBP_short = BEP_down
                self.IBP_long = BEP_down
            elif self.model.BE_marketoperator.control_state == 0:
                #simplification: DAP as mid price
                self.IBP_short =  DAP
                self.IBP_long =  DAP
            elif self.model.BE_marketoperator.control_state ==2:

                #Attention: prices in FRR control state 2 are drawn independendly from each other.
                #This is a simplification.
                #get scipy.stats kde object from dataframe
                pdf_up = self.model.exodata.IBP_kde_pdfs['pdf'].loc[('IB_price_short',
                                                            MTU_of_h,DAP_bin.iloc[0,0],
                                                            DAP_bin.iloc[0,1])]
                pdf_down = self.model.exodata.IBP_kde_pdfs['pdf'].loc[('IB_price_long',
                                                            MTU_of_h,DAP_bin.iloc[0,0],
                                                            DAP_bin.iloc[0,1])]
                #get sample from kde pdf. returns nested array. therefore [0][0]
                BEP_up = round(pdf_up.resample()[0][0])
                BEP_down= round(pdf_down.resample()[0][0])

                #reverse pricing (simplification DAP instead of mid price)
                if (BEP_up < DAP)|(BEP_down > DAP):
                    self.IBP_short = DAP
                    self.IBP_long = DAP
                else:
                    self.IBP_short = BEP_up
                    self.IBP_long = BEP_down
            else:
                raise Exception('imblance clearing requires balancing energy control state of 1,-1, 0 or 2')

        elif self.rules['pricing_method']=='exogenious':
            """Imbalance prices are taken from the exogenious database. This makes sense when comparing various
               simulations, in order to control randomness"""
            #Clearing of this step mtu (step counter has already proceeded, so calculate mtu back)
            day, MTU=self.model.clock.calc_timestamp_by_steps(self.model.schedule.steps -1, 0)

            self.IBP_short = self.model.exodata.IBP_exo_prices.loc[(self.model.exodata.IBP_exo_prices['delivery_day']==day)&(
                                                                   self.model.exodata.IBP_exo_prices['delivery_time']==MTU),
                                                                   'IBP_short'].iloc[0]
            self.IBP_long = self.model.exodata.IBP_exo_prices.loc[(self.model.exodata.IBP_exo_prices['delivery_day']==day)&(
                                                                   self.model.exodata.IBP_exo_prices['delivery_time']==MTU),
                                                                   'IBP_long'].iloc[0]
        self.model.rpt.publish_IBM_prices(self.IBP_short, self.IBP_long, day, MTU)





