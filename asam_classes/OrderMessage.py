# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 17:29:25 2017
@author: Samuel Glismann


A order message is a set of of orders which can be submitted to an order book of a market operator.

When initiating a order message the received orders are checked regarding consistency.
A DataFrame is returned.


Note:
    - init_time attribute is not the actual time of the model run, but the rank number in
      which the agents took steps. these rank is random (random scheduler). However important
      to prioritise orders with same price
    - Orders messages for the intra-day market are checked on internal matches. These are excluded.

order type: limit order means that the quantity can be partially cleared, while the rest stays in the order book.
order type: All-or-none means that the order can only be matched entirely (or not)
order type: market order means that it is fill-and-kill order where as much as possible is cleared, but no rest stays in the order book.
            Market orders have currently a (shadow) price of buy 1000 or sell -1000 EUR/MWh. this has an effect on market statistics.
order type: Intra-day Congestion Spread (IDCONS) order is a intra-day order which can also be used by a grid operator for redispatch.
            This order type is for example applied in the Netherlands.
"""
import pandas as pd
from pandas import Series, DataFrame
import numpy as np


class OrderMessage():

    def __init__(self, orders):
        #label lists needed to convert single orders in dataframe collections
        self.orderlabels = ['agent_id','associated_asset','delivery_location',
                             'quantity','price', 'delivery_day','delivery_time',
                             'order_type','init_time', 'order_id', 'direction','delivery_duration']

        self.order_df = self.consistency_check(orders)
        self.order_df, self.portfolio_exclusions = self.exclude_internal_IDtrades(self.order_df)

    def consistency_check(self, orders):
        if type(orders) is list:
            order_df = DataFrame(orders, columns = self.orderlabels)
        else:
            order_df = orders
        if (order_df['order_type']=='redispatch_demand').any():
            #Grid Operator orders do not contain price integers but nan which are considered as float64 by dtypes
            consistent_reference = DataFrame([['o','o','o',1,1.1,
                                     1,1,'o',1.1,
                                     'o','o',1]],columns=self.orderlabels).dtypes
        else:
            consistent_reference = DataFrame([['o','o','o',1,1,
                                     1,1,'o',1.1,
                                     'o','o',1]],columns=self.orderlabels).dtypes
        if (order_df['quantity']<=0).any():
            import pdb
            pdb.set_trace()
            raise Exception('order message quantity <=0...')

        if consistent_reference.equals(order_df.dtypes):
            return (order_df)
        else:
            import pdb
            pdb.set_trace()
            print(order_df)
            print(order_df.dtypes)
            print (consistent_reference)
            raise Exception('Order Message contains invalid dtypes at some columns')

    def exclude_internal_IDtrades(self, orders):
        """this is actually an agent portfolio 'matching'"""
        orders.set_index(['delivery_day','delivery_time'], inplace=True)
        orders.sort_index(inplace=True)
        if ((orders['order_type']=='intraday_limit_order').any())|(
                (orders['order_type']=='market_order').any())|(
                 (orders['order_type']=='IDCONS_order').any()):
            excl_lst=[]
            for deliveryday, orders_t in orders.groupby(level=[0,1]):
                sells = orders_t.loc[orders_t['direction']=='sell']
                buys = orders_t.loc[orders_t['direction']=='buy']
                low_sells=sells.loc[sells['price']<=buys['price'].max()].copy()
                high_buys=buys.loc[buys['price']>=sells['price'].min()].copy()
                if low_sells['quantity'].sum()>high_buys['quantity'].sum():
                    excl_lst =excl_lst + list(high_buys['order_id'])
                    low_sells.sort_values(['price'], ascending=True,inplace=True)
                    o_numb = len(low_sells.loc[low_sells['quantity'].cumsum()<high_buys['quantity'].sum()])
                    excl_lst =excl_lst+ list(low_sells['order_id'].iloc[:o_numb])
                    #quantity correction of the last order that overlaps with the buys
                    o_id = low_sells['order_id'].iloc[o_numb]
                    orders['quantity'].loc[orders['order_id']==o_id] = low_sells['quantity'].iloc[:o_numb+1].sum()-high_buys['quantity'].sum()

                elif low_sells['quantity'].sum()<high_buys['quantity'].sum():
                    #exclude all low_sells
                    excl_lst =excl_lst + list(low_sells['order_id'])
                    #exclude as much highbuys as quantity of lowsells, in decending price order
                    high_buys.sort_values(['price'], ascending=False,inplace=True)
                    o_numb = len(high_buys.loc[high_buys['quantity'].cumsum()<low_sells['quantity'].sum()])
                    excl_lst =excl_lst+ list(high_buys['order_id'].iloc[:o_numb])
                    #quantity correction of the last order that overlaps with the sells
                    o_id = high_buys['order_id'].iloc[o_numb]
                    orders['quantity'].loc[orders['order_id']==o_id] = high_buys['quantity'].iloc[:o_numb+1].sum()-low_sells['quantity'].sum()
                else:
                    excl_lst =excl_lst + list(high_buys['order_id']) + list(low_sells['order_id'])

                if excl_lst:
                    #filter the excluded orders and return
                    orders_df=orders.loc[~(orders['order_id'].isin(excl_lst))].reset_index()
                    excl_orders=orders.loc[orders['order_id'].isin(excl_lst)].reset_index()
                    print('portfolio internal matched ID orders')
                    print(excl_orders)
                    return (orders_df, excl_orders)
                else:
                    return(orders.reset_index(), DataFrame())
        else:
            orders.reset_index(inplace=True)
            excl_orders=DataFrame()
            return (orders, excl_orders)

    def get_as_df(self):
        return(self.order_df)




