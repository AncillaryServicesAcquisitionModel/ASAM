# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 12:43:04 2017
@author: Samuel Glismann

Time class for ASAM marketmodel class.

The time class provides several time related multi_index templates:
     - schedule template (used as forward horizon for agent actions)
     - report time matrix (used to store intermediate simulation results)
     - report time_location matrix (used to store intermediate simulation results, per grid_area)

Methods:
get_hour()
get_day()
get_MTU()
asset_schedules_horizon ()
calc_timestamp_by_steps ()
calc_delivery_period_end ()
calc_delivery_period_range ()

Note:

    - The MESA time class sets schedule.steps on 0. After the first schedule.step()
      is called, schedule.steps is set to +1.

    - The step size of the model is 15 minutes. At this moment no other options are implemented.

    - Time is usually expressed in tuples of (day, MTU).

    - One day has 96 MTU.

    - The first scheduled step (MESA starts counting with 0) is however MTU 1 (e.g. from 00:00 to 00:15).

    - The model allows to start at a specific startday and startMTU.
"""
import pandas as pd
from pandas import DataFrame

class Time():
    def __init__(self, model, step_size = "15_minutes", startday = 1, startMTU = 1, step_numbers=None, DA_GCT= 'D-1, MTU 45'):

        self.model = model
        if step_numbers == None:
            raise Exception('Initiation of Time() instance needs the step_number of the simulation')
        self.startday = startday
        self.startMTU = startMTU
        if (self.startday <= 0)|(self.startMTU <= 0)|(self.startMTU <= 0):
            raise Exception ('startday and startMTU must be > 0')

        if step_size == '15_minutes':
            #MTU = market time unit = 15 minutes per default
            self.step_size = step_size
        else:
            raise Exception('step_size other than 15 minutes MTU')

        #Note: after day-ahead gate closure time, it takes usually 1 to 2 hours before results are definitive. From this moment
        #onwards, market parties plan for the next day. Here it is assumed that results are known instantly.
        if DA_GCT == 'D-1, MTU 45':
            self.DA_results_known= 45
        else:
            raise Exception ('DA GTC not implemented in Time class')
        #end date of simulation(tuple)
        self.end_date = self.calc_timestamp_by_steps(0,step_numbers-1)
        endday = self.end_date[0]
        day = list(range(self.startday, endday + 1))
        time = list(range(1,96 + 1))
        gridarea = self.model.gridareas #list

        #make scheduling template
        mi2 =pd.MultiIndex.from_product([day, time],
                                             names=['delivery_day', 'delivery_time'])
        self.schedule_template= DataFrame(index =mi2)
        self.schedule_template.sort_index(level=0,inplace = True)

        #delete all dates before start moment
        self.schedule_template = self.schedule_template.iloc[self.startMTU-1:]
        self.schedule_template['commit'] = 0

        #make report matrix from schedule_template by adding step time dimension
        mi3 =self.schedule_template.index
        simu_days=[]
        simu_mtu=[]
        #limit rows to number of steps
        for i in range(step_numbers):
            d,t =self.calc_timestamp_by_steps(0,i)
            if simu_days.count(d) == 0:
                simu_days.extend([d])
            if simu_mtu.count(t) == 0:
                simu_mtu.extend([t])
        mi4=pd.MultiIndex.from_product([simu_days, simu_mtu],
                                             names=['simulation_day', 'simulation_time'])
        self.report_time_matrix = DataFrame(index = mi4, columns = mi3)

        #for redispatch reporting a location report_time matrix is needed.
        gridarea = self.model.gridareas #list
        dday_lst=list(self.report_time_matrix.index.get_level_values(level=0))
        mmtu_lst=list(self.report_time_matrix.index.get_level_values(level=1))
        llocation=[]
        for i in range(len(gridarea)):
            llocation.extend([gridarea[i]]*len(dday_lst))
        #make temporary dataframe to ensure the right index length
        df = DataFrame({'delivery_location':llocation,
                        'simulation_day':dday_lst*len(gridarea),
                        'simulation_time':mmtu_lst*len(gridarea)})
        df.set_index(['delivery_location','simulation_day','simulation_time'], inplace=True)
        self.report_location_time_matrix = DataFrame(index = df.index, columns = mi3)
        self.report_location_time_matrix.sort_index(inplace=True, level=[0,1,2])
        del df


    def get_hour(self):
        if self.step_size == "15_minutes":
          # // floor devision returns the quotient in which the digits after the decimal point are removed
           hour = ((self.model.schedule.steps + self.startMTU -1)//4) % 24 +1
        return(hour)

    def get_day(self):
        if self.step_size ==  "15_minutes":
            # // floor devision eturns the quotient in which the digits after the decimal point are removed
            day = ((self.model.schedule.steps + self.startMTU + self.startday * 24 * 4 -1) // 96)
        return (day)

    def get_MTU(self):
        if self.step_size ==  "15_minutes":
            #mtu between 1 and 96
            MTU = (self.model.schedule.steps + self.startMTU -1) % 96 +1
        return (MTU)

    def asset_schedules_horizon (self):
        cur_MTU = self.get_MTU()

        #on the last simluation day, the horizon is not extended to the next day.
        if (cur_MTU >= self.DA_results_known)&(
                self.schedule_template.index.get_level_values(0)[-1] >= self.get_day()+1):
           endday = self.get_day()+1
        else:
           endday = self.get_day()
        df = self.schedule_template.loc[(slice(endday),slice(None)),:].copy()
        df =df.iloc[self.model.schedule.steps:]
        return (df)

    def calc_timestamp_by_steps (self, paststeps, deltasteps):
        if self.step_size == "15_minutes":
            MTU = (self.startMTU + paststeps + deltasteps -1) % 96 +1
            #day
            day = ((self.startMTU + paststeps + deltasteps + self.startday * 24 * 4 -1) // 96)
        return (day, MTU)

    def calc_delivery_period_end (self, starttuple, deliveryperiod_MTUs):
        """this method is used to calculate the end mtu of a delivery period of block orders.
           With delivery period of 1 MTU the start delivery and end delivery MTU are equivalent."""
        if(starttuple[1]<0)|(starttuple[1]>96)|(starttuple[0]<0):
            raise Exception ('starttuple given to calc_delivery_period_end contains infeasable values for day or time')

        if self.step_size == "15_minutes":
            MTU = (starttuple[1] + deliveryperiod_MTUs -1 -1) % 96 +1
            #day
            day = ((starttuple[1] + deliveryperiod_MTUs + starttuple[0] * 24 * 4 -1 -1) // 96)
        return (day, MTU)

    def calc_delivery_period_range (self, startday, startmtu, deliveryperiod_MTUs):
        """this method is used to calculate the a delivery period range of a block order.
          it returns a list for the days and a list for the mtu.
          ATTENTION: it returns a list excluding the startday and startmtu
          """
        if(startmtu<0)|(startmtu>96)|(startday<0):
            raise Exception ('starttuple given to calc_delivery_period_range contains infeasable values for day or time')

        if self.step_size == "15_minutes":
            mtu_lst =[]
            day_lst=[]
            for i in range(int(deliveryperiod_MTUs)-1):
                MTU = (startmtu + i) % 96 +1
                #day
                day = ((startmtu + i + startday * 24 * 4) // 96)
                mtu_lst +=[int(MTU)]
                day_lst +=[int(day)]
        return (day_lst, mtu_lst)

















