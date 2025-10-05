tprofiler.time.manage
=================================================

.. currentmodule:: tprofiler.time.manage

.. automodule:: tprofiler.time.manage


TimeManager
----------------------------------------------------------

.. autoclass:: TimeManager
    :members: records,_append_time,_get_time,_get_time_torch,_get_time_with_rank,timer,clear,enable_timer,gather




GatheredTime
----------------------------------------------------------

.. autoclass:: GatheredTime
    :members: times,ranks,get_rank,__getitem__,__bool__,sum,count,mean,std,rank_count,hist


