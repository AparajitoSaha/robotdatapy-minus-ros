import numpy as np
from pathlib import Path
from copy import deepcopy
import os

from robotdatapy.data.robot_data import RobotData
    
class GeneralData(RobotData):
    """
    Class for easy access to generic robot data over time
    """
    
    def __init__(
        self, 
        data=None, 
        times=None, 
        time_tol=.1, 
        causal=False, 
        t0=None,
    ): 
        """
        Class for easy access to object poses over time

        Args:
            data (list): list of data items corresponding to times
            times (list): list of times corresponding to data items
            time_tol (float, optional): Tolerance used when finding a pose at a specific time. If 
                no pose is available within tolerance, None is returned. Defaults to .1.
            causal (bool): if True, only use data that is available at the time requested.
            t0 (float, optional): Local time at the first msg. If not set, uses global time from 
                the data_file. Defaults to None.
        """
        super().__init__(time_tol=time_tol, interp=False, causal=causal)
        self._data = deepcopy(data)
        self.set_times(np.array(times))
        if t0 is not None:
            self.set_t0(t0) 
            
    def data(self, t):
        """
        Data at time t.

        Args:
            t (float): time

        Returns:
            any: data item at time t
        """
        return self.get_val(self._data, t)