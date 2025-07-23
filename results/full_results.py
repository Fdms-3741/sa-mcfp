import numpy as np
import pandas as pd

import psycopg

query = """
SELECT 
    name,
    objective,
    method,
    sa_technique,
    MIN(found_cost)
FROM results 
WHERE viable_solution_found 
GROUP BY 
    name,
    objective,
    method,
    sa_technique;
"""

a = pd.read_sql(query, dbconn)

a = a.pivot(index=["name"], columns=["objective", "method", "sa_technique"])

a = a.T.sort_index().T

a = a.loc[:, ("min", ["MinCost", "QueueDelay"])]["min"]

a = a.drop(["letter-graph-lp-solved.json", "1-flow-random-60-100-1000-contracts.json"])

wfValueMinCost = np.nanmin(a["MinCost", "WF"], axis=1)
wfValueQueueDelay = np.nanmin(a["QueueDelay", "WF"], axis=1)

del a["MinCost", "WF"]
del a["QueueDelay", "WF"]

a["MinCost", "WF", np.nan] = wfValueMinCost
a["QueueDelay", "WF", np.nan] = wfValueQueueDelay

a = a.T.sort_index().T
print(a)
