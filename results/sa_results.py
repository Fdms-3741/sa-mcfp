import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import psycopg
import seaborn as sns
import scipy as sp


def meanAndCI(x, ci=0.95):
    n = len(x)
    mean = x.mean()
    std = x.std()
    ciVal = sp.stats.t.ppf(ci, n) * std / np.sqrt(n)
    return f"${mean:3g}\\pm{ciVal:3g}$"


def meanAndCIText(x):
    mean = x["mean"]
    ciVal = x["ci"]
    return f"${mean:3g}\\pm{ciVal:3g}$"


def meanAndCINumeric(x, ci=0.95):
    n = len(x)
    mean = x.mean()
    std = x.std()
    ciVal = sp.stats.t.ppf(ci, n) * std / np.sqrt(n)
    return pd.DataFrame({"mean": [mean], "ci": [ciVal]})


def GetGraphNames():
    with dbconn.cursor() as cursor:
        cursor.execute("SELECT DISTINCT graph_type FROM graphs")
        ret = cursor.fetchall()
    return ret


def GetGraph(result_id, name):
    with dbconn.cursor() as cursor:
        cursor.execute(
            f"SELECT x,y FROM graph_data WHERE graph_id=(SELECT id FROM graphs WHERE result_id={result_id} AND graph_type='{name}')"
        )
        data = cursor.fetchall()
    return pd.Series([i[1] for i in data], index=[i[0] for i in data])


#
# Data preparation
#

# Excluding results that had less than 100 executions
# and a single one that passed 1000 (forgot to kill process)
sqlQuery = """
SELECT 
    * 
FROM results 
WHERE 
    (k,n,t0,epsilon) IN (
        SELECT 
            (params->>'f1')::int as k, 
            (params->>'f2')::int as n, 
            (params->>'f3')::int as t0, 
            (params->>'f4')::double precision as epsilon 
        FROM (
            SELECT 
                row_to_json(params) as params 
            FROM (
                SELECT 
                    DISTINCT(k, n, t0, epsilon) as params, 
                    COUNT(*) as count FROM results 
                GROUP BY params 
                ORDER BY 
                    count DESC
            )    
            WHERE 
                count > 100 
                AND count < 1000
        )
    ) 
    OR method='LP'
;
"""

df = pd.read_sql(sqlQuery, dbconn)

df = df.drop(df[df["name"] == "1-flow-random-60-100-1000-contracts.json"].index)
df = df.drop(df[df["name"] == "letter-graph-lp-solved.json"].index)

# print(df["name"].unique())

# Name formatting
df["rnp"] = df["name"].str.contains("rnp")
df["random"] = df["name"].str.contains("random")
df["geant"] = df["name"].str.contains("geant")
df[500] = df["name"].str.contains("500-contracts")
df[100] = df["name"].str.contains("100-contracts")

df["graph_name"] = pd.from_dummies(df[["rnp", "random", "geant"]])
df["num_contracts"] = pd.from_dummies(df[[100, 500]])

del df["rnp"]
del df["random"]
del df["geant"]
del df[100]
del df[500]

# print(df["graph_name"].unique())
# print(df["num_contracts"].unique())

del df["objective"]

df["iteration_speed"] = (df["k"] * df["n"]) / df["execution_time"]

# Indexing
df = df.set_index(
    [
        "graph_name",
        "num_contracts",
        "method",
        "sa_technique",
        "k",
        "n",
        "t0",
        "epsilon",
        "number_of_paths",
    ]
)

# Registers a graph column for
for graphName in ["geant", "rnp", "random"]:
    for contracts in [100, 500]:
        df.loc[(graphName, contracts), ("optimal_cost",)] = int(
            df.loc[(graphName, contracts, "LP"), ("found_cost",)].iloc[0].iloc[0]
        )

# Calculates gap
df["gap"] = (df["found_cost"] - df["optimal_cost"]) / df["optimal_cost"]

# Remove invalid results (negative gap)
df = df.drop(df[df["gap"] < 0].index)

# Remove results from LP
df = df.drop(df.loc[:, :, ["LP"]].index)

df.index = df.index.droplevel(2)

#
# Results
#

# Smallest gap's parameters
data = {}
idx = []
data["gap"] = []
meanGaps = df.groupby(
    [
        "sa_technique",
        "num_contracts",
        "graph_name",
        "k",
        "n",
        "t0",
        "epsilon",
        "number_of_paths",
    ],
    dropna=False,
)["gap"].apply(meanAndCINumeric)
meanGaps.index = meanGaps.index.droplevel(-1)

meanGaps = meanGaps.groupby(["sa_technique", "num_contracts", "graph_name"]).apply(
    lambda x: x.sort_values(by="mean", ascending=True).head(1)
)

meanGaps["gap"] = meanGaps[["mean", "ci"]].apply(meanAndCIText, axis=1)

del meanGaps["mean"]
del meanGaps["ci"]

meanGaps.index = meanGaps.index.droplevel([3, 4, 5])

print()
print("Smallest gap's parameters")
print(meanGaps)
meanGaps.to_latex("output/1 - sa_smallest_gap_parameters.tex")

exit()


# Ratio overall

res = df.groupby(["sa_technique", "num_contracts", "graph_name"])[
    "viable_solution_found"
].apply(lambda x: {"ratio": x.sum() / x.count(), "n": x.count()})
# print("Total average")
res = pd.DataFrame({"ratio": res.loc[:, :, :, "ratio"], "n": res.loc[:, :, :, "n"]})
# print(res)
res.to_latex("output/1 - sa_viable_solution_overall_ratio.tex")
print(df)


# Iterations per second

itSecs = df.groupby(
    ["sa_technique", "number_of_paths", "num_contracts", "graph_name"], dropna=False
).apply(
    lambda x: pd.Series(
        (np.array(x.reset_index()["k"]) * np.array(x.reset_index()["n"]))
        / np.array(x["execution_time"]),
        name="iterations_sec",
    )
)
itSecs = itSecs.reset_index()

itSecs["contract-graph"] = (
    itSecs["num_contracts"].astype("str") + " - " + itSecs["graph_name"]
)
itSecs["technique_paths"] = (
    itSecs["sa_technique"] + " - " + itSecs["number_of_paths"].astype("str")
)

itSecs = itSecs.rename(
    columns={
        "contract-graph": "Rede e nº de contratos",
        "technique_paths": "Método",
        "iterations_sec": "Iterações/segundo",
    }
)

itSecs = itSecs.replace(
    {
        "100 - rnp": "RNP (100)",
        "100 - random": "Aleatório (100)",
        "100 - geant": "GÉANT (100)",
        "500 - geant": "GÉANT (500)",
        "Nullspace - nan": "Proposta",
        "Kpaths - 5.0": "5-caminhos",
        "Kpaths - 10.0": "10-caminhos",
        "Kpaths - 15.0": "15-caminhos",
        "Kpaths - 20.0": "20-caminhos",
        "Kpaths - 25.0": "25-caminhos",
        "Kpaths - 30.0": "30-caminhos",
        "Kpaths - 35.0": "35-caminhos",
        "Kpaths - 45.0": "45-caminhos",
        "Kpaths - 40.0": "40-caminhos",
    }
)
sns.catplot(
    data=itSecs,
    kind="point",
    y="Iterações/segundo",
    x="Rede e nº de contratos",
    hue="Método",
    order=["RNP (100)", "GÉANT (100)", "Aleatório (100)", "GÉANT (500)"],
)
plt.yscale("log")
plt.grid(True)
# plt.show()


# Average improvement and viable solution


meanViable = (
    df[df["viable_solution_found"] & df["time_to_viable_solution"] > 0]
    .groupby(["sa_technique", "num_contracts", "graph_name"])["time_to_viable_solution"]
    .apply(meanAndCI)
)
meanImprovement = (
    df[
        df["viable_solution_found"]
        & (df["time_to_viable_solution"] > 0)
        & (~np.isinf(df["average_answer_improvement_time"]))
    ]
    .groupby(["sa_technique", "num_contracts", "graph_name"])[
        "average_answer_improvement_time"
    ]
    .apply(meanAndCI)
)

res = pd.DataFrame(
    {
        "mean viable": meanViable,
        "mean improv": meanImprovement,
    }
)
print("average improvement times")
print(res)

res.to_latex("output/1 - times_improvement.tex")
