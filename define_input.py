import pandas as pd
import numpy as np

df = pd.read_csv("aeroporto.csv", sep=",", na_values="M")
"""
The value "M" represents either value that was reported as missing or
 a value that was set to missing after meeting some general quality 
 control check, or a value that was never reported by the sensor.
"""
df = df.ffill()

df["mps"] = df["sknt"] / 1.944
df["u"] = df["mps"] * np.sin(df["drct"] * np.pi / 180)
df["v"] = df["mps"] * np.cos(df["drct"] * np.pi / 180)
df = df.drop(columns=["station", "mps", "sknt", "drct"], axis=1)
df.to_csv("aeroporto_inp.csv", index=None, sep=";")
