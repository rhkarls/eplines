"""
Example using generated data and plotting numpy and pandas data.
"""

import string
import numpy as np
import pandas as pd
from eplines import ECDFLines

# generate test data, 1000 series each of length 100
from matplotlib import pyplot as plt

x_n = 100
y_n = 1000

y_res = 100
x = np.linspace(0, x_n, x_n)
ys = np.empty((y_n, x_n))
randg = np.random.default_rng(seed=645)
for i in range(y_n):
    ys[i] = np.sin(-x / 3) + 2 * randg.standard_normal(1) + x * 0.1 + 11

# Data frames with numeric index, datetime index and string index
ys_df = pd.DataFrame(index=x, data=ys.T)

ys_df_di = ys_df.copy()
ys_df_di.index = pd.date_range("2020-01-01", "2020-04-09", freq="D")

ys_df_stri = ys_df.copy()
a_str = string.ascii_lowercase
a_list = list(a_str*5)[:x_n]
ys_df_stri.index = a_list

ecdf = ECDFLines(y_res=y_res, mode='exceedance')

# DataFrame with datetime index
ecdf.ecdf(y_lines=ys_df_di)
fig, ax = plt.subplots()
ecdf.plot(ax=ax, cmap='summer')
fig.show()

# numpy arrays
ecdf.ecdf(y_lines=ys, x=x)
fig2, ax2 = plt.subplots()
ecdf.plot(ax=ax2)
fig2.show()

# DataFrame with string index
ecdf.ecdf(y_lines=ys_df_stri)
fig3, ax3 = plt.subplots()
ecdf.plot(ax=ax3, cmap='inferno')
fig3.show()
