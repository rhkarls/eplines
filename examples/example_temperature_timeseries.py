from matplotlib import pyplot as plt
from sondera.clients.smhi import MetObsClient, ParametersMetObs
import pandas as pd
from eplines import ECDFLines

# Read some long daily air temperature series from SMHI station "Uppsala" with id 97520
mc = MetObsClient()
temp = mc.get_observations(ParametersMetObs.TemperatureAirDay, 97520, 'corrected-archive')

data = pd.DataFrame(temp.data)
data = data.truncate(after='1984-12-31')

# Create a pivot frame with each year being indexed on day of year
data['DOY'] = data.index.dayofyear
data['YEAR'] = data.index.year

doy_frame = data.pivot(index='DOY', columns='YEAR', values='TemperatureAirDay')
doy_frame = doy_frame.dropna()

ecdf_test = ECDFLines()
ecdf_test.ecdf(y_lines=doy_frame)

fig, ax = plt.subplots()
_, ecdf_im = ecdf_test.plot(ax=ax, cmap='RdBu_r', mask_to_data=True)
ax.set_ylabel('Daily mean air temperature $°C$')
ax.set_xlabel('Day of Year')

cb = fig.colorbar(ecdf_im, ax=ax)
cb.ax.set_ylabel('Daily empirical cumulative frequency')
ax.set_title('Daily ECDF for air temperature records from Uppsala, Sweden (1840-1984)',
             fontsize=10)
fig.tight_layout()
fig.show()
