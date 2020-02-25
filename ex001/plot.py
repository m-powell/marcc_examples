import glob
import matplotlib.pyplot as plt
from matplotlib import pyplot as pyplt
import pandas as pd

li = []
all_files = glob.glob("output/*csv")

## Grab all the output files and read in a pd.DataFrames
for filename in all_files:
    df = pd.read_csv(filename, index_col=None)
    li.append(df)


fullDat = pd.concat(li)

fullDat = fullDat.sort_values(by=['openmlDataID'])
fullDat['openmlDataID'] = fullDat['openmlDataID'].astype('category')

fullDat.plot(x = "openmlDataID", y = 'error_rate', style = 'o')

fig, ax = plt.subplots(1, 1, figsize=(14,8))
fullDat.boxplot(ax=ax,column = ['error_rate'], by ='openmlDataID')
plt.title('')
plt.suptitle('sklearn randomForest run on openML datasets')
plt.savefig('error_rate_boxplot.png', dpi = 300)
