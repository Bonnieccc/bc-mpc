import numpy as np;
import seaborn as sns; sns.set(color_codes=True)
import matplotlib.pyplot as plt

gammas = sns.load_dataset("gammas")
print(gammas)
ax = sns.tsplot(time="timepoint", value="BOLD signal",
                 unit="subject", condition="ROI",
                 data=gammas)
plt.show()