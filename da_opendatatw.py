
import pandas as pd
import numpy as np

from pathlib import Path

CWD = Path('__file__').parent

################################################################################
########################### INITIALISATION #####################################
################################################################################

# ---------------------- Import District population ---------------------------#
csv_DistPop = list(CWD.rglob('opendata10802M030.csv'))[0]
dF_DistPop = pd.read_csv(csv_DistPop, skiprows=[1]) # Skip 2nd row with CH label
# Type correctly the categories
dF_DistPop["district_code"] = dF_DistPop["district_code"].astype('category')
dF_DistPop["site_id"] = dF_DistPop["site_id"].astype('category')
dF_DistPop["village"] = dF_DistPop["village"].astype('category')


