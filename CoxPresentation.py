from lifelines.datasets import load_stanford_heart_transplants

# Load the dataset
data = load_stanford_heart_transplants()

# Keep only the required columns
data = data[['start', 'stop', 'event','transplant', 'id']]

# Time independent analysis ##########################################
# Format the data
# 2 groups : 'has ever received transplant', 'never received transplant'
data_time_independent = data.groupby('id').agg({
    'event': 'max',
    'transplant': 'max',
    'stop' : 'max'
}).reset_index()
data_time_independent = data_time_independent.drop('id', axis=1)

from lifelines import CoxPHFitter
# initialize the Cox Proportional Hazards model
cph = CoxPHFitter()
# fit the model to the data
cph.fit(data_time_independent, duration_col='stop', event_col='event')
# print the summary of the model
print(cph.print_summary())

# time dependent analysis ##########################################
from lifelines import CoxTimeVaryingFitter
ctv = CoxTimeVaryingFitter()
ctv.fit(data, id_col='id', event_col='event', start_col='start', stop_col='stop', show_progress=True)
ctv.print_summary()