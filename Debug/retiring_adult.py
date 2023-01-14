import pandas as pd
from folktables import ACSDataSource

data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
ca_data: pd.DataFrame = data_source.get_data(states=["CA"], download=True)
ca_data.to_csv('./Data/RetiringAdult/retiring_adult.csv', index=False)

test_data_source = ACSDataSource(survey_year='2017', horizon='1-Year', survey='person')
test_ca_data: pd.DataFrame = data_source.get_data(states=["CA"], download=True)
test_ca_data.to_csv('./Data/RetiringAdult/retiring_adult_test.csv', index=False)