import pandas as pd
from fbprophet import Prophet

gold_data = pd.read_csv('monthly_csv.csv')

model = Prophet()

gold_data.rename(columns={'Date': 'ds', 'Price': 'y'}, inplace=True)

model.fit(gold_data)

future = model.make_future_dataframe(periods=12)

forecast = model.predict(future)

fig1 = model.plot(forecast)

forecast.to_csv("output.csv")