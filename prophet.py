import pandas as pd
from fbprophet import Prophet

gold_data = pd.read_csv('gold.csv')
model = Prophet()

gold_data.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
model.fit(gold_data)

future = model.make_future_dataframe(periods=365)

forecast = model.predict(future)

fig1 = model.plot(forecast)

forecast.to_csv("output.csv")