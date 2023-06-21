import math
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

st.write("""
# Jolibee Stock Price Prediction
Predict stock prices for Jollibee Food Corporation using this LSTM-based program.
""")
st.divider()
st.header("Input")

uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])

if uploaded_file is not None:
    jfc_data = pd.read_csv(uploaded_file,
                           index_col='Date',
                           parse_dates = True,
                           infer_datetime_format=True)
    jfc_data = jfc_data.sort_values('Date')

    st.divider()
    st.header('Output')

    with st.spinner('Operation in progress. Just a moment...'):
        jfc_prices = jfc_data['Price']
        values = jfc_prices.values
        training_data_len = math.ceil(len(values)* 0.8)

        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(values.reshape(-1,1))
        train_data = scaled_data[0: training_data_len, :]

        x_train = []
        y_train = []

        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        test_data = scaled_data[training_data_len-60: , : ]

        x_test = []
        y_test = values[training_data_len:]
        for i in range(60, len(test_data)):
          x_test.append(test_data[i-60:i, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        model = keras.Sequential()
        model.add(layers.LSTM(100,
                              return_sequences=True,
                              input_shape=(x_train.shape[1], 1)))
        model.add(layers.LSTM(100,
                              return_sequences=False))
        model.add(layers.Dense(25))
        model.add(layers.Dense(1))
        model.summary()

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, batch_size= 1, epochs=3)

        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)
        rmse = np.sqrt(np.mean(predictions - y_test)**2)

    st.success('Processing Complete!')
    print(rmse)
    st.text('Root Mean Square Error (RMSE): {}'.format(rmse))

    data = jfc_data.filter(['Price'])
    train = data[:training_data_len]
    validation = data[training_data_len:]
    validation['Predictions'] = predictions

    st.subheader("Actual vs Predicted Stock Prices Table")
    st.dataframe(validation, use_container_width=True)
    st.subheader("Actual vs Predicted Stock Prices Line Chart")
    output_data = pd.DataFrame(validation, columns=['Price', 'Predictions'])
    st.line_chart(output_data)

