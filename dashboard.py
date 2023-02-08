import streamlit as st
import pandas as pd
import plotly.express as px
import tensorflow as tf
import os


choose_model = st.sidebar.selectbox("Choose_model", ('BMW SERIES 5','BMW SERIES 3','MERCEDES-BENZ S-CLASS','MERCEDES-BENZ E-CLASS','VOLKSWAGEN GOLF'))
option = st.sidebar.selectbox("Choose option", ('DNN REGRESSION',"SHOW DATAFRAME", 'ABOUT'))
st.header(option)

cwd = os.getcwd()
csv_files = {
    'BMW SERIES 5': "bmw_series_5_otomoto",
    'BMW SERIES 3': "bmw3",
    'MERCEDES-BENZ S-CLASS': "s_class",
    'MERCEDES-BENZ E-CLASS': "e_class",
    'VOLKSWAGEN GOLF': "vw_golf",
}
dnn_folders = {
    'BMW SERIES 5': 'dnn_bmw5',
    'BMW SERIES 3' : "bmw3",
    'MERCEDES-BENZ S-CLASS': "s_class",
    'MERCEDES-BENZ E-CLASS': "e_class",
    'VOLKSWAGEN GOLF': "vw_golf",
}

cwd = os.getcwd()
df = pd.read_csv(f'{cwd}/csv/{csv_files[choose_model]}.csv', delimiter= ';')
dnn_model = tf.keras.models.load_model(f'{cwd}/models/{dnn_folders[choose_model]}')

if option == 'SHOW DATAFRAME':
    price = df[['production_date','price']].copy()
    price = df.groupby('production_date')['price'].mean()
    fig = px.line(price, x=price.index, y="price",title="Mean price")
    fig.update_layout(xaxis_title='production date', yaxis_title='price')
    st.plotly_chart(fig, use_container_width=True)

    mileage = df[['production_date','mileage']].copy()
    mileage = df.groupby('production_date')['mileage'].mean()
    fig2 = px.line(mileage, x=mileage.index, y="mileage",title="Mean mileage")
    fig2.update_layout(xaxis_title='production date', yaxis_title='mileage')
    st.plotly_chart(fig2, use_container_width=True)

    capacity = df[['production_date','engine_capacity']].copy()
    capacity = df.groupby('engine_capacity')['production_date'].count()
    fig3 = px.scatter(capacity, x=capacity.index, y="production_date",size="production_date",title="Engine capacity quantity",size_max=60)
    fig3.update_layout(xaxis_title='engine capacity', yaxis_title='count')
    st.plotly_chart(fig3, use_container_width=True)

    volume = df[['production_date','engine_capacity']].copy()
    volume = df.groupby('production_date')['engine_capacity'].count()
    fig4 = px.bar(volume, x=volume.index, y="engine_capacity",title="Volume")
    fig3.update_layout(xaxis_title='volume', yaxis_title='count')
    st.plotly_chart(fig4, use_container_width=True)

    st.dataframe(df)

if option == 'DNN REGRESSION':
    production_date = st.number_input("Production date(year)",value = 2000, step = 1)
    mileage = st.number_input("Mileage (km)",value = 100000, step = 1000)
    engine_capacity = st.number_input("Engine capacity (cm3)",value = 2000, step = 100)

    if st.button('Calculate price'):
        test_predictions = int(dnn_model.predict([production_date,mileage,engine_capacity]).flatten())
        st.header(f"{test_predictions} PLN")

        with st.spinner('Wait for additional symulation chart'):
            pred_data = {'price':[], 'mileage':[]}
            for mil in range(0, 200000 + mileage, 10000):
                pred_data['price'].append(int(dnn_model.predict([production_date,mil,engine_capacity]).flatten()))
                pred_data["mileage"].append(mil)
        
            pred_df = pd.DataFrame(data=pred_data)
            fig = px.line(pred_df, x='mileage', y="price",title="Symulated valuation with changing mileage")
            st.plotly_chart(fig, use_container_width=True)

        with st.spinner('Wait for additional symulation chart'):
            pred_data = {'price':[], 'capacity':[]}
            for cap in range(df['engine_capacity'].min(), df['engine_capacity'].max()+1, 200):
                pred_data['price'].append(int(dnn_model.predict([production_date,mileage,cap]).flatten()))
                pred_data["capacity"].append(cap)
            pred_df = pd.DataFrame(data=pred_data)
            fig = px.line(pred_df, x='capacity', y="price",title="Symulated valuation with changing engine capacity")
            st.plotly_chart(fig, use_container_width=True)
    

if option == 'ABOUT':
    st.write("Machine learning project to calculate and symulate praces of popular car models. \
        Deep neural network regression models where trained using tensorfolw liblary and visualized using streamlit. \
         \n\n Data in csv files ,that where used in modeling comes from otomoto scraper script, that is avaible in my github repo -  \
         https://github.com/prosowiec/otomoto_scraper.")
    st.image('images/model.png')
    st.image('images/loss.png')
    