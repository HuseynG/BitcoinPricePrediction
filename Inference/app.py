import streamlit as st
import plotly.graph_objs as go
import requests
import time
from datetime import datetime, timedelta
from helper_util import Train_Tester, get_btc_data, prep_btc_data_for_prediction


def preditc_next():
    args_dict_test = {
    "root_path": "./",
    "data_path": "data.csv",
    "model_id": "d_linear_btc_12_1_1_close_only",
    "model": "DLinear",
    "data": "custom",
    "seq_len": 12,
    "label_len": 1,
    "pred_len": 1,
    "enc_in": 1,
    "individual": True,
    "use_gpu": 0,
    "batch_size": 1,
    "devices": "0,3",
    }

    preds, trues, mse, mae = Train_Tester(**args_dict_test).test("../Prediction_Models/other_models_to_compare/LTSF-Linear-main/checkpoints/d_linear_btc_12_1_1_close_only_DLinear_custom_ftMS_sl12_ll1_pl1_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth")
    preds = preds.flatten().tolist()
    # print("Preds form predict next", preds)
    return preds



def main():
    st.title('Live BTC data')

    # creating a placeholder for the graph
    chart = st.empty()

    while True:
        prep_btc_data_for_prediction(12) # for prediction
        end_time = int(time.time())
        start_time = end_time - (12 * 60 * 60)
        interval = 3600
        btc_data = get_btc_data(start_time, end_time, interval)

        X = [datetime.fromtimestamp(int(data['timestamp'])).strftime('%Y-%m-%d %H:%M:%S') for data in btc_data]
        open_data = [data['open'] for data in btc_data]
        high_data = [data['high'] for data in btc_data]
        low_data = [data['low'] for data in btc_data]
        close_data = [data['close'] for data in btc_data]


        preds = preditc_next()
        X_pred = [(datetime.fromtimestamp(end_time) + timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')]
        open_data_pred = [preds[0]]
        high_data_pred = [preds[0]]
        low_data_pred = [preds[0]]
        close_data_pred = [preds[0]]

        # creating a Plotly figure with two traces
        fig = go.Figure()

        fig.add_trace(go.Candlestick(x=X,
                                    open=open_data,
                                    high=high_data,
                                    low=low_data,
                                    close=close_data,
                                    name='actual',
                                    increasing_line_color= 'green', 
                                    decreasing_line_color= 'red'))
        
        fig.add_trace(go.Candlestick(x=X_pred,
                                    open=open_data_pred,
                                    high=high_data_pred,
                                    low=low_data_pred,
                                    close=close_data_pred,
                                    name='predicted',
                                    increasing_line_color= '#d7fc03', 
                                    decreasing_line_color= '#fc7703'))
                                    
        # colors for the line
        up_color = '#d7fc03'
        down_color = '#fc7703'
        
        line_color = up_color if float(close_data_pred[0]) > float(close_data[-1]) else down_color
        
        fig.add_trace(go.Scatter(x=[X[0], X_pred[0]], 
                                y=[close_data[-1], close_data_pred[0]], 
                                mode='lines',
                                line=dict(color=line_color),
                                showlegend=False))


        fig.update_layout(
            autosize=True,
            width=900,
            height=800,
            margin=dict(
                l=50,
                r=50,
                b=100,
                t=100,
                pad=4
            ),
            yaxis=dict(
                type="log",  # logarithmic scale
                title="Price",
            ),
            xaxis_title="Time",
            )

        st.plotly_chart(fig)

        # updating the graph every 30 minutes (1800 seconds)
        time.sleep(1800)

if __name__ == "__main__":
    main()