import streamlit as st
import plotly.graph_objs as go
import requests
import time
from datetime import datetime, timedelta
from helper_util import Train_Tester, get_btc_data, prep_btc_data_for_prediction


def NLinear_preditc_next12th():
    args_dict_test = {
    "root_path": "./",
    "data_path": "data.csv",
    "model_id": "n_linear_btc_12_12_12_close_only",
    "model": "NLinear",
    "data": "custom",
    "seq_len": 12,
    "label_len": 12,
    "pred_len": 12,
    "enc_in": 1,
    "individual": True,
    "use_gpu": 0,
    "batch_size": 1,
    "devices": "0,3",
    }

    preds, trues, mse, mae = Train_Tester(**args_dict_test).test("../Prediction_Models/other_models_to_compare/LTSF-Linear-main/checkpoints/n_linear_btc_12_12_12_close_only_NLinear_custom_ftMS_sl12_ll12_pl12_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth")
    preds = preds.flatten().tolist()
    preds_12th_and_last_12 = preds[::12] + preds[-12:]  # get every 12th prediction and the last 12 predictions
    return preds_12th_and_last_12


def main():
    st.title('Live BTC data')
    # creating a placeholder for the graph
    chart = st.empty()

    while True:
        lookback = 504 * 2 # weeks
        
        end_time = int(time.time())
        start_time = end_time - (lookback * 60 * 60)
        interval = 3600
        btc_data = get_btc_data(start_time, end_time, interval)

        X = [datetime.fromtimestamp(int(data['timestamp'])).strftime('%Y-%m-%d %H:%M:%S') for data in btc_data]
        open_data = [data['open'] for data in btc_data]
        high_data = [data['high'] for data in btc_data]
        low_data = [data['low'] for data in btc_data]
        close_data = [data['close'] for data in btc_data]

        # predictions from the NLinear_preditc_next12th model
        prep_btc_data_for_prediction(lookback, pred_len = 12) # for prediction
        preds_12th = NLinear_preditc_next12th()
        preds_12th = preds_12th[::-1]
        X_pred_12th = [(datetime.fromtimestamp(end_time) + timedelta(hours=i-(lookback))).replace(minute=0, second=0).strftime('%Y-%m-%d %H:%M:%S') for i in range(len(preds_12th))]
        close_data_pred_12th = preds_12th
        # creating a Plotly figure with three traces
        fig = go.Figure()

        fig.add_trace(go.Candlestick(x=X,
                                    open=open_data,
                                    high=high_data,
                                    low=low_data,
                                    close=close_data,
                                    name='actual',
                                    increasing_line_color= 'green', 
                                    decreasing_line_color= 'red'))

        fig.add_trace(go.Scatter(x=X_pred_12th,
                         y=close_data_pred_12th,
                         mode='lines',
                         line=dict(color='yellow'),
                         name='12th time point prediction',
                         showlegend=True))
                         

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