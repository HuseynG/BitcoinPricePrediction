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
    return preds


def main():
    st.title('Live BTC data')
    # creating a placeholder for the graph
    chart = st.empty()
    window_num = 7
    window_size = 24
    while True:
        lookback = window_size * window_num # days
        
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
        prep_btc_data_for_prediction(lookback, pred_len = 0) # for prediction
        preds = NLinear_preditc_next12th()
        # print(f"predictions len {len(preds)}")
        
        next_ith = 1
        next_ith_ = next_ith
        next_ith = abs(next_ith - 13) # this is because the ordering is in reverse 1 means 12th and 12th index mean the next item 
        preds = preds[:(window_size*window_num*next_ith):next_ith] 
        X_preds = [(datetime.fromtimestamp(end_time) + timedelta(hours=i-(lookback)+13-next_ith_)).replace(minute=0, second=0).strftime('%Y-%m-%d %H:%M:%S') for i in range(len(preds))]
        # print(f"X: {len(X)}, X_preds: {len(X_preds)}")

        fig = go.Figure()

        fig.add_trace(go.Candlestick(x=X,
                                    open=open_data,
                                    high=high_data,
                                    low=low_data,
                                    close=close_data,
                                    name='actual',
                                    increasing_line_color= 'green', 
                                    decreasing_line_color= 'red'))

        fig.add_trace(go.Scatter(x=X_preds,
                         y=preds,
                         mode='lines',
                         line=dict(color='yellow'),
                         name='Next Step Prediction',
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