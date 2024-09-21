import dash
from dash import Dash, html, dcc, Output, Input
import plotly.graph_objs as go
import pandas as pd
from flask import Flask, request
import threading
import time
import requests
import random
import datetime


class MyDashApp:
    def __init__(self):
        self.app = Dash(__name__)
        self.server = self.app.server  # 获取 Flask 服务器实例

        # 数据存储
        # 数值类型数据：key -> DataFrame，包含 'timestamp' 和 'value' 列
        self.numerical_data = {}

        # 成功/失败类型数据：key -> DataFrame，包含 'timestamp' 和 'success' 列
        self.status_data = {}

        # 线程锁，保证线程安全
        self.lock = threading.Lock()

        # 设置布局和回调函数
        self.setup_layout()
        self.setup_callbacks()

        # 设置数据接收的 HTTP 接口
        self.setup_data_input_endpoint()

        # 启动后台线程，模拟测试数据
        # threading.Thread(target=self.simulate_data, daemon=True).start()

    def setup_layout(self):
        # 定义应用的布局
        self.app.layout = html.Div([
            html.H1('仪表盘'),
            dcc.Interval(
                id='interval-component',
                interval=1*1000,  # 毫秒
                n_intervals=0
            ),
            html.Div(id='graphs-container')
        ])

    def setup_callbacks(self):
        @self.app.callback(
            Output('graphs-container', 'children'),
            Input('interval-component', 'n_intervals')
        )
        def update_graphs(n_intervals):
            # 根据数据创建图表
            with self.lock:
                graphs = []

                # 处理数值类型数据
                for key, df in self.numerical_data.items():
                    df = df.sort_values('timestamp')

                    # 将时间戳转换为可读的日期时间格式
                    df['timestamp'] = df['timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['value'], mode='lines+markers', name=key))
                    graphs.append(html.Div([
                        html.H3(f'{key} 的数值变化'),
                        dcc.Graph(figure=fig)
                    ]))

                # 处理成功/失败类型数据
                for key, df in self.status_data.items():
                    # 计算每秒成功率
                    df_grouped = df.groupby('timestamp').mean().reset_index()
                    df_grouped = df_grouped.sort_values('timestamp')

                    # 将时间戳转换为可读的日期时间格式
                    df_grouped['timestamp'] = df_grouped['timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_grouped['timestamp'], y=df_grouped['success'], mode='lines+markers', name=key))
                    graphs.append(html.Div([
                        html.H3(f'{key} 的每秒成功率'),
                        dcc.Graph(figure=fig)
                    ]))

                return graphs


    def setup_data_input_endpoint(self):
        @self.server.route('/data', methods=['POST'])
        def receive_data():
            data = request.json
            key = data.get('key')
            data_type = data.get('type')  # 'numerical' 或 'status'
            timestamp = int(time.time())  # 当前时间戳（秒）
            with self.lock:
                if data_type == 'numerical':
                    value = data.get('value')
                    if value is None:
                        return '数值类型的数据需要提供 value', 400
                    if key not in self.numerical_data:
                        self.numerical_data[key] = pd.DataFrame(columns=['timestamp', 'value'])
                    # 如果当前时间戳已经存在，计算新的平均值
                    if timestamp in self.numerical_data[key]['timestamp'].values:
                        # 找到相同时间戳的数据并计算新的平均值
                        existing_values = self.numerical_data[key][self.numerical_data[key]['timestamp'] == timestamp]['value'].tolist()
                        new_value = (sum(existing_values) + value) / (len(existing_values) + 1)
                        # 更新当前时间戳的平均值
                        self.numerical_data[key].loc[self.numerical_data[key]['timestamp'] == timestamp, 'value'] = new_value
                    else:
                        # 如果当前时间戳不存在，直接添加新数据
                        self.numerical_data[key] = pd.concat([
                            self.numerical_data[key],
                            pd.DataFrame({'timestamp': [timestamp], 'value': [value]})
                        ], ignore_index=True)
                elif data_type == 'status':
                    status = data.get('status')  # 'success' 或 'failure'
                    if status not in ('success', 'failure'):
                        return '无效的 status 值', 400
                    success = 1 if status == 'success' else 0
                    if key not in self.status_data:
                        self.status_data[key] = pd.DataFrame(columns=['timestamp', 'success'])
                    # 添加状态数据
                    self.status_data[key] = pd.concat([
                        self.status_data[key],
                        pd.DataFrame({'timestamp': [timestamp], 'success': [success]})
                    ], ignore_index=True)
                else:
                    return '无效的数据类型', 400
            return '数据已接收', 200


    def simulate_data(self):
        # 模拟通过 HTTP 请求写入数据
        while True:
            # 生成测试数据
            # 数值类型数据
            numerical_key = '数值键'
            numerical_value = random.random() * 100
            numerical_data = {
                'key': numerical_key,
                'type': 'numerical',
                'value': numerical_value
            }
            try:
                requests.post('http://127.0.0.1:8050/data', json=numerical_data)
            except requests.exceptions.ConnectionError:
                pass  # 服务器可能还未启动

            # 成功/失败类型数据
            status_key = '状态键'
            status_value = random.choice(['success', 'failure'])
            status_data = {
                'key': status_key,
                'type': 'status',
                'status': status_value
            }
            try:
                requests.post('http://127.0.0.1:8050/data', json=status_data)
            except requests.exceptions.ConnectionError:
                pass

            # 成功/失败类型数据
            status_key = '测试数据'
            status_value = random.choice(['success', 'failure'])
            status_data = {
                'key': status_key,
                'type': 'status',
                'status': status_value
            }
            try:
                requests.post('http://127.0.0.1:8050/data', json=status_data)
            except requests.exceptions.ConnectionError:
                pass

            time.sleep(0.01)

    def run(self):
        # 在本机地址运行应用
        self.app.run_server(host='127.0.0.1', port=8050)

if __name__ == '__main__':
    app = MyDashApp()
    app.run()
