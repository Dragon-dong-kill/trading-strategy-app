import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
import traceback
import base64
from io import BytesIO

# 设置页面配置
st.set_page_config(
    page_title="交易策略回测工具",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 数据源库
DATA_SOURCES = {
    "AK47 | 血腥运动": "https://sdt-api.ok-skins.com/user/steam/category/v1/kline?timestamp={}&type=2&maxTime={}&typeVal=553370749&platform=YOUPIN&specialStyle",
    "蝴蝶刀": "https://sdt-api.ok-skins.com/user/steam/category/v1/kline?timestamp={}&type=2&maxTime={}&typeVal=22779&platform=YOUPIN&specialStyle",
    "树篱迷宫": "https://sdt-api.ok-skins.com/user/steam/category/v1/kline?timestamp={}&type=2&maxTime={}&typeVal=525873303&platform=YOUPIN&specialStyle",
    "水栽竹": "https://sdt-api.ok-skins.com/user/steam/category/v1/kline?timestamp={}&type=2&maxTime={}&typeVal=24283&platform=YOUPIN&specialStyle",
    "怪兽在b": "https://sdt-api.ok-skins.com/user/steam/category/v1/kline?timestamp={}&type=2&maxTime={}&typeVal=1315999843394654208&platform=YOUPIN&specialStyle",
    "金刚犬": "https://sdt-api.ok-skins.com/user/steam/category/v1/kline?timestamp={}&type=2&maxTime={}&typeVal=1315844312734502912&platform=YOUPIN&specialStyle",
    "tyloo": "https://sdt-api.ok-skins.com/user/steam/category/v1/kline?timestamp={}&type=2&maxTime={}&typeVal=925497374167523328&platform=YOUPIN&specialStyle",
    "迈阿密人士": "https://sdt-api.ok-skins.com/user/steam/category/v1/kline?timestamp={}&type=2&maxTime={}&typeVal=808805648347430912&platform=YOUPIN&specialStyle",
}

# 保存您的原始函数
def get_kline(url, start_date=None, end_date=None):
    """爬取网站K线数据（保持原始结构，增加时间范围筛选）"""
    kline_ls = []
    
    # 处理时间范围
    end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp()) if end_date else int(datetime.now().timestamp())
    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp()) if start_date else 0
    
    while 1:
        ts = int(datetime.now().timestamp() * 1000)
        try:
            with st.spinner(f"正在获取数据..."):
                data = requests.get(url.format(ts, end_ts)).json()['data']
            
            if len(data) == 0:
                break
                
            kline_ls += data
            end_ts = int(data[0][0]) - 86400
            
            if start_date and end_ts < start_ts:
                break
        except Exception as e:
            st.error(f"获取数据出错: {e}")
            break
    
    if not kline_ls:
        return pd.DataFrame(columns=['close']).set_index('date')
        
    kline_df = pd.DataFrame(kline_ls)[[0, 2]]
    kline_df.columns = ['date', 'close']
    kline_df['date'] = kline_df['date'].apply(lambda x: datetime.fromtimestamp(int(x)))
    
    if start_date or end_date:
        mask = True
        if start_date:
            mask = mask & (kline_df['date'] >= datetime.strptime(start_date, '%Y-%m-%d'))
        if end_date:
            mask = mask & (kline_df['date'] <= datetime.strptime(end_date, '%Y-%m-%d'))
        kline_df = kline_df[mask]
    
    return kline_df.set_index('date')[['close']].sort_index()

def t7_adjust(flag):
    """t+7模式调整"""
    for i in range(1, len(flag)):
        if flag.iloc[i] > flag.iloc[i - 1]:
            start = i
        elif flag.iloc[i] < flag.iloc[i - 1] and i - start < 7:
            flag.iloc[i] = 1
    return flag

def backtest(kline_df, k0=6.7, bias_th=0.07, sell_days=3, sell_drop_th=-0.05):
    """回测函数，增加仓位记录和买卖信号"""
    # 计算指标
    ret = kline_df['close'].pct_change()
    ma5 = kline_df['close'].rolling(5).mean()
    ma10 = kline_df['close'].rolling(10).mean()
    ma20 = kline_df['close'].rolling(20).mean()
    
    # 执行回测
    pos = {}
    ret_ls = []
    
    for i in range(19, len(kline_df)):
        close = kline_df['close'].iloc[i]
        bias = close / ma5.iloc[i] - 1
        
        # 计算价格跌幅
        price_drop = 0
        ma10_break = False
        if i >= sell_days:
            drop_cal = kline_df['close'].iloc[i-sell_days]
            price_drop = close / drop_cal - 1
            ma10_break = close < ma10.iloc[i]
        
        current_pos = sum(list(pos.values()))
        buy = 0
        sell = 0
        sold_pos = 0
        
        # 买入逻辑
        if ma5.iloc[i] > ma20.iloc[i] and close > ma10.iloc[i] and bias < bias_th:
            if not pos:
                pos[i] = 0.3
                buy = 0.3
            elif current_pos < 1:
                pos[i] = 0.1
                buy = 0.1
        # 卖出逻辑
        else:
            # 清仓条件：3日跌幅超5%且跌破MA10
            if i >= sell_days and price_drop < sell_drop_th and ma10_break:
                sell_pos = current_pos  # 全额卖出
                for k in list(pos.keys()):  # 清空所有持仓
                    sold_pos += pos[k]
                    del pos[k]
            else:
                # 保持原有止盈逻辑
                sell_pos = current_pos * (1 - np.exp(-k0 * bias_th)) if bias >= bias_th else 0
                for k in list(pos.keys()):
                    if i - k >= 7:
                        sold_pos += pos[k]
                        del pos[k]
                        if sold_pos >= sell_pos:
                            break
            
            sell = sold_pos
        
        # 记录当日结果
        ret_ls.append({
            'date': kline_df.index[i],
            'pos': current_pos + buy - sell,
            'ret': (current_pos + buy - sell) * ret.iloc[i],
            'buy': buy,
            'sell': sell
        })
    
    return pd.DataFrame(ret_ls).set_index('date')

def get_risk(df, num=365):
    """计算策略收益情况"""
    value_df = (1 + df).cumprod()
    annual_ret = value_df.iloc[-1] ** (num / len(df)) - 1
    vol = df.std() * np.sqrt(num)
    sharpe = annual_ret / vol
    max_dd = (1 - value_df / value_df.cummax()).max()
    calmar = annual_ret / max_dd
    return {
        '总收益率': (value_df.iloc[-1] - 1).tolist(),
        '年化收益': annual_ret.tolist(),
        '波动率': vol.tolist(),
        'Sharpe': sharpe.tolist(),
        '最大回撤': max_dd.tolist(),
        'Calmar': calmar.tolist()
    }

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1976D2;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .param-title {
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .success-box {
        padding: 1rem;
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f1f7fd;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #1976D2;
    }
</style>
""", unsafe_allow_html=True)

# 应用标题
st.markdown('<h1 class="main-header">交易策略回测工具</h1>', unsafe_allow_html=True)

# 侧边栏设置
with st.sidebar:
    st.header("参数设置")
    
    # 数据源选择
    data_source = st.selectbox(
        "选择数据源",
        options=list(DATA_SOURCES.keys()),
        index=3  # 默认选择水栽竹
    )
    
    # 日期选择
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "开始日期",
            value=datetime.now() - timedelta(days=45)
        )
    with col2:
        end_date = st.date_input(
            "结束日期",
            value=datetime.now()
        )
    
    # 策略参数
    st.subheader("策略参数")
    
    k_value = st.number_input("止盈参数 k", 
                              value=6.7, 
                              step=0.1, 
                              format="%.1f")
    
    bias_threshold = st.number_input("止盈阈值", 
                                    value=0.07, 
                                    step=0.01, 
                                    format="%.2f")
    
    sell_days = st.number_input("止损天数", 
                               value=3, 
                               step=1)
    
    sell_drop_th = st.number_input("止损阈值", 
                                  value=-0.05, 
                                  step=0.01, 
                                  format="%.2f")
    
    # 显示设置
    st.subheader("显示设置")
    show_benchmark = st.checkbox("大盘走势", value=True)
    show_basic = st.checkbox("5/20基本策略", value=True)
    show_extended = st.checkbox("5/20拓展策略", value=True)
    
    # 运行按钮
    run_button = st.button("运行回测", use_container_width=True)

# 初始化会话状态
if 'result_data' not in st.session_state:
    st.session_state.result_data = None
if 'kline_data' not in st.session_state:
    st.session_state.kline_data = None
if 'bt_df' not in st.session_state:
    st.session_state.bt_df = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None

# 运行回测
if run_button:
    try:
        st.markdown('<h2 class="sub-header">回测进度</h2>', unsafe_allow_html=True)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 获取数据URL
        data_url = DATA_SOURCES.get(data_source, "")
        if not data_url:
            st.error("数据URL不能为空")
            st.stop()
        
        # 转换日期为字符串格式
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # 获取K线数据
        status_text.text("正在获取K线数据...")
        progress_bar.progress(10)
        
        kline_df = get_kline(data_url, start_date_str, end_date_str)
        st.session_state.kline_data = kline_df
        
        if kline_df.empty:
            st.error("指定时间范围内没有K线数据，请调整日期范围")
            st.stop()
        
        status_text.text(f"已获取 {len(kline_df)} 条数据记录")
        progress_bar.progress(40)
        
        # 计算指标
        status_text.text("计算技术指标...")
        progress_bar.progress(50)
        
        ret = kline_df['close'].pct_change()
        ma5 = kline_df['close'].rolling(5).mean()
        ma10 = kline_df['close'].rolling(10).mean()
        ma20 = kline_df['close'].rolling(20).mean()
        
        # 策略计算
        status_text.text("执行策略回测...")
        progress_bar.progress(60)
        
        ret_map = {
            'benchmark': ret  # 大盘走势
        }
        
        # ma5/20策略
        flag = ((ma5 > ma20) & (kline_df['close'] > ma10)).apply(int).shift()
        flag = t7_adjust(flag)
        ret_map['basic'] = ret * flag  # 5/20基本策略
        
        # ma5/20策略（仓位管理）
        bt_df = backtest(kline_df, k_value, bias_threshold, sell_days, sell_drop_th)
        ret_map['extended'] = bt_df['ret']  # 5/20拓展策略
        st.session_state.bt_df = bt_df
        
        # 转换为DataFrame
        ret_df = pd.DataFrame(ret_map)
        
        # 计算风险/收益指标
        status_text.text("计算绩效指标...")
        progress_bar.progress(80)
        
        risk_metrics = get_risk(ret_df)
        st.session_state.metrics = risk_metrics
        
        # 累积收益
        value_df = (ret_df + 1).cumprod() - 1
        
        # 保存结果
        st.session_state.result_data = {
            'returns': ret_df,
            'cumulative': value_df,
            'metrics': risk_metrics,
            'source': data_source
        }
        
        progress_bar.progress(100)
        status_text.text("回测完成！")
        
        # 显示成功消息
        st.markdown(f"""
        <div class="success-box">
            <h3>回测完成</h3>
            <p>数据源: {data_source}</p>
            <p>参数: K={k_value}, 阈值={bias_threshold}, 止损天数={sell_days}, 止损阈值={sell_drop_th}</p>
            <p>数据范围: {start_date_str} 至 {end_date_str}, 共 {len(kline_df)} 条记录</p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"回测过程出错: {str(e)}")
        st.text(traceback.format_exc())

# 显示结果
if st.session_state.result_data is not None:
    # 显示图表
    st.markdown('<h2 class="sub-header">回测结果图表</h2>', unsafe_allow_html=True)
    
    # 使用Plotly创建交互式图表
    cum_returns = st.session_state.result_data['cumulative']
    bt_df = st.session_state.bt_df
    
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('策略累积收益', '总仓位分布', '买卖明细'),
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # 第一个子图：累积收益率
    if show_benchmark:
        fig.add_trace(
            go.Scatter(
                x=cum_returns.index, 
                y=cum_returns['benchmark'],
                mode='lines',
                name='大盘走势',
                line=dict(color='#4E79A7', width=2)
            ),
            row=1, col=1
        )
    
    if show_basic:
        fig.add_trace(
            go.Scatter(
                x=cum_returns.index, 
                y=cum_returns['basic'],
                mode='lines',
                name='5/20基本策略',
                line=dict(color='#F28E2B', width=2)
            ),
            row=1, col=1
        )
    
    if show_extended:
        fig.add_trace(
            go.Scatter(
                x=cum_returns.index, 
                y=cum_returns['extended'],
                mode='lines',
                name='5/20拓展策略',
                line=dict(color='#59A14F', width=2)
            ),
            row=1, col=1
        )
    
    # 第二个子图：仓位分布
    fig.add_trace(
        go.Bar(
            x=bt_df.index,
            y=bt_df['pos'],
            name='持仓',
            marker_color='steelblue'
        ),
        row=2, col=1
    )
    
    # 第三个子图：买卖明细
    fig.add_trace(
        go.Bar(
            x=bt_df.index,
            y=bt_df['buy'],
            name='买入',
            marker_color='green'
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=bt_df.index,
            y=-bt_df['sell'],
            name='卖出',
            marker_color='red'
        ),
        row=3, col=1
    )
    
    # 更新布局
    fig.update_layout(
        height=800,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_white',
        hovermode="x unified"
    )
    
    # 格式化y轴为百分比
    fig.update_yaxes(tickformat='.1%', row=1, col=1)
    
    # 显示图表
    st.plotly_chart(fig, use_container_width=True)
    
    # 显示绩效指标
    st.markdown('<h2 class="sub-header">绩效指标</h2>', unsafe_allow_html=True)
    
    metrics = st.session_state.metrics
    
    # 使用列布局
    col1, col2, col3 = st.columns(3)
    
    # 显示各项指标
    metrics_list = list(metrics.keys())
    
    for i, metric in enumerate(metrics_list):
        col_idx = i % 3
        values = metrics[metric]
        
        if col_idx == 0:
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="param-title">{metric}</p>
                    <p>大盘走势: {values[0]:.2%}</p>
                    <p>5/20基本策略: {values[1]:.2%}</p>
                    <p>5/20拓展策略: {values[2]:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
        elif col_idx == 1:
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="param-title">{metric}</p>
                    <p>大盘走势: {values[0]:.2%}</p>
                    <p>5/20基本策略: {values[1]:.2%}</p>
                    <p>5/20拓展策略: {values[2]:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="param-title">{metric}</p>
                    <p>大盘走势: {values[0]:.2%}</p>
                    <p>5/20基本策略: {values[1]:.2%}</p>
                    <p>5/20拓展策略: {values[2]:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # 导出功能
    st.markdown('<h2 class="sub-header">数据导出</h2>', unsafe_allow_html=True)
    
    # 将数据转换为CSV
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv().encode('utf-8')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.kline_data is not None:
            csv = convert_df_to_csv(st.session_state.kline_data)
            st.download_button(
                label="下载原始K线数据",
                data=csv,
                file_name='kline_data.csv',
                mime='text/csv',
            )
    
    with col2:
        if st.session_state.bt_df is not None:
            csv = convert_df_to_csv(st.session_state.bt_df)
            st.download_button(
                label="下载交易记录",
                data=csv,
                file_name='trade_data.csv',
                mime='text/csv',
            )
    
    with col3:
        if st.session_state.result_data is not None:
            csv = convert_df_to_csv(st.session_state.result_data['cumulative'])
            st.download_button(
                label="下载累积收益数据",
                data=csv,
                file_name='cumulative_returns.csv',
                mime='text/csv',
            )

# 应用底部信息
st.markdown("---")
st.markdown("📊 交易策略回测工具 - 可在手机和电脑上使用的轻量级应用")
