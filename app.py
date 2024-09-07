import sys
sys.path.append('/Users/koujinkume/.pyenv/versions/3.12.5/lib/python3.12/site-packages')
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
import chardet

# ページ設定
st.set_page_config(layout="wide", page_title="従業員エンゲージメント分析")

# CSSでデザインをカスタマイズ
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
    }
    .Widget>label {
        color: #31333F;
        font-weight: bold;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #4CAF50;
        border-radius: 5px;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
    h1 {
        color: #31333F;
    }
    h2 {
        color: #4CAF50;
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(#4CAF50, #2E7D32);
    }
    .sidebar .sidebar-content .stRadio > div {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .sidebar .sidebar-content .stRadio > div:hover {
        background-color: rgba(255, 255, 255, 0.2);
    }
    .sidebar .sidebar-content .stRadio > div > label {
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# 日本語フォントの設定
plt.rcParams['font.family'] = 'MS Gothic'

@st.cache_data
def load_csv(uploaded_file):
    if uploaded_file is not None:
        # エンコーディング検出
        raw_data = uploaded_file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']

        # ファイルを再読み込み（最初に全て文字列型として読み込み）
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding=encoding)
        return df
    return None


def preprocess_data(df):
    # 欠損値があっても無視して処理を続ける
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    return df, numeric_columns


def descriptive_statistics(df, numeric_columns):
    st.write("基本統計量:")
    st.write(df[numeric_columns].describe())
    
    for column in numeric_columns:
        fig = px.histogram(df, x=column, title=f"{column}の分布")
        st.plotly_chart(fig)

def correlation_analysis(df, numeric_columns):
    corr_matrix = df[numeric_columns].corr()
    
    fig = px.imshow(corr_matrix, 
                    labels=dict(color="相関係数"),
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    color_continuous_scale="RdBu_r")
    fig.update_layout(title="相関係数のヒートマップ")
    st.plotly_chart(fig)

    # 新機能: 特定の変数との高相関項目を見つける
    st.subheader("特定の変数との高相関項目")
    target_var = st.selectbox("対象変数を選択してください", numeric_columns)
    n_items = st.slider("表示する項目数", min_value=1, max_value=len(numeric_columns)-1, value=5)
    
    correlations = corr_matrix[target_var].abs().sort_values(ascending=False)
    top_correlations = correlations[correlations.index != target_var][:n_items]
    
    fig = px.bar(x=top_correlations.index, y=top_correlations.values,
                 labels={'x': '変数', 'y': f'{target_var}との相関係数'},
                 title=f"{target_var}との上位{n_items}個の高相関項目")
    st.plotly_chart(fig)

def regression_analysis(df, numeric_columns):
    target = st.selectbox("目的変数を選択してください", numeric_columns)
    features = st.multiselect("説明変数を選択してください", [col for col in numeric_columns if col != target])
    
    if len(features) == 0:
        st.warning("説明変数を選択してください。")
        return
    
    X = df[features]
    y = df[target]
    
    # if len(features) == 1:
    #     slope, intercept, r_value, p_value, std_err = stats.linregress(X[features[0]], y)
        
    #     fig = px.scatter(df, x=features[0], y=target, trendline="ols")
    #     fig.update_layout(title=f"{features[0]} vs {target}")
    #     st.plotly_chart(fig)
        
    #     st.write(f"回帰式: y = {slope:.4f}x + {intercept:.4f}")
    #     st.write(f"決定係数 (R^2): {r_value**2:.4f}")
    #     st.write(f"p値: {p_value:.4f}")
    # else:
    model = LinearRegression()
    model.fit(X, y)
    
    coefficients = pd.DataFrame({"特徴量": features, "係数": model.coef_})
    fig = px.bar(coefficients, x="特徴量", y="係数", title="重回帰分析: 特徴量の係数")
    st.plotly_chart(fig)
    
    st.write(f"切片: {model.intercept_:.4f}")
    st.write(f"決定係数 (R^2): {model.score(X, y):.4f}")

# 新機能：クロス集計の実装
def crosstab_analysis(df):
    st.subheader("クロス集計")
    
    # クロス集計に使う変数を選択
    column_x = st.selectbox("クロス集計のX軸を選択してください", df.columns)
    column_y = st.selectbox("クロス集計のY軸を選択してください", df.columns)
    
    # クロス集計実行
    crosstab = pd.crosstab(df[column_x], df[column_y])
    st.write("クロス集計結果（頻度）:")
    st.write(crosstab)
    
    # クロス集計のヒートマップ
    fig = px.imshow(crosstab, text_auto=True, aspect="auto", labels=dict(color="頻度"))
    st.plotly_chart(fig)

    # クロス集計を「横軸で合計100%になる割合」に変換
    crosstab_percent = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
    st.write("クロス集計結果（横軸で合計100%になる割合）:")
    st.write(crosstab_percent)
    
    # 割合のヒートマップ
    fig_percent = px.imshow(crosstab_percent, text_auto=True, aspect="auto", labels=dict(color="割合 (%)"))
    st.plotly_chart(fig_percent)


def main():
    st.sidebar.title("メニュー")
    page = st.sidebar.radio("", ["ホーム", "データ分析", "可視化", "回帰分析", "クロス集計"], format_func=lambda x: f"📊 {x}")

    if 'df' not in st.session_state:
        st.session_state.df = None

    if page == "ホーム":
        st.title("従業員エンゲージメント分析アプリ")
        st.write("このアプリケーションでは、従業員エンゲージメントデータの分析を行うことができます。")
        st.write("左側のメニューから各機能を選択してください。")

        uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type="csv")
        if uploaded_file is not None:
            st.session_state.df = load_csv(uploaded_file)
            st.success("ファイルが正常にアップロードされました。他のページで分析を行うことができます。")

    elif page == "データ分析":
        st.title("データ分析")
        
        if st.session_state.df is not None:
            st.write("データの先頭5行:")
            st.write(st.session_state.df.head())
            
            df, numeric_columns = preprocess_data(st.session_state.df)
            
            st.subheader("記述統計")
            descriptive_statistics(df, numeric_columns)
        else:
            st.warning("データがアップロードされていません。ホームページでCSVファイルをアップロードしてください。")

    elif page == "可視化":
        st.title("データ可視化")
        
        if st.session_state.df is not None:
            df, numeric_columns = preprocess_data(st.session_state.df)
            
            st.subheader("相関分析")
            correlation_analysis(df, numeric_columns)
            
            st.subheader("散布図")
            x_axis = st.selectbox("X軸を選択", numeric_columns)
            y_axis = st.selectbox("Y軸を選択", numeric_columns)
            
            fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis}")
            st.plotly_chart(fig)
        else:
            st.warning("データがアップロードされていません。ホームページでCSVファイルをアップロードしてください。")

    elif page == "回帰分析":
        st.title("回帰分析")
        
        if st.session_state.df is not None:
            df, numeric_columns = preprocess_data(st.session_state.df)
            regression_analysis(df, numeric_columns)
        else:
            st.warning("データがアップロードされていません。ホームページでCSVファイルをアップロードしてください。")

    elif page == "クロス集計":
        st.title("クロス集計")
        
        if st.session_state.df is not None:
            crosstab_analysis(st.session_state.df)
        else:
            st.warning("データがアップロードされていません。ホームページでCSVファイルをアップロードしてください。")

if __name__ == "__main__":
    main()
