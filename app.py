import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import chardet
import time
import plotly.express as px
import os
import base64
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ページ設定
st.set_page_config(layout="wide", page_title="従業員エンゲージメント分析")

# CSSでデザインをカスタマイズ
def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSSファイルが見つかりません: {file_name}")

# CSSをロード
load_css("styles.css")

# 日本語フォントの設定
plt.rcParams['font.family'] = 'MS Gothic'

# 画像ファイルをBase64形式に変換
def get_image_as_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

#上部を消す


# ポップアップを表示する関数
def display_popup(image_base64):
    popup_html = f"""
    <div id="popup">
        <img src="data:image/png;base64,{image_base64}" id="popup-icon" alt="robot icon">
        <div id="popup-content">
            <h5>HELP</h5>
            <p>データの解釈に困ったり、調査設計や分析の相談は<a href="https://example.com">こちら</a>からどうぞ！</p>
        </div>
    </div>
    """
    st.markdown(popup_html, unsafe_allow_html=True)

# 画像のパスを動的に取得
image_path = os.path.join(os.getcwd(), "robot.png")
if os.path.exists(image_path):
    image_base64 = get_image_as_base64(image_path)
    display_popup(image_base64)
else:
    st.warning("画像が見つかりません")

# CSVファイルの読み込み
@st.cache_data
def load_csv(uploaded_file):
    if uploaded_file is not None:
        try:
            raw_data = uploaded_file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=encoding)
            return df
        except Exception as e:
            st.error(f"ファイルの読み込み中にエラーが発生しました: {str(e)}")
            return None
    return None

# データの前処理
def preprocess_data(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    return df, numeric_columns

# 記述統計の実行
def descriptive_statistics(df, numeric_columns):
    st.subheader("記述統計")
    
    # カラムをユーザーに選ばせる
    selected_column = st.selectbox("変数（カラム）を選んでください", numeric_columns)
    
    if selected_column:
        st.write(f"**選択されたカラム: {selected_column}**")
        
        # 平均値を表示
        mean_value = df[selected_column].mean()
        st.write(f"**平均値:** {mean_value:.2f}")
        
        # 標準偏差を表示
        std_value = df[selected_column].std()
        st.write(f"**標準偏差:** {std_value:.2f}")
        
        # 最大値と最小値を表示
        max_value = df[selected_column].max()
        min_value = df[selected_column].min()
        st.write(f"**最大値:** {max_value}")
        st.write(f"**最小値:** {min_value}")
        
        # 分布（ヒストグラム）を表示
        st.write(f"**{selected_column}の分布**")
        fig = px.histogram(df, x=selected_column, title=f"{selected_column}の分布")
        st.plotly_chart(fig)

# 相関分析の実行
def correlation_analysis(df, numeric_columns):
    corr_matrix = df[numeric_columns].corr()
    
    st.subheader("特定の変数との高相関項目")
    target_var = st.selectbox("分析する変数を選択してください", numeric_columns)
    n_items = st.slider("表示する項目数", min_value=1, max_value=len(numeric_columns)-1, value=5)
    
    correlations = corr_matrix[target_var].abs().sort_values(ascending=False)
    top_correlations = correlations[correlations.index != target_var][:n_items]
    
    fig = px.bar(x=top_correlations.index, y=top_correlations.values,
                 labels={'x': '変数', 'y': f'{target_var}との相関係数'},
                 title=f"{target_var}との上位{n_items}個の高相関項目（相関係数が高い項目から順に並びます）")
    st.plotly_chart(fig)

# マルチコリニアリティのチェック
def check_multicollinearity(X):
    vif_data = pd.DataFrame()
    vif_data["特徴量"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    return vif_data

# 回帰分析の実行
def regression_analysis(df, numeric_columns):
    target = st.selectbox("目的変数を選択してください（因果の果）", numeric_columns)
    features = st.multiselect("説明変数を選択してください（因果の因）", [col for col in numeric_columns if col != target])
    
    if len(features) == 0:
        st.warning("説明変数を選択してください。")
        return
    
    X = df[features]
    y = df[target]

    # 欠損値を削除する
    X = X.dropna()
    y = y.loc[X.index]  # Xの欠損値を削除した後の行に合わせてyを調整
    
    if X.isnull().values.any() or y.isnull().values.any():
        st.error("欠損値を含むデータがあります。欠損値を除去または補完してください。")
        return
    
    # マルチコリニアリティチェック
    vif_result = check_multicollinearity(X)
    st.write("マルチコリニアリティチェック結果:　VIFが全て５未満であれば多重共線性（似過ぎ問題）がないと言えます")
    st.write(vif_result)
    
    # 回帰分析を実行
    model = LinearRegression()
    model.fit(X, y)
    
    coefficients = pd.DataFrame({"特徴量": features, "係数": model.coef_})
    fig = px.bar(coefficients, x="特徴量", y="係数", title="重回帰分析結果: 特徴量の係数（目的変数に対してそれぞれの説明変数がどの程度因果関係が強いかを表します）")
    st.plotly_chart(fig)
    
    # R^2を強調して表示
    r_squared = model.score(X, y)
    st.markdown(f"<p>切片: {model.intercept_:.4f}</p>", unsafe_allow_html=True)
    st.markdown(f"<p>決定係数 (<span style='color:red; font-weight:bold;'>R²: {r_squared:.4f}</span>)　0.5以上で中程度以上の因果関係があると言えます</p>", unsafe_allow_html=True)

# 変数の加工
def variable_processing(df):
    st.subheader("変数の加工")
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_columns = st.multiselect("加工対象の変数を選択してください", numeric_columns)

    if len(selected_columns) > 0:
        # 選択された変数の平均を計算
        average_of_columns = df[selected_columns].mean(axis=1)

        # 平均の分布を表示
        st.write("選択された変数の平均の分布")
        fig = px.histogram(average_of_columns, title="平均の分布")
        st.plotly_chart(fig)

        # 分析データに追加するボタン
        if 'new_column_name' not in st.session_state:
            st.session_state.new_column_name = ""

        new_column_name = st.text_input("新しい変数名を入力してください", value=st.session_state.new_column_name)
        if st.button("分析データに追加する"):
            if new_column_name:
                df[new_column_name] = average_of_columns
                st.session_state.df = df
                st.session_state.new_column_name = new_column_name  # 入力値をセッションに保存
                st.success("分析データに追加しました。")

# クロス集計の実行
@st.cache_data
def crosstab_analysis(df, column_x, column_y, decimal_places):
    # クロス集計の計算（目的変数を行、説明変数を列として設定）
    crosstab = pd.crosstab(df[column_y], df[column_x])

    # 説明変数テキストの位置調整
    col1, col2 = st.columns([1, 4])

    # 説明変数の位置を中央に調整
    with col1:
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write(f"説明変数（表側）: {column_x}")

    with col2:
        st.write(f"クロス集計結果【数表】　目的変数（表頭: {column_y}）")
        crosstab_with_labels = crosstab.copy()
        crosstab_with_labels.index.name = f"{column_y}"
        crosstab_with_labels.columns.name = f"{column_x}"

        # クロス集計結果を整数にキャストして表示
        crosstab_with_labels = crosstab_with_labels.astype(int)
        st.write(crosstab_with_labels)

    # 横軸で合計100%になる割合を計算
    crosstab_percent = crosstab.div(crosstab.sum(axis=1), axis=0) * 100

    # 説明変数テキストの位置調整（割合表）
    col1, col2 = st.columns([1, 4])
    with col1:
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write(f"説明変数（表側）: {column_x}")

    with col2:
        st.write(f"クロス集計結果【%表】　目的変数（表頭: {column_y}）")
        crosstab_percent_with_labels = crosstab_percent.copy()
        crosstab_percent_with_labels.index.name = f"{column_y}"
        crosstab_percent_with_labels.columns.name = f"{column_x}"

        # 選択された小数点位数で丸めて表示
        st.dataframe(crosstab_percent_with_labels.round(decimal_places))

def main():
    st.sidebar.title("Menu")
    menu_items = ["データアップロード", "記述統計", "相関分析", "回帰分析", "クロス集計", "変数の加工"]  # メニューに「変数の加工」を追加

    page = st.sidebar.selectbox("メニューを選択してください", menu_items)

    if 'df' not in st.session_state:
        st.session_state.df = None

    if page == "データアップロード":
        st.title("従業員エンゲージメント分析アプリ")

        # アップロードUIを常に表示し、新しいファイルを選択できる
        uploaded_file = st.file_uploader("CSVファイルをアップロードしてください。個人情報や機密情報は含めないでください", type="csv")

        # アップロードされたファイルが新しい場合のみ処理を行う
        if uploaded_file is not None:
            if 'uploaded_file' not in st.session_state or uploaded_file.name != st.session_state.uploaded_file_name:
                # アップロードしたファイルをセッションに保存
                st.session_state.uploaded_file = uploaded_file
                st.session_state.uploaded_file_name = uploaded_file.name
                with st.spinner('分析中...'):
                    st.session_state.df = load_csv(uploaded_file)
                    st.success("ファイルが正常にアップロードされました。メニューから他のページで分析を行うことができます。")

        # アップロード済みのファイルを表示
        if 'uploaded_file' in st.session_state:
            st.write(f"アップロードされたファイル: {st.session_state.uploaded_file_name}")

    elif page == "記述統計":
        st.title("記述統計「概要を把握する」")
        if st.session_state.df is not None:
            df, numeric_columns = preprocess_data(st.session_state.df)
            descriptive_statistics(df, numeric_columns)
        else:
            st.warning("データがアップロードされていません。データアップロードページでCSVファイルをアップロードしてください。")

    elif page == "相関分析":
        st.title("相関分析「重要なKPIを見つける」")
        if st.session_state.df is not None:
            df, numeric_columns = preprocess_data(st.session_state.df)
            correlation_analysis(df, numeric_columns)
        else:
            st.warning("データがアップロードされていません。データアップロードページでCSVファイルをアップロードしてください。")

    elif page == "回帰分析":
        st.title("回帰分析「因果関係を検証する」")
        if st.session_state.df is not None:
            df, numeric_columns = preprocess_data(st.session_state.df)
            regression_analysis(df, numeric_columns)
        else:
            st.warning("データがアップロードされていません。データアップロードページでCSVファイルをアップロードしてください。")

    elif page == "変数の加工":
        st.title("変数の加工")
        if st.session_state.df is not None:
            variable_processing(st.session_state.df)
        else:
            st.warning("データがアップロードされていません。データアップロードページでCSVファイルをアップロードしてください。")      

    elif page == "クロス集計":
        st.title("クロス集計「群ごとの傾向の違いを見る」")
        if st.session_state.df is not None:
            column_y = st.selectbox("クロス集計の表頭（目的変数）を選択してください", st.session_state.df.columns)
            column_x = st.selectbox("クロス集計の表側（説明変数）を選択してください", st.session_state.df.columns)
            # セレクトボックスに特定のIDを付与する
            st.markdown('<div class="small-selectbox">', unsafe_allow_html=True)
            decimal_places = st.selectbox("%表の小数点の表示桁数を選択してください", [1, 2, 3, 4], index=1)
            st.markdown('</div>', unsafe_allow_html=True)

            with st.spinner('分析中...'):
                time.sleep(2)
                crosstab_analysis(st.session_state.df, column_x, column_y, decimal_places)
        else:
            st.warning("データがアップロードされていません。データアップロードページでCSVファイルをアップロードしてください。")


if __name__ == "__main__":
    main()

