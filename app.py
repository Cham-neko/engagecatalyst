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
from wordcloud import WordCloud
from janome.tokenizer import Tokenizer
from collections import Counter  # ワードクラウド作成に必要
from itertools import combinations
import streamlit.components.v1 as components

# Google Analytics tracking code
ga_code = """
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-G2KBPE365L"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-G2KBPE365L');
</script>
"""

# StreamlitにGoogle Analyticsトラッキングコードを埋め込む
def add_ga_tracking():
    components.html(ga_code, height=0)

# ページ設定
st.set_page_config(layout="wide", page_title="従業員サーベイデータ分析")

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
            <p>データの解釈に困ったり、調査設計や分析の相談は<a href="https://whitebank.site">こちら</a>からどうぞ！</p>
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
            # エンコーディング自動検出
            raw_data = uploaded_file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            uploaded_file.seek(0)

            # 自動検出したエンコーディングでファイルを読み込む
            df = pd.read_csv(uploaded_file, encoding=encoding)
            if df.empty or df.columns.size == 0:
                raise ValueError("ファイルにカラムが存在しません。フォーマットを確認してください。")
            return df
        except UnicodeDecodeError:
            # UTF-8で失敗した場合は、別のエンコーディングを試す
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='utf-8')
                if df.empty or df.columns.size == 0:
                    raise ValueError("ファイルにカラムが存在しません。フォーマットを確認してください。")
                return df
            except UnicodeDecodeError:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='cp932')
                    if df.empty or df.columns.size == 0:
                        raise ValueError("ファイルにカラムが存在しません。フォーマットを確認してください。")
                    return df
                except Exception as e:
                    raise e
        except Exception as e:
            raise e
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
    n_items = st.slider("下に表示する項目数", min_value=1, max_value=len(numeric_columns)-1, value=5)
    
    correlations = corr_matrix[target_var].abs().sort_values(ascending=False)
    top_correlations = correlations[correlations.index != target_var][:n_items]
    
    fig = px.bar(x=top_correlations.index, y=top_correlations.values,
                 labels={'x': '変数', 'y': f'{target_var}との相関係数'},
                 title=f"{target_var}との上位{n_items}個の高相関項目（相関係数が高い項目から順に並びます）")
    st.plotly_chart(fig)

# 日本語テキストのストップワードリストを作成（必要に応じて追加可能）
stopwords = set([
    "の", "に", "を", "は", "が", "で", "と", "た", "も", "て", "だ", "な", "い", "し", "れる", "する", "こと", "これ", "それ", "あれ"
])

# 日本語テキストのストップワードリストを作成（不要な単語を追加）
stopwords = set([
        "の", "に", "を", "は", "が", "で", "と", "た", "も", "て", "だ", "な", 
        "い", "し", "れる", "する", "こと", "これ", "それ", "あれ", "いる", "ある", "よう", "いう", "ため", "なる", "おる", "られる", "ない", "やる", "感じる", "思う", "できる", 
    ])

def create_wordcloud_with_cooccurrence(text_column):
    # テキストをすべて文字列に変換し、連結して1つの文字列にする
    text = " ".join(text_column.dropna().astype(str))  # 数値を文字列に変換
    
    # 日本語の形態素解析を行い、名詞・動詞・形容詞を抽出
    t = Tokenizer()
    words = [token.base_form for token in t.tokenize(text) 
             if token.part_of_speech.startswith(('名詞', '動詞', '形容詞'))  # 名詞、動詞、形容詞を含める
             and len(token.base_form) > 1  # 単語の長さが1文字以上
             and token.base_form not in stopwords]  # ストップワードに含まれていない

    # 特定のキーワードに関連する単語ペアを強調する
    focus_keywords = ["組織", "風土", "文化", "制度"]  # 注目するキーワード
    cooccurrence_pairs = []
    
    # 共起関係を計算
    window_size = 3  # 共起の範囲
    for i in range(len(words) - window_size + 1):
        window = words[i:i + window_size]
        cooccurrence_pairs.extend(combinations(window, 2))  # ペアごとの組み合わせを生成
    
    # 頻出単語ペアのカウント
    cooccurrence_freq = Counter(cooccurrence_pairs)
    
    # 重要な単語ペアを一つの「単語」として扱う（特定のキーワードが含まれるものを優先）
    modified_words = words.copy()
    for pair, freq in cooccurrence_freq.items():
        if freq > 2:  # 出現頻度が一定以上の場合
            if pair[0] in focus_keywords or pair[1] in focus_keywords:
                # 単語ペアを結合して一つの「単語」として扱う
                modified_words.append(f"{pair[0]}_{pair[1]}")

    # 単語と単語ペアの頻出カウント
    word_freq = Counter(modified_words)

    # ワードクラウドの作成
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white', 
        font_path='NotoSansJP-SemiBold.ttf'  # 日本語フォントのパスを指定
    ).generate_from_frequencies(word_freq)

    # グラフの表示
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    
    st.pyplot(fig)



# サンプルでの感情分析の適用（感情分析を視覚的に取り入れる場合の例）
def sentiment_analysis_for_wordcloud(text_column):
    # テキストをすべて文字列に変換し、連結して1つの文字列にする
    text = " ".join(text_column.dropna().astype(str))
    
    # 日本語の形態素解析と同時に、簡易的な感情分析（例: ポジティブ・ネガティブな単語のカウント）を導入
    positive_words = ["改善", "効率", "向上", "成功", "成長"]
    negative_words = ["課題", "問題", "障害", "低下", "減少"]

    # 感情ラベルの付与
    positive_count = 0
    negative_count = 0

    # 形態素解析を行い、ポジティブ・ネガティブな単語のカウント
    t = Tokenizer()
    for token in t.tokenize(text):
        base_form = token.base_form
        if base_form in positive_words:
            positive_count += 1
        elif base_form in negative_words:
            negative_count += 1
    
    # 感情スコアを表示
    st.write(f"ポジティブ単語数: {positive_count}")
    st.write(f"ネガティブ単語数: {negative_count}")

    # 結果に基づいてワードクラウドを作成する（ポジティブ・ネガティブに応じて色を変えるなど）
    # ここでは簡易な例として、ポジティブが多い場合は青、ネガティブが多い場合は赤のワードクラウドを表示
    cloud_color = 'blue' if positive_count > negative_count else 'red'

    # ワードクラウドの作成
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap=cloud_color, font_path='/Library/Fonts/Arial Unicode.ttf').generate(text)
    
    # グラフの表示
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    
    st.pyplot(fig)



# マルチコリニアリティのチェック
def check_multicollinearity(X):
    vif_data = pd.DataFrame()
    vif_data["特徴量"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

# 回帰分析の実行
def regression_analysis(df, numeric_columns):
    # 目的変数と説明変数の選択
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

    # 欠損値やデータの空チェック
    if X.empty or y.empty:
        st.error("有効なデータがありません。データに欠損がないか確認してください。")
        return

    if len(X) < 2 or len(y) < 2:
        st.error("データが少なすぎて分析ができません。")
        return
    
    try:
        # マルチコリニアリティチェック
        vif_result = check_multicollinearity(X)
        if (vif_result['VIF'] > 5).any():
            st.warning("VIFが5を超える変数があります。多重共線性の可能性があるため、変数の再選定をお勧めします。")

        st.write("マルチコリニアリティチェック結果: VIFが全て５未満であれば多重共線性（似過ぎ問題）がないと言えます")
        st.write(vif_result)

        # 回帰分析を実行
        model = LinearRegression()
        model.fit(X, y)

        # 重回帰分析の結果表示
        coefficients = pd.DataFrame({"特徴量": features, "係数": model.coef_})
        fig = px.bar(coefficients, x="特徴量", y="係数", title="重回帰分析結果: 特徴量の係数（目的変数に対して説明変数がどの程度因果関係が強いかを示します）")
        st.plotly_chart(fig)

        # R^2を強調して表示
        r_squared = model.score(X, y)
        st.markdown(f"<p>切片: {model.intercept_:.4f}</p>", unsafe_allow_html=True)
        st.markdown(f"<p>決定係数 (<span style='color:red; font-weight:bold;'>R²: {r_squared:.4f}</span>) 0.5以上で中程度以上の因果関係があると言えます</p>", unsafe_allow_html=True)

    except ValueError as e:
        st.error(f"エラーが発生しました。変数を追加したりデータに問題がないか確認してください。")
    except Exception as e:
        st.error(f"予期せぬエラーが発生しました: {str(e)}")



# 変数の加工
# 変数の加工における再分類機能の追加
def variable_processing(df):
    st.subheader("変数の加工")
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_column = st.selectbox("再分類する変数を選択してください", numeric_columns)

    if selected_column:
        unique_values = sorted(df[selected_column].dropna().unique())  # NaNは無視

        # ユニークな値をリストで表示し、何が入っているかをわかりやすく表示
        st.write("**選択肢の値の一覧**")
        unique_table = pd.DataFrame({"ユニークな値": unique_values})  # インデックスをリセットして表示しない
        st.dataframe(unique_table, hide_index=True)

        # ユニークな値がすべて整数かどうかを確認し、整数でなければステップを浮動小数点に設定
        if all(float(val).is_integer() for val in unique_values):
            min_val = int(min(unique_values))
            max_val = int(max(unique_values))
            step = 1
        else:
            min_val = float(min(unique_values))
            max_val = float(max(unique_values))
            step = (max_val - min_val) / 100  # スライダーのステップを調整

        num_bins = st.number_input("分類するグループ（群）の数を指定してください", min_value=2, max_value=len(unique_values), value=2)
        
        # 範囲を指定するためのリストを作成（例：1-3, 4, 5）
        bin_labels = []
        for i in range(num_bins):
            bin_range = st.slider(f"グループ {i+1} の範囲（どの選択肢の値をまとめるか）を指定してください", min_value=min_val, max_value=max_val, value=(min_val, max_val), step=step, format="%f")
            bin_labels.append(bin_range)

        # グループ名の設定
        group_labels = [f"Group{i+1}" for i in range(num_bins)]

        # 加工後のデータの表形式でのプレビュー
        if st.button("再分類を実行"):
            bins = [min_val] + [r[1] for r in bin_labels]  # スライダーの上限値を使って区切る
            try:
                df[f"{selected_column}_reclassified"] = pd.cut(df[selected_column], bins=bins, labels=group_labels, include_lowest=True)
                st.success(f"{selected_column}が再分類されました。")

                # 再分類後の変数を元の変数と一緒に表示
                reclassified_data = pd.DataFrame({
                    "加工前の変数": df[selected_column],
                    "加工後の変数": df[f"{selected_column}_reclassified"]
                }).dropna()

            except ValueError as e:
                st.error(f"再分類に失敗しました: {e}")

        # 分析データに追加するボタン
        if 'new_column_name' not in st.session_state:
            st.session_state.new_column_name = ""

        new_column_name = st.text_input("分類した新しい変数名を入力してください", value=st.session_state.new_column_name)
        if st.button("分析データに追加する"):
            if new_column_name:
                df[new_column_name] = df[f"{selected_column}_reclassified"]
                st.session_state.df = df
                st.session_state.new_column_name = new_column_name  # 入力値をセッションに保存
                st.success("分析データに追加しました。分析の変数選択の際、末尾に表示されます。")


# クロス集計の実行
@st.cache_data
def crosstab_analysis(df, column_x, column_y, decimal_places):
    # 再分類された変数が存在するかを確認
    if f"{column_y}_reclassified" in df.columns:
        # 再分類された変数を使用してクロス集計を実行
        column_y = f"{column_y}_reclassified"
        st.write(f"変数 {column_y} が再分類されました。")
    if f"{column_x}_reclassified" in df.columns:
        # 再分類された変数を使用してクロス集計を実行
        column_x = f"{column_x}_reclassified"
        st.write(f"変数 {column_x} が再分類されました。")
    
    # クロス集計の計算（目的変数を行、説明変数を列として設定）
    crosstab = pd.crosstab(df[column_y], df[column_x])

    # 行名がカテゴリの場合、そのカテゴリ情報を保持
    if pd.api.types.is_categorical_dtype(df[column_y]):
        crosstab.index = df[column_y].cat.categories

    # 説明変数テキストの位置調整
    col1, col2 = st.columns([1, 4])

    # 説明変数の位置を中央に調整
    with col1:
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write(f"説明変数（表側）: {column_y}")

    with col2:
        st.write(f"クロス集計結果【数表】　目的変数（表頭: {column_x}）")
        crosstab_with_labels = crosstab.copy()

        # 行と列のラベル名を削除
        crosstab_with_labels.index.name = None
        crosstab_with_labels.columns.name = None

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
        st.write(f"説明変数（表側）: {column_y}")

    with col2:
        st.write(f"クロス集計結果【%表】　目的変数（表頭: {column_x}）")
        crosstab_percent_with_labels = crosstab_percent.copy()

        # 行と列のラベル名を削除
        # crosstab_percent_with_labels.index.name = None
        # crosstab_percent_with_labels.columns.name = None

        # 選択された小数点位数で丸めて表示
        st.dataframe(crosstab_percent_with_labels.round(decimal_places))


def main():
    st.sidebar.title("Menu")

    menu_items = ["データアップロード", "記述統計", "変数の加工", "クロス集計", "相関分析", "回帰分析", "ワードクラウド作成"]  # メニューに「変数の加工」を追加

    page = st.sidebar.selectbox("メニューを選択してください", menu_items)

    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'uploaded_file_name' not in st.session_state:
        st.session_state.uploaded_file_name = None

    if page == "データアップロード":

        # Google Analyticsのトラッキングコードを追加
        add_ga_tracking()
        # サービス名をページ上部に表示
        st.title("Smart Matrics")
        st.markdown("<h3 style='text-align: center;'>従業員サーベイなどアンケートデータの分析ツールです。</h3>", unsafe_allow_html=True)

        # 説明文を改行して表示
        st.markdown("<br><br>CSVファイルをアップロードしてください。<br>個人情報や機密情報は含めないでください。<br>日本語テキストを含むデータでワードクラウドを作成する場合はUTF-8形式のCSVで保存されたものを使ってください。<br>", unsafe_allow_html=True)

        # すでにアップロードされたファイルがある場合、その名前を表示しつつ、新しいファイルのアップロードも可能にする
        if st.session_state.df is not None:
            st.write(f"現在アップロードされているファイル: {st.session_state.uploaded_file_name}")
            uploaded_file = st.file_uploader("新しいCSVファイルをアップロード", type="csv")
            if uploaded_file is not None:
                st.session_state.df = load_csv(uploaded_file)
                st.session_state.uploaded_file_name = uploaded_file.name
                st.success(f"新しいファイルがアップロードされました: {uploaded_file.name}")
        else:
            uploaded_file = st.file_uploader("CSVファイルをアップロード", type="csv")
            if uploaded_file is not None:
                st.session_state.df = load_csv(uploaded_file)
                st.session_state.uploaded_file_name = uploaded_file.name
                st.success(f"ファイルが正常にアップロードされました: {uploaded_file.name}")

    # 他のページに移動してもデータを保持する処理
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
        st.title("変数の加工「回答の値によって群分けする」")
        if st.session_state.df is not None:
            variable_processing(st.session_state.df)
        else:
            st.warning("データがアップロードされていません。データアップロードページでCSVファイルをアップロードしてください。")      

    elif page == "クロス集計":
        st.title("クロス集計「群ごとの傾向の違いを見る」")
        if st.session_state.df is not None:
            column_y = st.selectbox("クロス集計の表側（説明変数）を選択してください", st.session_state.df.columns)
            column_x = st.selectbox("クロス集計の表頭（目的変数）を選択してください", st.session_state.df.columns)
            decimal_places = st.selectbox("%表の小数点の表示桁数を選択してください", [1, 2, 3, 4], index=0)

            with st.spinner('分析中...'):
                time.sleep(2)
                crosstab_analysis(st.session_state.df, column_x, column_y, decimal_places)
        else:
            st.warning("データがアップロードされていません。データアップロードページでCSVファイルをアップロードしてください。")

    # ワードクラウド作成
    elif page == "ワードクラウド作成":
        st.title("ワードクラウド作成")
        if st.session_state.df is not None:
            text_column = st.selectbox("テキストデータを含むカラムを選んでください", st.session_state.df.columns)
            if st.button("ワードクラウドを作成"):
                with st.spinner('ワードクラウドを作成中...'):
                    create_wordcloud_with_cooccurrence(st.session_state.df[text_column])
        else:
            st.warning("データがアップロードされていません。データアップロードページでCSVファイルをアップロードしてください。")

if __name__ == "__main__":
    main()