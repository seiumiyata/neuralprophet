import streamlit as st
import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ページ設定
st.set_page_config(
    page_title="需要予測アプリ",
    layout="wide"
)

# サイドバーの設定
st.sidebar.title("モデル設定")

@st.cache_data
def load_data(uploaded_file):
    """データの読み込みと初期処理"""
    try:
        df = pd.read_excel(uploaded_file)
        if len(df.columns) >= 2:
            df = df.iloc[:, :2]
            df.columns = ['ds', 'y']
            return df, None
        else:
            return None, "データには少なくとも2列（日付と数量）が必要です"
    except Exception as e:
        return None, f"データ読み込みエラー: {str(e)}"

def validate_data(df):
    """データの検証と前処理"""
    try:
        # 日付型への変換
        df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
        if df['ds'].isnull().any():
            return None, "無効な日付が含まれています"

        # 重要：日次データへのリサンプリング処理を追加
        df.set_index('ds', inplace=True)
        df = df.resample('D').mean()
        df['y'] = df['y'].interpolate(method='linear')
        df.reset_index(inplace=True)

        # 数値型への変換と欠損値の確認
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        null_count = df['y'].isnull().sum()
        if null_count > 0:
            st.warning(f"{null_count}件の欠損値が見つかりました。線形補間で補完します。")
            df['y'] = df['y'].interpolate(method='linear')

        # 負の値の処理
        negative_count = (df['y'] < 0).sum()
        if negative_count > 0:
            st.warning(f"{negative_count}件の負の値を0に置換します。")
            df['y'] = df['y'].clip(lower=0)

        # 重複日付の処理
        duplicate_count = df.duplicated('ds').sum()
        if duplicate_count > 0:
            st.warning(f"{duplicate_count}件の重複日付があります。平均値に集約します。")
            df = df.groupby('ds')['y'].mean().reset_index()

        # 日付でソート
        df = df.sort_values('ds')
        
        return df, None
    except Exception as e:
        return None, f"データ検証エラー: {str(e)}"

def create_model_params():
    """モデルパラメータの設定"""
    return {
        'growth': st.sidebar.selectbox(
            'トレンドモデル',
            ['linear', 'off'],
            format_func=lambda x: 'トレンドあり' if x == 'linear' else 'トレンドなし',
            help='需要のトレンドを考慮するかどうか'
        ),
        'seasonality_mode': st.sidebar.selectbox(
            '季節性の計算方法',
            ['multiplicative', 'additive'],
            format_func=lambda x: '乗法的' if x == 'multiplicative' else '加法的',
            help='季節変動の計算方法を選択'
        ),
        'yearly_seasonality': st.sidebar.selectbox(
            '年間季節性',
            [True, False],
            format_func=lambda x: '考慮する' if x else '考慮しない',
            help='年間の需要パターンを考慮'
        ),
        'weekly_seasonality': st.sidebar.selectbox(
            '週間季節性',
            [True, False],
            format_func=lambda x: '考慮する' if x else '考慮しない',
            help='週間の需要パターンを考慮'
        ),
        'daily_seasonality': st.sidebar.selectbox(
            '日次季節性',
            [True, False],
            format_func=lambda x: '考慮する' if x else '考慮しない',
            help='日次の需要パターンを考慮'
        ),
        'epochs': st.sidebar.slider(
            '学習回数',
            min_value=100,
            max_value=1000,
            value=300,
            step=50,
            help='モデルの学習回数（多いほど精度が上がりますが、時間がかかります）'
        )
    }
def main():
    st.title("需要予測アプリケーション")

    # 予測期間の選択（サイドバーに配置）
    forecast_periods = st.sidebar.multiselect(
        '予測期間の選択',
        [30, 60, 90, 180, 360],
        default=[30],
        format_func=lambda x: f'{x}日後まで',
        help='予測したい期間を選択してください'
    )

    # モデルパラメータの取得
    model_params = create_model_params()
    
    # ファイルアップロード
    uploaded_file = st.file_uploader(
        "Excelファイルをアップロード（必須列：日付、数量）",
        type=['xlsx']
    )
    
    if uploaded_file is not None:
        # データ読み込み
        df, error = load_data(uploaded_file)
        if error:
            st.error(error)
            return

        # データの検証と前処理
        df, error = validate_data(df)
        if error:
            st.error(error)
            return

        # データ概要の表示
        st.write("### データの概要")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"データ期間: {df['ds'].min().strftime('%Y年%m月%d日')} から {df['ds'].max().strftime('%Y年%m月%d日')}")
            st.write(f"データ件数: {len(df):,} 件")
        
        # 基本統計量の表示
        with col2:
            stats = df['y'].describe().round(2)
            st.write("基本統計量:")
            st.write(f"平均値: {stats['mean']:,.2f}")
            st.write(f"標準偏差: {stats['std']:,.2f}")
            st.write(f"最小値: {stats['min']:,.2f}")
            st.write(f"最大値: {stats['max']:,.2f}")

        if st.button('予測開始'):
            if not forecast_periods:
                st.error("予測期間を選択してください。")
                return

            # プログレスバーの初期化
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                # モデルの作成と学習
                status_text.text('モデルを学習中...')
                progress_bar.progress(25)
                
                model = NeuralProphet(**model_params)
                model.fit(df)
                
                progress_bar.progress(50)

                # 各期間での予測
                for i, period in enumerate(forecast_periods):
                    # プログレスバーの更新
                    progress = int(50 + ((i + 1) / len(forecast_periods)) * 50)
                    progress_bar.progress(progress)
                    status_text.text(f'{period}日間の予測を計算中...')

                    # 予測の実行
                    future = model.make_future_dataframe(df, periods=period)
                    forecast = model.predict(future)

                    # グラフの作成
                    fig = go.Figure()

                    # 実績値
                    fig.add_trace(go.Scatter(
                        x=df['ds'],
                        y=df['y'],
                        name='実績値',
                        line=dict(color='blue')
                    ))

                    # 予測値
                    fig.add_trace(go.Scatter(
                        x=forecast['ds'],
                        y=forecast['yhat1'],
                        name='予測値',
                        line=dict(color='red')
                    ))

                    # グラフのレイアウト設定
                    fig.update_layout(
                        title=f'{period}日間の需要予測',
                        xaxis_title='日付',
                        yaxis_title='数量',
                        hovermode='x unified',
                        height=500
                    )

                    # グラフの表示
                    st.plotly_chart(fig, use_container_width=True)

                    # 予測結果のExcel出力準備
                    output_df = pd.DataFrame({
                        '日付': forecast['ds'].dt.strftime('%Y年%m月%d日'),
                        '予測値': forecast['yhat1'].round(2)
                    })

                    # Excelファイルとして保存
                    output_file = f'予測結果_{period}日.xlsx'
                    output_df.to_excel(output_file, index=False)

                    # ダウンロードボタン
                    with open(output_file, 'rb') as f:
                        st.download_button(
                            label=f"{period}日予測結果をダウンロード",
                            data=f,
                            file_name=output_file,
                            mime='application/vnd.ms-excel'
                        )

                # 完了表示
                progress_bar.progress(100)
                status_text.text('予測完了！')
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"予測計算中にエラーが発生しました: {str(e)}")

if __name__ == '__main__':
    main()
