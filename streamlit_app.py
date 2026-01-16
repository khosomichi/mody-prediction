import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import base64
from pathlib import Path
import os

# 画像をBase64エンコードする関数
def load_image_as_base64(image_path):
    """PNG画像を読み込んでBase64エンコード"""
    try:
        # 相対パスを絶対パスに変換
        abs_path = Path(__file__).parent / image_path
        
        if not abs_path.exists():
            # フォールバック：直接パスで試す
            abs_path = Path(image_path)
        
        if not abs_path.exists():
            st.warning(f"画像ファイルが見つかりません: {image_path}")
            st.info(f"検索パス: {abs_path}")
            return ""
        
        with open(abs_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        st.warning(f"画像読み込みエラー: {image_path} - {e}")
        return ""

# 画像アイコンの読み込み（アプリ起動時に1回だけ）
@st.cache_data
def load_icons():
    icons = {
        'lipid': load_image_as_base64('icons/lipid.png'),
        'liver': load_image_as_base64('icons/liver.png'),
        'kidney': load_image_as_base64('icons/kidney.png')
    }
    return icons

# アイコンをグローバルで読み込む
icons = load_icons()

# ページ設定
st.set_page_config(
    page_title="MODY予測システム",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# モデルとデータの読み込み
@st.cache_resource
def load_models():
    try:
        model = joblib.load('mody_multiclass_model.pkl')
        imputer = joblib.load('imputer_ovr.pkl')
        scaler = joblib.load('scaler_multiclass.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return model, imputer, scaler, feature_names
    except FileNotFoundError as e:
        st.error(f"モデルファイルが見つかりません: {e}")
        st.info("まず機械学習モデルを訓練してください。")
        st.code("""
# モデル訓練スクリプトを実行:
python mody_model_training_save.py
        """)
        st.stop()

model, imputer, scaler, feature_names = load_models()

# MODYタイプの定義
mody_types = ['GCK/MODY2', 'HNF1A/MODY3', 'HNF1B/MODY5', 'HNF4A/MODY1']
mody_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

# セッションステートの初期化
if 'predicted' not in st.session_state:
    st.session_state.predicted = False
if 'input_data' not in st.session_state:
    st.session_state.input_data = {}

# 空欄を許可する入力関数
def optional_selectbox(label, options, format_func=None, key=None, default_value=None):
    """空欄を許可するselectbox"""
    options_with_none = [None] + options
    if format_func:
        format_func_with_none = lambda x: "未選択（自動補完）" if x is None else format_func(x)
    else:
        format_func_with_none = lambda x: "未選択（自動補完）" if x is None else str(x)
    
    default_index = 0
    if default_value is not None and default_value in options:
        default_index = options.index(default_value) + 1
    
    help_text = f"未選択の場合は自動で補完されます"
    
    return st.selectbox(label, options_with_none, 
                       format_func=format_func_with_none, 
                       key=key,
                       index=default_index,
                       help=help_text)

def optional_number_input(label, min_value=None, max_value=None, step=None, format=None, key=None, default_value=None):
    """空欄を許可するnumber_input"""
    help_text = f"未入力の場合は空欄のまま構いません（自動で補完されます）"
    
    if default_value is not None:
        value = st.number_input(
            label, 
            min_value=min_value, 
            max_value=max_value, 
            step=step, 
            format=format, 
            key=key, 
            value=default_value,
            help=help_text
        )
    else:
        value = st.number_input(
            label, 
            min_value=min_value, 
            max_value=max_value, 
            step=step, 
            format=format, 
            key=key,
            help=help_text,
            value=None,
            placeholder="未入力（自動補完）"
        )
    
    if value is not None and value <= 0:
        return None
    
    return value

def render_icon_section(icon_key, title):
    """アイコン付きセクションを描画する関数"""
    if icons[icon_key]:
        st.markdown(f"""
        <div style='display: flex; align-items: center; margin-bottom: 0.5rem;'>
            <img src='data:image/png;base64,{icons[icon_key]}' width='24' height='24' style='margin-right: 8px;'/>
            <span style='font-size: 1.1rem; font-weight: 600;'>{title}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"#### {title}")

def create_input_form(title="患者情報入力", button_text="🔮 予測実行", default_data=None):
    """入力フォームを作成する関数"""
    st.header(f"📝 {title}")
    
    if default_data is None:
        default_data = {}
    
    input_data = {}
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("#### 👤 基本情報")
        input_data['family_history'] = optional_selectbox(
            "家族歴",
            [0, 1],
            format_func=lambda x: "なし" if x == 0 else "あり",
            key="family_history",
            default_value=default_data.get('family_history')
        )
        input_data['gender'] = optional_selectbox(
            "性別",
            [0, 1],
            format_func=lambda x: "女性" if x == 0 else "男性",
            key="gender",
            default_value=default_data.get('gender')
        )
        input_data['diagnostic_age'] = optional_number_input(
            "診断時年齢 (歳)",
            min_value=0, max_value=100, step=1,
            key="diagnostic_age",
            default_value=default_data.get('diagnostic_age')
        )
        
        st.markdown("#### 📏 身体測定")
        input_data['height'] = optional_number_input(
            "身長 (cm)",
            min_value=100.0, max_value=250.0, step=0.1,
            key="height",
            default_value=default_data.get('height')
        )
        input_data['body weight'] = optional_number_input(
            "体重 (kg)",
            min_value=30.0, max_value=200.0, step=0.1,
            key="body_weight",
            default_value=default_data.get('body weight')
        )
        input_data['BMI'] = optional_number_input(
            "BMI",
            min_value=10.0, max_value=50.0, step=0.1,
            key="BMI",
            default_value=default_data.get('BMI')
        )
    
    with col2:
        st.markdown("#### 💉 糖尿病関連")
        input_data['years_to_insulin'] = optional_number_input(
            "インスリン導入までの年数",
            min_value=0.0, max_value=50.0, step=0.1,
            key="years_to_insulin",
            default_value=default_data.get('years_to_insulin')
        )
        input_data['HbA1c_initial'] = optional_number_input(
            "初回HbA1c (%)",
            min_value=4.0, max_value=15.0, step=0.1,
            key="HbA1c_initial",
            default_value=default_data.get('HbA1c_initial')
        )
        
        st.markdown("""
        <div style='color: transparent; pointer-events: none; user-select: none; height: 85px;'>
            <div style='font-size: 1rem; font-weight: 600; margin-bottom: 0.5rem; color: transparent;'>ダミー項目</div>
            <div style='height: 38px; border: 1px solid transparent; border-radius: 0.25rem; background-color: rgba(240, 242, 246, 0); margin-bottom: 1rem;'></div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 🩸 血糖・インスリン検査")
        input_data['FPG'] = optional_number_input(
            "空腹時血糖 (mg/dL)",
            min_value=50, max_value=500, step=1,
            key="FPG",
            default_value=default_data.get('FPG')
        )
        input_data['fIRI'] = optional_number_input(
            "空腹時インスリン (μU/mL)",
            min_value=0.0, max_value=100.0, step=0.1,
            key="fIRI",
            default_value=default_data.get('fIRI')
        )
        input_data['CPR_index'] = optional_number_input(
            "CPRインデックス",
            min_value=0.0, max_value=10.0, step=0.1,
            key="CPR_index",
            default_value=default_data.get('CPR_index')
        )
        input_data['ΔCPR'] = optional_number_input(
            "ΔCPR (ng/mL)",
            min_value=0.0, max_value=10.0, step=0.1,
            key="delta_CPR",
            default_value=default_data.get('ΔCPR')
        )
    
    with col3:
        st.markdown("##### 📊 HOMA指数")
        input_data['HOMA-B'] = optional_number_input(
            "HOMA-β (%)",
            min_value=0.0, max_value=500.0, step=1.0,
            key="HOMA_B",
            default_value=default_data.get('HOMA-B')
        )
        input_data['HOMA-R'] = optional_number_input(
            "HOMA-R",
            min_value=0.0, max_value=20.0, step=0.1,
            key="HOMA_R",
            default_value=default_data.get('HOMA-R')
        )
        input_data['I.I'] = optional_number_input(
            "インスリンインデックス",
            min_value=0.0, max_value=5.0, step=0.1,
            key="insulin_index",
            default_value=default_data.get('I.I')
        )
        
        render_icon_section('lipid', '脂質検査')
        
        input_data['T-chol'] = optional_number_input(
            "総コレステロール (mg/dL)",
            min_value=100, max_value=400, step=1,
            key="T_chol",
            default_value=default_data.get('T-chol')
        )
        input_data['HDL'] = optional_number_input(
            "HDLコレステロール (mg/dL)",
            min_value=20, max_value=150, step=1,
            key="HDL",
            default_value=default_data.get('HDL')
        )
        input_data['TG'] = optional_number_input(
            "中性脂肪 (mg/dL)",
            min_value=30, max_value=1000, step=1,
            key="TG",
            default_value=default_data.get('TG')
        )
    
    with col4:
        render_icon_section('liver', '肝機能検査')
        
        input_data['γGTP'] = optional_number_input(
            "γ-GTP (U/L)",
            min_value=5, max_value=500, step=1,
            key="gamma_GTP",
            default_value=default_data.get('γGTP')
        )
        input_data['GOT'] = optional_number_input(
            "AST/GOT (U/L)",
            min_value=10, max_value=200, step=1,
            key="GOT",
            default_value=default_data.get('GOT')
        )
        input_data['GPT'] = optional_number_input(
            "ALT/GPT (U/L)",
            min_value=10, max_value=200, step=1,
            key="GPT",
            default_value=default_data.get('GPT')
        )
        
        render_icon_section('kidney', '腎機能検査')
        input_data['Cre'] = optional_number_input(
            "クレアチニン (mg/dL)",
            min_value=0.3, max_value=5.0, step=0.1,
            key="Cre",
            default_value=default_data.get('Cre')
        )
        input_data['BUN'] = optional_number_input(
            "尿素窒素 (mg/dL)",
            min_value=5, max_value=100, step=1,
            key="BUN",
            default_value=default_data.get('BUN')
        )
        input_data['UA'] = optional_number_input(
            "尿酸 (mg/dL)",
            min_value=1.0, max_value=15.0, step=0.1,
            key="UA",
            default_value=default_data.get('UA')
        )
    
    st.markdown("---")
    
    col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
    with col_btn2:
        predict_button = st.button(button_text, type="primary", use_container_width=True)
    
    return input_data, predict_button

def perform_prediction(input_data):
    """予測を実行する関数"""
    # 入力データの準備
    input_df = pd.DataFrame([input_data])
    
    # 列の順序をfeature_namesに合わせる
    input_df = input_df[feature_names]
    
    # 入力状況の確認
    missing_count = input_df.isnull().sum().sum()
    total_features = len(feature_names)
    available_features = total_features - missing_count
    
    # 入力状況の詳細情報を表示
    with st.expander("🔍 空欄項目の処理について"):
        st.markdown(f"**入力された特徴量: {available_features}/{total_features}** ({available_features/total_features:.1%})")
        
        if missing_count > 0:
            missing_features = input_df.columns[input_df.isnull().any()].tolist()
            st.markdown("**未入力の項目:**")
            cols = st.columns(3)
            for i, feature in enumerate(missing_features):
                with cols[i % 3]:
                    st.write(f"• {feature}")
            
            st.markdown("**🔧 処理方法:**")
            st.info("""
            未入力項目は**訓練データの平均値**で自動補完されます：
            
            - **数値データ**: 各検査値の平均値を使用
            - **カテゴリデータ**: 最頻値を使用
            - この処理により、部分的なデータでも予測が可能になります
            
            ⚠️ **注意**: 入力項目が多いほど予測精度が向上します
            """)
        else:
            st.success("✅ 全ての項目が入力されています！最高の予測精度が期待できます。")
    
    try:
        # 前処理（欠損値補完＋正規化）
        input_imputed = imputer.transform(input_df)
        input_scaled = scaler.transform(input_imputed)
        
        # 予測
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        
        return {
            'prediction': prediction,
            'probabilities': probabilities,
            'available_features': available_features,
            'total_features': total_features,
            'missing_count': missing_count
        }
        
    except Exception as e:
        st.error(f"予測エラー: {e}")
        return None

def display_results(result_data):
    """予測結果を表示する関数"""
    st.header("📊 予測結果")
    
    prediction = result_data['prediction']
    probabilities = result_data['probabilities']
    available_features = result_data['available_features']
    total_features = result_data['total_features']
    
    # 予測結果のハイライト
    predicted_type = mody_types[prediction]
    confidence = np.max(probabilities)
    
    # 結果を3列で表示
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success(f"**最も可能性が高いMODYタイプ**\n\n# {predicted_type}")
    
    with col2:
        st.info(f"**信頼度**\n\n# {confidence:.1%}")
    
    with col3:
        st.info(f"**入力された特徴量**\n\n# {available_features}/{total_features}")
    
    # 確率の棒グラフ
    fig = go.Figure(data=[
        go.Bar(
            x=mody_types,
            y=probabilities,
            marker_color=mody_colors,
            text=[f"{p:.1%}" for p in probabilities],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="各MODYタイプの予測確率",
        xaxis_title="MODYタイプ",
        yaxis_title="確率",
        showlegend=False,
        height=400,
        xaxis_tickangle=-45
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 詳細な確率表示
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("詳細な予測確率")
        prob_df = pd.DataFrame({
            'MODYタイプ': mody_types,
            '確率': [f"{p:.3f}" for p in probabilities],
            '百分率': [f"{p:.1%}" for p in probabilities]
        })
        
        # 結果をソートして表示
        prob_df['確率_数値'] = probabilities
        prob_df = prob_df.sort_values('確率_数値', ascending=False)
        prob_df = prob_df.drop('確率_数値', axis=1)
        
        st.dataframe(prob_df, use_container_width=True, hide_index=True)
    
    with col2:
        # 信頼性の指標
        st.subheader("🎯 予測の信頼性")
        
        completeness_factor = available_features / total_features
        
        if available_features >= 20 and confidence >= 0.7:
            st.success("🟢 **高い信頼性**\n\nこの予測は信頼できます")
        elif available_features >= 15 and confidence >= 0.5:
            st.warning("🟡 **中程度の信頼性**\n\n追加検査を検討してください")
        elif available_features >= 10:
            st.warning("🟠 **限定的な信頼性**\n\n重要な検査データが不足しています")
        else:
            st.error("🔴 **低い信頼性**\n\nより詳細な検査が必要です")

# メインアプリケーション
def main():
    # タイトル
    st.title("🧬 MODY遺伝子型予測システム")
    
    # 予測前の状態（初期画面）
    if not st.session_state.predicted:
        st.markdown("""
        **このシステムは、患者の臨床情報から最も可能性の高いMODYタイプを予測します。**
        - 機械学習モデル: Random Forest（多クラス分類）
        - 対象: *GCK*/MODY2, *HNF1A*/MODY3, *HNF1B*/MODY5, *HNF4A*/MODY1
        - ⭐ **空欄があっても予測可能**（未入力項目は訓練データの平均値で自動補完）
        
        📝 **使い方**: 利用可能な検査値を入力して「予測実行」ボタンを押してください。空欄のままでも構いません。
        """)
        st.markdown("---")
        
        # 入力フォーム（全面表示）
        input_data, predict_button = create_input_form()
        
        # 予測ボタンが押された場合
        if predict_button:
            # セッションステートに入力データを保存
            st.session_state.input_data = input_data
            
            # 予測実行
            result = perform_prediction(input_data)
            
            if result:
                st.session_state.prediction_result = result
                st.session_state.predicted = True
                st.rerun()
    
    # 予測後の状態（結果表示画面）
    else:
        # 結果表示
        display_results(st.session_state.prediction_result)
        
        st.markdown("---")
        
        # 再予測用フォーム（入力済みデータで初期化）
        st.markdown("## 🔄 データ修正・再予測")
        st.markdown("必要に応じて値を修正して、再度予測を実行できます。")
        
        input_data, predict_button = create_input_form(
            title="データ修正",
            button_text="🔄 再予測実行",
            default_data=st.session_state.input_data
        )
        
        # 再予測ボタンが押された場合
        if predict_button:
            # 新しい入力データで予測
            result = perform_prediction(input_data)
            
            if result:
                st.session_state.input_data = input_data
                st.session_state.prediction_result = result
                st.rerun()
        
        # 新しい患者の入力に戻るボタン
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("🆕 新しい患者の予測", use_container_width=True):
                st.session_state.predicted = False
                st.session_state.input_data = {}
                if 'prediction_result' in st.session_state:
                    del st.session_state.prediction_result
                st.rerun()

if __name__ == "__main__":
    main()

# フッター
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>MODY遺伝子型予測システム | 機械学習による臨床支援ツール</p>
    <p><small>※欠損値対応機能付き・研究目的システム</small></p>
</div>
""", unsafe_allow_html=True)
