import streamlit as st
import numpy as np, pandas as pd, joblib, io
from scipy.io import loadmat
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ===== 페이지 설정 =====
st.set_page_config(page_title="베어링 고장 진단 시스템", page_icon="🔧", layout="wide")

# ===== 한글 폰트 =====
try:
    fm.fontManager.addfont('NanumGothic.ttf')
    plt.rcParams['font.family'] = 'NanumGothic'
except Exception:
    pass
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'

# ===== 네온 다크 테마 CSS =====
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1a2e 100%);
    }
    /* 메인 타이틀 */
    .neon-title {
        font-size: 2.6rem; font-weight: 800;
        background: linear-gradient(90deg, #00f0ff, #00ff9d);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(0,240,255,0.3);
        margin-bottom: 0.2rem;
    }
    .subtitle { color: #8b92b8; font-size: 1rem; margin-bottom: 1.5rem; }
    /* 카드 */
    .metric-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(0,240,255,0.2);
        border-radius: 16px; padding: 1.2rem 1.5rem;
        box-shadow: 0 0 20px rgba(0,240,255,0.08);
    }
    .card-label { color: #8b92b8; font-size: 0.85rem; letter-spacing: 1px; text-transform: uppercase; }
    .card-value { font-size: 1.8rem; font-weight: 700; color: #ffffff; margin-top: 0.3rem; }
    .result-normal { color: #00ff9d; text-shadow: 0 0 20px rgba(0,255,157,0.5); }
    .result-fault  { color: #ff3860; text-shadow: 0 0 20px rgba(255,56,96,0.5); }
    /* 상태 배너 */
    .status-banner {
        border-radius: 14px; padding: 1.2rem 1.8rem; margin: 1rem 0;
        font-size: 1.3rem; font-weight: 700; text-align: center;
    }
    .banner-normal {
        background: rgba(0,255,157,0.1); border: 1px solid #00ff9d; color: #00ff9d;
        box-shadow: 0 0 25px rgba(0,255,157,0.2);
    }
    .banner-fault {
        background: rgba(255,56,96,0.1); border: 1px solid #ff3860; color: #ff3860;
        box-shadow: 0 0 25px rgba(255,56,96,0.2);
    }
    section[data-testid="stSidebar"] { background: #0a0e27; border-right: 1px solid rgba(0,240,255,0.15); }
    .stProgress > div > div > div { background: linear-gradient(90deg, #00f0ff, #00ff9d); }
    h2, h3 { color: #e0e6ff !important; }
</style>
""", unsafe_allow_html=True)

# ===== 상수 =====
FS, WIN = 12000, 2048
FAULT_KR = {"N": "정상", "IR": "내륜 결함", "OR": "외륜 결함", "B": "볼 결함"}
NEON = {"N": "#00ff9d", "IR": "#00f0ff", "OR": "#ffb800", "B": "#ff3860"}
feat_cols = ['mean','std','rms','peak','p2p','skew','kurt','crest','shape',
             'impulse','clearance','spec_centroid','spec_rms',
             'band0','band1','band2','band3','band4']

@st.cache_resource
def load_model():
    return joblib.load('cwru_rf_model.pkl')
model = load_model()

def extract_features(x):
    x = np.asarray(x, np.float64); ax = np.abs(x)
    rms, peak, ma = np.sqrt(np.mean(x**2)), ax.max(), ax.mean()
    f = {"mean":x.mean(),"std":x.std(),"rms":rms,"peak":peak,"p2p":x.max()-x.min(),
         "skew":skew(x),"kurt":kurtosis(x),
         "crest":peak/rms if rms>0 else 0,"shape":rms/ma if ma>0 else 0,
         "impulse":peak/ma if ma>0 else 0,
         "clearance":peak/(np.mean(np.sqrt(ax))**2+1e-12)}
    xf = x-x.mean(); sp = np.abs(np.fft.rfft(xf)); fr = np.fft.rfftfreq(len(xf),1/FS)
    p = sp**2; ps = p.sum()+1e-12
    f["spec_centroid"]=(fr*p).sum()/ps; f["spec_rms"]=np.sqrt(np.mean(p))
    edges = np.linspace(0,FS/2,6)
    for i in range(5):
        msk=(fr>=edges[i])&(fr<edges[i+1]); f[f"band{i}"]=p[msk].sum()/ps
    return f

def predict_signal(sig):
    sig = np.asarray(sig).ravel()
    n_win = len(sig)//WIN
    if n_win == 0: return None, None, None
    rows = [extract_features(sig[i*WIN:(i+1)*WIN]) for i in range(n_win)]
    X = pd.DataFrame(rows)[feat_cols]
    proba = model.predict_proba(X).mean(axis=0)
    classes = model.classes_
    main = classes[proba.argmax()]
    return main, {classes[i]: proba[i] for i in range(len(classes))}, X

# ===== 게이지(도넛) 차트 =====
def gauge_chart(value, color):
    fig, ax = plt.subplots(figsize=(4,4), subplot_kw={'aspect':'equal'})
    # 배경 링
    ax.pie([1], colors=['#1a1f3a'], radius=1.0,
           wedgeprops={'width':0.28, 'edgecolor':'none'})
    # 값 링
    ax.pie([value, 1-value], colors=[color, '#0a0e27'], radius=1.0,
           startangle=90, counterclock=False,
           wedgeprops={'width':0.28, 'edgecolor':'none'})
    ax.text(0, 0.05, f"{value*100:.1f}%", ha='center', va='center',
            fontsize=30, fontweight='bold', color=color)
    ax.text(0, -0.22, "신뢰도", ha='center', va='center', fontsize=13, color='#8b92b8')
    return fig

# ===== 사이드바 =====
with st.sidebar:
    st.markdown("### ⚙️ SYSTEM INFO")
    st.markdown("""
    <div style='color:#8b92b8; font-size:0.9rem; line-height:1.8'>
    회전기계 베어링의 진동 신호를<br>분석해 고장 유형을 진단합니다.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style='color:#e0e6ff'>
    <b>진단 유형</b><br>
    <span style='color:#00ff9d'>●</span> 정상<br>
    <span style='color:#00f0ff'>●</span> 내륜 결함 (IR)<br>
    <span style='color:#ffb800'>●</span> 외륜 결함 (OR)<br>
    <span style='color:#ff3860'>●</span> 볼 결함 (B)<br>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style='color:#8b92b8; font-size:0.85rem; line-height:1.9'>
    <b style='color:#00f0ff'>MODEL</b> RandomForest<br>
    <b style='color:#00f0ff'>ACCURACY</b> 98.7%<br>
    <b style='color:#00f0ff'>DATASET</b> CWRU (12kHz)<br>
    </div>
    """, unsafe_allow_html=True)
    st.caption("CWRU 형식 .mat (12kHz, _DE_time) 지원")

# ===== 메인 =====
st.markdown('<div class="neon-title">⚡ BEARING FAULT DIAGNOSIS</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">베어링 진동 기반 회전기계 고장 진단 시스템</div>', unsafe_allow_html=True)

uploaded = st.file_uploader("진동 신호 파일 업로드 (.mat)", type=['mat'])

if uploaded is None:
    st.markdown("""
    <div style='text-align:center; padding:3rem; color:#5a6190;
                border:1px dashed rgba(0,240,255,0.3); border-radius:16px; margin-top:1rem'>
        📡 진동 신호 파일(.mat)을 업로드하면 진단이 시작됩니다
    </div>
    """, unsafe_allow_html=True)
else:
    try:
        m = loadmat(io.BytesIO(uploaded.read()))
    except Exception as e:
        st.error(f"파일을 읽을 수 없습니다: {e}"); st.stop()

    de = [k for k in m if k.endswith('_DE_time')]
    if not de:
        st.error("⚠️ drive-end 신호(_DE_time)를 찾을 수 없습니다. CWRU 형식인지 확인하세요."); st.stop()

    sig = np.asarray(m[de[0]]).ravel()
    main, proba_dict, X = predict_signal(sig)
    if main is None:
        st.error("⚠️ 신호가 너무 짧습니다 (최소 2048 포인트)."); st.stop()

    conf = proba_dict[main]
    color = NEON[main]

    # --- 상태 배너 ---
    if main == "N":
        st.markdown(f'<div class="status-banner banner-normal">✅ 정상 — 베어링에 이상이 감지되지 않았습니다</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-banner banner-fault">⚠️ {FAULT_KR[main]} 감지 — 점검이 필요합니다</div>', unsafe_allow_html=True)

    # --- 게이지 + 메트릭 ---
    col1, col2 = st.columns([1, 1.3])
    with col1:
        st.pyplot(gauge_chart(conf, color), use_container_width=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom:1rem">
            <div class="card-label">진단 결과</div>
            <div class="card-value" style="color:{color}; text-shadow:0 0 20px {color}80">
                {FAULT_KR[main]}
            </div>
        </div>
        <div class="metric-card">
            <div class="card-label">분석 윈도우 / 신호 길이</div>
            <div class="card-value">{len(X)}개 / {len(sig):,} pts</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📊 고장 유형별 확률")
    order = ["N","IR","OR","B"]
    for k in order:
        p = proba_dict.get(k, 0)
        c1, c2 = st.columns([1, 5])
        c1.markdown(f"<span style='color:{NEON[k]}; font-weight:700'>{FAULT_KR[k]}</span>", unsafe_allow_html=True)
        c2.progress(float(p), text=f"{p*100:.1f}%")

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**📈 진동 신호 파형**")
        fig, ax = plt.subplots(figsize=(6,3))
        ax.plot(sig[:WIN], linewidth=0.7, color=color)
        ax.set_xlabel("Sample", color='#8b92b8'); ax.set_ylabel("Amplitude", color='#8b92b8')
        ax.tick_params(colors='#8b92b8'); ax.grid(alpha=0.15)
        for s in ax.spines.values(): s.set_color('#2a3158')
        st.pyplot(fig)
    with col4:
        st.markdown("**🔬 주파수 스펙트럼 (FFT)**")
        xf = sig[:WIN]-sig[:WIN].mean()
        spec = np.abs(np.fft.rfft(xf)); fr = np.fft.rfftfreq(len(xf), 1/FS)
        fig2, ax2 = plt.subplots(figsize=(6,3))
        ax2.plot(fr, spec, linewidth=0.7, color='#00f0ff')
        ax2.set_xlabel("Frequency (Hz)", color='#8b92b8'); ax2.set_ylabel("Magnitude", color='#8b92b8')
        ax2.tick_params(colors='#8b92b8'); ax2.grid(alpha=0.15)
        for s in ax2.spines.values(): s.set_color('#2a3158')
        st.pyplot(fig2)

    with st.expander("🔍 추출된 feature 보기 (윈도우 평균)"):
        st.dataframe(X.mean().to_frame("평균값").round(4), use_container_width=True)