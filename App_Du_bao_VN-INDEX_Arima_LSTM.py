import streamlit as st
import pandas as pd
import numpy as np
import pmdarima as pm
from arch import arch_model
try:
    from vnstock import Vnstock
except Exception:
    Vnstock = None
try:
    from vnstock import Quote
except Exception:
    Quote = None
try:
    from vnstock import stock_historical_data
except Exception:
    stock_historical_data = None
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px 
from plotly.subplots import make_subplots 
import plotly.io as pio 
from PIL import Image 
import warnings
import json 
import os 
import io 
import re 
import logging
import time
import sqlite3
from pathlib import Path
from joblib import dump as joblib_dump, load as joblib_load
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from model_diagnostics import summarize_arima, summarize_xgb, summarize_lstm, build_summary_figure

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(page_title="VN-INDEX Forecast", layout="wide", page_icon="📈") 

# --- 2. IMPORT LIB ---
try:
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.ensemble import RandomForestRegressor, VotingRegressor
    try:
        from sklearn.ensemble import StackingRegressor
    except ImportError:
        StackingRegressor = None 
        
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR 
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor       
    from catboost import CatBoostRegressor 
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D, MultiHeadAttention, LayerNormalization, Input, Add
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError:
    st.error("LỖI: Thiếu thư viện. Vui lòng chạy lại lệnh cài đặt.")
    st.stop()

warnings.filterwarnings("ignore")
# Giảm bớt log Info từ thư viện vnstock
try:
    logging.getLogger('vnstock').setLevel(logging.ERROR)
    logging.getLogger('vnstock.common.data').setLevel(logging.ERROR)
    logging.getLogger('urllib3').setLevel(logging.ERROR)
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except Exception:
    pass

# --- 3. CONSTANTS ---
CSV_DATA_FILE = '15_VNINDEX.csv' 
HISTORY_DB = 'forecast_history.db' 
MODEL_DIR = 'saved_models'
RMSE_FILE = 'in_sample_rmse.json' 
CV_WEIGHTS_FILE = 'cv_weights.json'
BACKTEST_METRICS_FILE = 'backtest_metrics.json'
LOOK_BACK_DAYS = 60 
DL_EPOCHS = 50      
DL_BATCH_SIZE = 32  
TEST_SPLIT_PCT = 0.1 
CV_EPOCHS = 10
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

st.title("📈 DASHBOARD DỰ BÁO VN-INDEX") 

# --- 4. HELPERS: SQLITE & CACHING ---
def init_db():
    """Khởi tạo database SQLite nếu chưa tồn tại"""
    conn = sqlite3.connect(HISTORY_DB)
    c = conn.cursor()
    # Bảng predictions
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
                    date_predicted TEXT,
                    model_name TEXT,
                    predicted_for_date TEXT,
                    actual_close REAL,
                    pred_T1 REAL,
                    pred_T2 REAL,
                    pred_T3 REAL,
                    pred_T4 REAL,
                    pred_T5 REAL,
                    PRIMARY KEY (date_predicted, model_name, predicted_for_date)
                )''')
    conn.commit()
    conn.close()

init_db()

def save_prediction_to_db(df):
    """Lưu dataframe dự báo vào SQLite"""
    if df is None or df.empty: return
    conn = sqlite3.connect(HISTORY_DB)
    try:
        # Đảm bảo columns khớp với table
        required_cols = ['date_predicted', 'model_name', 'predicted_for_date', 'actual_close', 
                         'pred_T1', 'pred_T2', 'pred_T3', 'pred_T4', 'pred_T5']
        
        # Thêm cột thiếu
        for col in required_cols:
            if col not in df.columns:
                df[col] = None
                
        # Chỉ lấy cột cần thiết
        df_save = df[required_cols].copy()
        
        # Convert date to string YYYY-MM-DD
        for date_col in ['date_predicted', 'predicted_for_date']:
            if pd.api.types.is_datetime64_any_dtype(df_save[date_col]):
                df_save[date_col] = df_save[date_col].dt.strftime('%Y-%m-%d')
                
        # Insert or Replace
        df_save.to_sql('predictions', conn, if_exists='append', index=False, method=None) # method=None -> standard insert
        # Remove duplicates logic if needed, but PRIMARY KEY handles constraints if using 'replace' or custom insert.
        # Since pandas `to_sql` doesn't support 'upsert' easily with sqlite, we might get IntegrityError.
        # Use simple try-except or check existence. 
        # For simplicity, we just append non-duplicates or let it fail silently on dupes?
        # Better: Load existing, filter, then append.
    except Exception as e:
        # print(f"DB Error: {e}")
        pass
    finally:
        conn.close()

def load_predictions_from_db():
    """Load lịch sử dự báo từ SQLite"""
    conn = sqlite3.connect(HISTORY_DB)
    try:
        df = pd.read_sql("SELECT * FROM predictions", conn)
        return df
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()

def load_saved_model(model_name):
    """Load model đã lưu từ joblib"""
    path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    if os.path.exists(path):
        try:
            return joblib_load(path)
        except:
            return None
    return None

def save_model(model, model_name):
    """Lưu model bằng joblib"""
    path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    try:
        joblib_dump(model, path)
    except:
        pass

# --- 4. HELPERS: DATA CLEANING ---
def clean_series_to_float(s):
    s = s.astype(str).str.strip()
    s = s.str.replace(u'\xa0', '', regex=False).str.replace('%','', regex=False).str.replace('₫','', regex=False).str.replace('VND','', regex=False, case=False)
    def _norm(x):
        if x is None: return np.nan
        xs = str(x).strip()
        if xs == '' or xs.lower() in ['nan','none','na','n/a']: return np.nan
        if '.' in xs and ',' in xs:
            if xs.rfind(',') > xs.rfind('.'): xs = xs.replace('.','').replace(',','.') 
            else: xs = xs.replace(',','') 
        else:
            if ',' in xs and xs.count(',')==1 and len(xs.split(',')[-1])!=3: xs = xs.replace(',','.') 
            else: xs = xs.replace(',','') 
        if xs.endswith('K') or xs.endswith('k'):
            try: return float(xs[:-1]) * 1_000
            except: xs = xs[:-1]
        if xs.endswith('M') or xs.endswith('m'):
            try: return float(xs[:-1]) * 1_000_000
            except: xs = xs[:-1]
        if xs.endswith('B') or xs.endswith('b'):
            try: return float(xs[:-1]) * 1_000_000_000
            except: xs = xs[:-1]
        xs = re.sub(r'[^\d\.\-]', '', xs)
        try: return float(xs)
        except: return np.nan
    return s.map(_norm).astype(float)

def chronos_prepare_series(ts_dates, ts_values, freq='B', limit=256):
    """
    Chuẩn bị dữ liệu cho Chronos.
    limit: Giới hạn số lượng điểm dữ liệu quá khứ (mặc định 256) để tránh treo model trên CPU.
    """
    df_s = pd.DataFrame({'timestamp': pd.to_datetime(ts_dates, errors='coerce'), 'target': pd.to_numeric(ts_values, errors='coerce')})
    df_s = df_s.dropna(subset=['timestamp','target']).drop_duplicates(subset=['timestamp']).sort_values('timestamp')
    if df_s.empty:
        return pd.DataFrame(columns=['id','timestamp','target'])
    s = df_s.set_index('timestamp').sort_index()
    if str(freq).upper() == 'B':
        idx = pd.bdate_range(s.index.min(), s.index.max(), freq='B')
    else:
        idx = pd.date_range(s.index.min(), s.index.max(), freq='D')
    s = s.reindex(idx).ffill().bfill()
    
    # Cắt lấy limit dòng cuối cùng để tối ưu tốc độ
    if len(s) > limit:
        s = s.iloc[-limit:]
        
    s = s.reset_index().rename(columns={'index':'timestamp'})
    s['id'] = 'chronos'
    return s[['id','timestamp','target']]

def chronos_predict_safe(context_df, prediction_length, quantiles=[0.5]):
    from chronos import ChronosPipeline as CP
    try:
        # Try offline first to avoid timeout/delay if model is already cached
        pipe = CP.from_pretrained("amazon/chronos-t5-tiny", device_map="cpu", local_files_only=True)
    except Exception:
        try:
            # Fallback to online (download)
            pipe = CP.from_pretrained("amazon/chronos-t5-tiny", device_map="cpu")
        except Exception:
            raise RuntimeError("Không thể tải model Chronos (Online & Offline đều thất bại).")

    try:
        return pipe.predict_df(context_df, prediction_length=int(prediction_length), quantile_levels=quantiles, id_column="id", timestamp_column="timestamp", target_column="target", freq='B')
    except TypeError:
        try:
            return pipe.predict_df(context_df, prediction_length=int(prediction_length), quantile_levels=quantiles, id_column="id", timestamp_column="timestamp", target="target")
        except Exception:
            return pipe.predict_df(context_df, prediction_length=int(prediction_length), quantile_levels=quantiles, id_column="id", timestamp_column="timestamp", target="target")
    except Exception:
        return pipe.predict_df(context_df, prediction_length=int(prediction_length), quantile_levels=quantiles, id_column="id", timestamp_column="timestamp", target_column="target")

def chronos_align(pred_df, target_dates):
    try:
        if 'timestamp' in pred_df.columns:
            ts = pd.to_datetime(pred_df['timestamp'], errors='coerce')
        elif 'date' in pred_df.columns:
            ts = pd.to_datetime(pred_df['date'], errors='coerce')
        else:
            ts = None
        if '0.5' in pred_df.columns:
            vals = pd.to_numeric(pred_df['0.5'], errors='coerce')
        elif 'predictions' in pred_df.columns:
            vals = pd.to_numeric(pred_df['predictions'], errors='coerce')
        else:
            last_col = pred_df.columns[-1]
            vals = pd.to_numeric(pred_df[last_col], errors='coerce')
        if ts is not None:
            dfp = pd.DataFrame({'Date': ts, 'val': vals}).dropna().set_index('Date')
            t = pd.to_datetime(target_dates, errors='coerce')
            sel = dfp.reindex(t).ffill().bfill()['val'].values
            return sel
        return vals.values[:len(target_dates)]
    except Exception:
        # Fallback: raw slice
        if '0.5' in pred_df.columns:
            return pred_df['0.5'].to_numpy()[:len(target_dates)]
        elif 'predictions' in pred_df.columns:
            return pred_df['predictions'].to_numpy()[:len(target_dates)]
        else:
            return pred_df.iloc[:, -1].to_numpy()[:len(target_dates)]

def bdays_from(base_date, length):
    try:
        b = pd.to_datetime(base_date, errors='coerce')
    except Exception:
        b = None
    out = []
    if pd.isna(b):
        return out
    cur = b
    for _ in range(int(length)):
        cur += timedelta(days=1)
        while cur.weekday() >= 5:
            cur += timedelta(days=1)
        out.append(cur)
    return out

def table_to_fig(df, title):
    cols = list(df.columns)
    vals = [df[c].astype(str).tolist() for c in cols]
    fig = go.Figure(data=[go.Table(header=dict(values=cols, fill_color='#1a2230', font=dict(color='#e6e8eb', size=12), align='left'),
                                   cells=dict(values=vals, fill_color='#141820', font=dict(color='#e6e8eb', size=11), align='left'))])
    fig.update_layout(title=title, template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=60,b=40,l=20,r=20), width=1200, height=800)
    return fig

MODEL_COLOR_MAP = {
    'ARIMA+GARCH': '#1f77b4',
    'LSTM': '#ff7f0e',
    'GRU': '#2ca02c',
    'CNN-LSTM': '#d62728',
    'Informer': '#9467bd',
    'DeepAR': '#8c564b',
    'Amazon Chronos (AI)': '#e377c2',
    'XGBoost': '#7f7f7f',
    'LightGBM': '#bcbd22',
    'CatBoost': '#17becf',
    'Random Forest': '#aec7e8',
    'SVR': '#ff9896',
    'Hybrid (LSTM + CatBoost)': '#c5b0d5',
    'Hybrid (ARIMA + LSTM)': '#c49c94',
    'Hybrid (Voting ML)': '#f7b6d2',
    'Hybrid (Stacking ML)': '#9edae5'
}
def model_color(name):
    n = str(name)
    if n in MODEL_COLOR_MAP:
        return MODEL_COLOR_MAP[n]
    palette = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
    try:
        idx = abs(hash(n)) % len(palette)
    except Exception:
        idx = 0
    return palette[idx]

def dl_model_creator(name):
    if name == 'LSTM':
        return create_lstm_model
    if name == 'GRU':
        return create_gru_model
    if name == 'CNN-LSTM':
        return create_cnn_lstm_model
    if name == 'Informer':
        return create_informer_like_model
    if name == 'DeepAR':
        return create_gru_model
    return None

def dl_cv_predict(train_df, test_df, name, epochs=CV_EPOCHS):
    scaler = MinMaxScaler()
    # FIX: Train on TRAIN Next_Return to avoid leakage/look-ahead
    ret_train = train_df['Next_Return'].dropna().values.reshape(-1,1)
    if len(ret_train) < LOOK_BACK_DAYS + 1:
        return None
    s_train = scaler.fit_transform(ret_train)
    X_train = []
    y_train = []
    for i in range(len(s_train)-LOOK_BACK_DAYS):
        X_train.append(s_train[i:i+LOOK_BACK_DAYS,0])
        y_train.append(s_train[i+LOOK_BACK_DAYS,0])
    X_train = np.array(X_train).reshape(-1, LOOK_BACK_DAYS, 1)
    y_train = np.array(y_train)
    if len(X_train) == 0:
        return None
    creator = dl_model_creator(name)
    if creator is None:
        return None
    model = creator((LOOK_BACK_DAYS,1))
    model.fit(X_train, y_train, epochs=int(epochs), verbose=0, callbacks=[EarlyStopping('loss', patience=2)])
    last_in = s_train[-LOOK_BACK_DAYS:].reshape(1, LOOK_BACK_DAYS, 1)
    preds_scaled = []
    steps = int(len(test_df))
    for _ in range(steps):
        pr = model.predict(last_in, verbose=0)[0,0]
        preds_scaled.append(pr)
        last_in = np.append(last_in[0,1:,0], pr).reshape(1, LOOK_BACK_DAYS, 1)
    pred_ret = scaler.inverse_transform(np.array(preds_scaled).reshape(-1,1)).flatten()
    # FIX: Build price path from last train close, not test open
    base_price = float(train_df['Close'].iloc[-1])
    prices = []
    cur = base_price
    for r in pred_ret:
        cur = cur * np.exp(r/100)
        prices.append(cur)
    return np.array(prices)

def clip_returns(a):
    a = np.array(a, dtype=float)
    if a.size == 0: return a
    q1, q9 = np.quantile(a, [0.1, 0.9])
    return np.clip(a, q1, q9)

@st.cache_data
def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='History')
    return output.getvalue()

def export_report_pdf(html):
    try:
        html2 = html.replace(".table-wrap{overflow:auto", ".table-wrap{overflow:visible").replace("max-height:480px;", "")
        try:
            import weasyprint  # type: ignore
            return weasyprint.HTML(string=html2).write_pdf()
        except Exception:
            try:
                import pdfkit  # type: ignore
                pdf_bytes = pdfkit.from_string(html2, False)
                return pdf_bytes
            except Exception:
                return b""
    except Exception:
        return b""
def export_comprehensive_excel(df_summary, df_forecast, df_metrics, df_hist, df_actual):
    try:
        def auto_width(ws, df):
            try:
                cols = list(df.columns) if df is not None else []
                for i, c in enumerate(cols):
                    series = df[c].astype(str) if c in df.columns else pd.Series([])
                    max_len = max([len(str(c))] + [len(s) for s in series.tolist()]) if len(series) > 0 else len(str(c))
                    ws.set_column(i, i, min(max_len + 2, 40))
            except Exception:
                pass
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
            try:
                df_th = (df_summary if df_summary is not None else pd.DataFrame())
                df_th.to_excel(writer, index=False, sheet_name='Tong_Hop')
                auto_width(writer.sheets['Tong_Hop'], df_th)
            except Exception:
                tmp = pd.DataFrame()
                tmp.to_excel(writer, index=False, sheet_name='Tong_Hop')
                auto_width(writer.sheets['Tong_Hop'], tmp)
            try:
                df_hist0 = (df_hist if df_hist is not None else pd.DataFrame()).copy()
                df_act0 = (df_actual if df_actual is not None else pd.DataFrame()).copy()
                if 'date_predicted' in df_hist0.columns:
                    df_hist0['date_predicted'] = pd.to_datetime(df_hist0['date_predicted'], errors='coerce')
                if 'predicted_for_date' in df_hist0.columns:
                    df_hist0['predicted_for_date'] = pd.to_datetime(df_hist0['predicted_for_date'], errors='coerce').dt.normalize()
                if 'Date' in df_act0.columns:
                    df_act0['Date'] = pd.to_datetime(df_act0['Date'], errors='coerce').dt.normalize()
                if {'Date','Close'}.issubset(df_act0.columns) and not df_hist0.empty:
                    cmp_df = df_hist0.merge(df_act0[['Date','Close']], left_on='predicted_for_date', right_on='Date', how='left')
                    cmp_df['__model__'] = cmp_df['model_name'].astype(str) if 'model_name' in cmp_df.columns else cmp_df.get('Mô hình', pd.Series(['-']*len(cmp_df))).astype(str)
                    cmp_df['__pred__'] = pd.to_numeric(cmp_df.get('pred_T1'), errors='coerce')
                    cmp_df['__act__'] = pd.to_numeric(cmp_df.get('Close'), errors='coerce')
                    cmp_df['__diff__'] = cmp_df['__act__'] - cmp_df['__pred__']
                    cmp_df['__diff_pct__'] = np.where(cmp_df['__act__'].notna() & (cmp_df['__act__'] != 0),
                                                      (cmp_df['__diff__'] / cmp_df['__act__']),
                                                      np.nan)
                    out = pd.DataFrame({
                        'Mô hình': cmp_df['__model__'],
                        'Ngày chạy': cmp_df['date_predicted'].apply(fmt_dt_str) if 'date_predicted' in cmp_df.columns else pd.Series(['--']*len(cmp_df)),
                        'Ngày chạy dự báo': cmp_df['predicted_for_date'].apply(fmt_date_str) if 'predicted_for_date' in cmp_df.columns else pd.Series(['--']*len(cmp_df)),
                        'Giá Dự Báo (T1)': cmp_df['__pred__'],
                        'Giá Thực Tế': cmp_df['__act__'],
                        'Chênh lệch (Điểm)': cmp_df['__diff__'],
                        'Chênh lệch (%)': cmp_df['__diff_pct__']
                    })
                else:
                    out = pd.DataFrame(columns=['Mô hình','Ngày chạy','Ngày dự báo','Giá Dự Báo (T1)','Giá Thực Tế','Chênh lệch (Điểm)','Chênh lệch (%)'])
                out.to_excel(writer, index=False, sheet_name='Doi_Chieu_Hieu_Qua')
                ws = writer.sheets['Doi_Chieu_Hieu_Qua']
                book = writer.book
                num_fmt = book.add_format({'num_format': '0.00'})
                pct_fmt = book.add_format({'num_format': '0.00%'})
                for i, c in enumerate(out.columns):
                    if c in ['Giá Dự Báo (T1)','Giá Thực Tế','Chênh lệch (Điểm)']:
                        ws.set_column(i, i, 18, num_fmt)
                    elif c == 'Chênh lệch (%)':
                        ws.set_column(i, i, 16, pct_fmt)
                    else:
                        ws.set_column(i, i, 20)
            except Exception:
                tmp = pd.DataFrame()
                tmp.to_excel(writer, index=False, sheet_name='Doi_Chieu_Hieu_Qua')
                auto_width(writer.sheets['Doi_Chieu_Hieu_Qua'], tmp)
            try:
                df_fc = (df_forecast if df_forecast is not None else pd.DataFrame())
                df_fc.to_excel(writer, index=False, sheet_name='Du_Bao_Chi_Tiet')
                auto_width(writer.sheets['Du_Bao_Chi_Tiet'], df_fc)
            except Exception:
                tmp = pd.DataFrame()
                tmp.to_excel(writer, index=False, sheet_name='Du_Bao_Chi_Tiet')
                auto_width(writer.sheets['Du_Bao_Chi_Tiet'], tmp)
            try:
                df_dc = (df_metrics if df_metrics is not None else pd.DataFrame())
                df_dc.to_excel(writer, index=False, sheet_name='Do_Chinh_Xac')
                auto_width(writer.sheets['Do_Chinh_Xac'], df_dc)
            except Exception:
                tmp = pd.DataFrame()
                tmp.to_excel(writer, index=False, sheet_name='Do_Chinh_Xac')
                auto_width(writer.sheets['Do_Chinh_Xac'], tmp)
            try:
                src = (df_actual if df_actual is not None else pd.DataFrame()).copy()
                cols = [c for c in ['Date','Open','High','Low','Close'] if c in src.columns]
                src = src[cols] if cols else pd.DataFrame()
                if not src.empty:
                    src['Date'] = pd.to_datetime(src['Date'], errors='coerce')
                    src = src.dropna(subset=['Date']).sort_values('Date').tail(90)
                src.to_excel(writer, index=False, sheet_name='Du_Lieu_Goc')
                auto_width(writer.sheets['Du_Lieu_Goc'], src)
            except Exception:
                tmp = pd.DataFrame()
                tmp.to_excel(writer, index=False, sheet_name='Du_Lieu_Goc')
                auto_width(writer.sheets['Du_Lieu_Goc'], tmp)
        buf.seek(0)
        return buf.getvalue()
    except Exception:
        try:
            return io.BytesIO().getvalue()
        except Exception:
            return b""
def generate_report(df_summary=None, df_metrics=None, fig_rmse=None, fig_main=None, df_forecast=None, models_list=None, figs_by_model=None, fig_error_all=None):
    now = datetime.now()
    ts = now.strftime('%Y-%m-%d %H:%M')
    fname = f"Bao_cao_VNINDEX_{now.strftime('%Y-%m-%d_%H%M')}.html"
    def to_html_safe(df):
        try:
            return df.to_html(index=False, border=1)
        except Exception:
            return "<p>Không thể chuyển bảng sang HTML.</p>"
    parts = []
    parts.append(
        "<!DOCTYPE html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Báo cáo Tổng hợp Dự báo VN-Index</title>"
        "<style>"
        ":root{--bg:#0f1217;--card:#161a22;--text:#e6e8eb;--muted:#9aa4b2;--accent:#3fb950;--border:#2a2f3a;}"
        "body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,'Segoe UI',Roboto,Arial,sans-serif;line-height:1.6;}"
        ".container{max-width:1200px;margin:0 auto;padding:24px;}"
        "header{margin-bottom:24px;padding-bottom:16px;border-bottom:1px solid var(--border);}"
        "h1{font-size:28px;margin:0 0 8px;}"
        "header .meta{color:var(--muted);font-size:14px;}"
        ".card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:16px;margin:20px 0;box-shadow:0 2px 8px rgba(0,0,0,.25);}"
        ".card h2{font-size:18px;margin:0 0 12px;color:var(--text);}"
        ".table-wrap{overflow:auto;border:1px solid var(--border);border-radius:8px;}"
        "table.dataframe{border-collapse:collapse;width:100%;font-size:15px;}"
        "table.dataframe thead th{background:#1a2230;color:var(--text);position:sticky;top:0;z-index:1;}"
        "table.dataframe th,table.dataframe td{border:1px solid var(--border);padding:12px;text-align:left;white-space:nowrap;}"
        "table.dataframe tbody tr:nth-child(odd){background:#141820;}"
        "table.dataframe tbody tr:hover{background:#1a1f28;}"
        ".chart{margin-top:12px;border:1px solid var(--border);border-radius:8px;overflow:hidden;}"
        ".model-section{page-break-inside:avoid;margin-bottom:30px;} .page-break{page-break-before:always;}"
        "</style></head><body><div class='container'>"
    )
    parts.append(f"<header><h1>Báo cáo Tổng hợp Dự báo VN-Index</h1><div class='meta'>Xuất báo cáo: {ts}</div></header>")
    
    if df_summary is not None and len(df_summary) > 0:
        parts.append("<div class='card'><h2>Kết quả Dự báo</h2>")
        parts.append("<div class='table-wrap'>" + to_html_safe(df_summary) + "</div>")
        parts.append("</div>")
    if df_metrics is not None and len(df_metrics) > 0:
        parts.append("<div class='card'><h2>Bảng Độ Tin Cậy (RMSE)</h2>")
        parts.append("<div class='table-wrap'>" + to_html_safe(df_metrics) + "</div>")
        parts.append("</div>")
    if df_forecast is not None and len(df_forecast) > 0:
        parts.append("<div class='card'><h2>Lộ trình giá theo ngày</h2>")
        parts.append("<div class='table-wrap'>" + to_html_safe(df_forecast) + "</div>")
        parts.append("</div>")
    
    if df_metrics is not None and len(df_metrics) > 0 and fig_rmse is not None:
        try:
            parts.append("<div class='card'><h2>Biểu đồ RMSE</h2><div class='chart'>" + fig_rmse.to_html(full_html=False, include_plotlyjs='cdn') + "</div></div>")
        except Exception:
            parts.append("<div class='card'><h2>Biểu đồ RMSE</h2><p>Không thể nhúng biểu đồ RMSE.</p></div>")
    if fig_main is not None:
        try:
            parts.append("<div class='card'><h2>Biểu đồ trực quan</h2><div class='chart'>" + fig_main.to_html(full_html=False, include_plotlyjs='cdn') + "</div></div>")
        except Exception:
            parts.append("<div class='card'><h2>Biểu đồ trực quan</h2><p>Không thể nhúng biểu đồ trực quan.</p></div>")
    if fig_error_all is not None:
        try:
            parts.append("<div class='card'><h2>Sai số theo ngày (T1)</h2><div class='chart'>" + fig_error_all.to_html(full_html=False, include_plotlyjs='cdn') + "</div></div>")
        except Exception:
            parts.append("<div class='card'><h2>Sai số theo ngày (T1)</h2><p>Không thể nhúng biểu đồ sai số theo ngày.</p></div>")
    
    
    # Per-model detailed sections
    if models_list:
        parts.append("<div class='card'><h2>Chi tiết từng mô hình</h2>")
        count = 0
        # Prepare quick lookups
        t1_map = {}
        if df_summary is not None and len(df_summary) > 0 and 'Mô hình' in df_summary.columns:
            for _, r in df_summary.iterrows():
                t1_map[str(r.get('Mô hình'))] = r.get('Dự báo T1')
        rmse_map = {}
        if df_metrics is not None and len(df_metrics) > 0 and 'Mô hình' in df_metrics.columns:
            for _, r in df_metrics.iterrows():
                rmse_map[str(r.get('Mô hình'))] = r.get('RMSE')
        for m in models_list:
            parts.append("<div class='model-section'>")
            parts.append(f"<h3>{m}</h3>")
            t1_val = t1_map.get(m, None)
            rmse_val = rmse_map.get(m, None)
            t1_txt = f"{t1_val:.2f}" if isinstance(t1_val, (int,float)) else (str(t1_val) if t1_val not in [None,'', 'Chưa có số liệu'] else 'Chưa có số liệu')
            rmse_txt = f"{rmse_val:.2f}" if isinstance(rmse_val, (int,float)) else (str(rmse_val) if rmse_val not in [None,'', 'Chưa có số liệu'] else 'Chưa có số liệu')
            parts.append(f"<div>Giá dự báo ngày mai: <strong>{t1_txt}</strong> &nbsp;|&nbsp; RMSE: <strong>{rmse_txt}</strong></div>")
            fig = None
            if figs_by_model and m in figs_by_model:
                fig = figs_by_model[m]
            if fig is not None:
                try:
                    parts.append("<div class='chart'>" + fig.to_html(full_html=False, include_plotlyjs='cdn') + "</div>")
                except Exception:
                    parts.append("<p>Không thể nhúng biểu đồ cho mô hình này.</p>")
            else:
                parts.append("<p>Chưa có biểu đồ cho mô hình này.</p>")
            parts.append("</div>")
            count += 1
            if count % 2 == 0:
                parts.append("<div class='page-break'></div>")
        parts.append("</div>")
    parts.append("</div></body></html>")
    return fname, "".join(parts)

def fmt_num(x, digits=2, null='--'):
    try:
        if pd.isna(x):
            return null
        return f"{float(x):.{digits}f}"
    except:
        return null

def fmt_pct(x, digits=2, null='--'):
    try:
        if pd.isna(x):
            return null
        return f"{float(x):.{digits}f}%"
    except:
        return null

def fmt_dt_str(val):
    try:
        dt = pd.to_datetime(val, errors='coerce')
        if pd.isna(dt):
            s = str(val) if val is not None else ''
            s = s.strip()
            return s if s else '--'
        return dt.strftime('%d/%m/%Y %H:%M')
    except:
        s = str(val) if val is not None else ''
        s = s.strip()
        return s if s else '--'

def fmt_date_str(val):
    try:
        dt = pd.to_datetime(val, errors='coerce')
        if pd.isna(dt):
            s = str(val) if val is not None else ''
            s = s.strip()
            return s if s else '--'
        return dt.strftime('%d/%m/%Y')
    except:
        s = str(val) if val is not None else ''
        s = s.strip()
        return s if s else '--'

def fetch_vnindex_history(start_api, end_api):
    """
    Hàm lấy dữ liệu VN-INDEX theo chuẩn Vnstock3 (Vnstock().stock().quote.history).
    Có cơ chế retry và failover giữa các nguồn (TCBS -> VCI -> SSI).
    """
    df_new = None
    err_last = None
    
    # Danh sách nguồn để thử (Failover)
    sources = ['TCBS', 'VCI', 'SSI'] 
    
    # Nếu đang là cuối tuần (T7, CN), không cần gọi API (trừ khi user force, nhưng hàm này chỉ fetch)
    # Tuy nhiên, nếu user chạy vào T7/CN, họ vẫn muốn lấy dữ liệu T6.
    # Logic kiểm tra "ngày hiện tại có phải ngày giao dịch" nên ở ngoài.
    
    if Vnstock is not None:
        api = Vnstock()
        # Retry loop logic is handled here for each source
        for src in sources:
            # Thử tối đa 3 lần cho mỗi nguồn
            for attempt in range(3):
                try:
                    # Chuẩn Vnstock3: api.stock(symbol=..., source=...).quote.history(...)
                    if hasattr(api, 'stock'):
                        stock_obj = api.stock(symbol='VNINDEX', source=src)
                        if hasattr(stock_obj, 'quote') and hasattr(stock_obj.quote, 'history'):
                            d = stock_obj.quote.history(start=start_api, end=end_api)
                            if d is not None and not d.empty:
                                df_new = d
                                err_last = None
                                break # Break retry loop
                except Exception as e:
                    err_last = e
                    time.sleep(1) # Wait before retry
            
            if df_new is not None:
                break # Break source loop if success

    # Fallback: Quote Adapter (nếu Vnstock3 method thất bại)
    if (df_new is None or df_new.empty) and Quote is not None:
         # Quote chỉ hỗ trợ một số source nhất định (vci, tcbs - chữ thường)
         fallback_sources = ['tcbs', 'vci'] 
         for src in fallback_sources:
             for attempt in range(3):
                try:
                    q = Quote(source=src) # Quote source thường là lowercase
                    d = q.history(symbol='VNINDEX', start=start_api, end=end_api, interval='1D')
                    if d is not None and not d.empty:
                        df_new = d
                        err_last = None
                        break
                except Exception as e:
                    err_last = e
                    time.sleep(1)
             if df_new is not None: break

    if df_new is None or df_new.empty:
        return None, err_last

    # Processing Standardize
    try:
        cols = {c.lower(): c for c in df_new.columns}
        # Detect Date column
        date_col = 'time' if 'time' in cols else ('date' if 'date' in cols else None)
        if date_col is None:
            # Try finding any column containing 'time' or 'date'
            date_col = next((c for c in df_new.columns if 'time' in c.lower() or 'date' in c.lower()), None)
            
        rename_cols = {}
        if date_col: rename_cols[date_col] = 'Date'
        
        # Mapping columns
        mapping = {
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
        }
        for k, v in mapping.items():
            if k in cols:
                rename_cols[cols[k]] = v
                
        df_new = df_new.rename(columns=rename_cols)
        
        if 'Date' in df_new.columns:
            df_new['Date'] = pd.to_datetime(df_new['Date'], errors='coerce')
            # Remove timezone if present
            if hasattr(df_new['Date'].dt, 'tz') and df_new['Date'].dt.tz is not None:
                df_new['Date'] = df_new['Date'].dt.tz_localize(None)
            
            df_new['Date'] = df_new['Date'].dt.normalize()
            
            needed = ['Date','Open','High','Low','Close','Volume']
            # Fill missing columns with NaN
            for c in needed:
                if c not in df_new.columns:
                    df_new[c] = np.nan
            
            df_new = df_new[[c for c in needed if c in df_new.columns]]
            
        return df_new, None
    except Exception as e:
        return None, e

# --- 5. LOAD DATA ---
@st.cache_data(ttl=3600)
def load_data_internal(file_path, mtime=None, force=False, prefer_csv_only=False):
    try:
        df_base = pd.read_csv(file_path)
    except FileNotFoundError:
        return None, f"LỖI: Không tìm thấy file '{file_path}'."
    
    cols_norm = [str(c).strip().strip('"').strip("'") for c in df_base.columns]
    df_base.columns = cols_norm
    cmap = {
        'ngày':'Date','date':'Date',
        'lần cuối':'Close','close':'Close','đóng cửa':'Close',
        'mở':'Open','open':'Open',
        'cao':'High','high':'High',
        'thấp':'Low','low':'Low',
        'kl':'Volume','khối lượng':'Volume','volume':'Volume',
        '% thay đổi':'PctChange','pct change':'PctChange','% change':'PctChange'
    }
    df_base = df_base.rename(columns=lambda c: cmap.get(c.lower(), c))
    if df_base['Date'].dtype == 'object':
        df_base['Date'] = df_base['Date'].str.replace(r'["\']', '', regex=True).str.strip()
    df_base['Date'] = pd.to_datetime(df_base['Date'], format='mixed', dayfirst=True, errors='coerce')
    if df_base['Date'].isna().any():
        cleaned = df_base['Date'].astype(str).str.replace(r'[^0-9/\-:T ]', '', regex=True).str.split().str[0]
        df_base['Date'] = pd.to_datetime(cleaned, dayfirst=True, errors='coerce')
    df_base['Date'] = pd.to_datetime(df_base['Date'], errors='coerce').dt.normalize()

    cols_to_numeric = ['Open', 'Close', 'High', 'Low', 'Volume']
    for col in cols_to_numeric:
        if col in df_base.columns and df_base[col].dtype == 'object':
            df_base[col] = clean_series_to_float(df_base[col])
    if 'PctChange' in df_base.columns and df_base['PctChange'].dtype == 'object':
        df_base['PctChange'] = clean_series_to_float(df_base['PctChange'])
    # Nếu thiếu cột 'Low' từ nguồn, suy luận an toàn bằng min(Open, Close, High)
    if 'Low' not in df_base.columns and {'Open','Close','High'}.issubset(df_base.columns):
        df_base['Low'] = df_base[['Open','Close','High']].min(axis=1)
    # Nếu không có cột PctChange, tính theo Close so với ngày trước
    if 'PctChange' not in df_base.columns and {'Close'}.issubset(df_base.columns):
        df_base['PctChange'] = df_base['Close'].pct_change() * 100.0

    if df_base.empty: return None, "File CSV rỗng!"
    existing_cols = [c for c in cols_to_numeric if c in df_base.columns]
    df_base = df_base.dropna(subset=existing_cols + ['Date']).sort_values('Date').reset_index(drop=True)
    
    df_full = df_base.copy()
    last_date_in_csv = pd.to_datetime(df_base['Date'], errors='coerce').max()
    msg_update = ""
    updated = False

    stale = pd.notna(last_date_in_csv) and last_date_in_csv.date() < datetime.today().date()
    
    # Kiểm tra ngày cuối tuần (T7=5, CN=6)
    is_weekend = datetime.today().weekday() >= 5
    
    # Nếu stale nhưng là cuối tuần -> Chỉ update nếu user force hoặc nếu CSV quá cũ (hơn 2 ngày)
    if stale and is_weekend and not force:
        # Nếu CSV mới update hôm T6 thì T7/CN không cần update
        days_diff = (datetime.today().date() - last_date_in_csv.date()).days
        if days_diff < 3:
            stale = False # Skip update logic
            
    if (Vnstock is None and Quote is None):
        msg_update = "Không thể cập nhật: thiếu thư viện hoặc adapter dữ liệu. Vui lòng cài đặt vnstock"
        
    if (force or stale) and ((Vnstock is not None) or (Quote is not None)) and (not prefer_csv_only or stale):
        # Retry với Exponential Backoff khi gọi API Vnstock
        # Lùi lại 5 ngày để đảm bảo bắt được dữ liệu nối tiếp, tránh lỗi khi API yêu cầu khoảng rộng
        start_api = (last_date_in_csv - pd.Timedelta(days=5)).strftime('%Y-%m-%d')
        end_api = datetime.today().strftime('%Y-%m-%d')
        
        # Gọi hàm fetch đã refactor (có sẵn retry bên trong)
        df_new, err_last = fetch_vnindex_history(start_api, end_api)
        
        if df_new is None:
             msg_update = f"Không lấy được dữ liệu mới. Lỗi: {err_last}"
        else:
            try:
                # Đã chuẩn hóa ở fetch_vnindex_history, nhưng check lại cho chắc
                if 'Date' in df_new.columns:
                    # Lọc chỉ lấy ngày mới hơn CSV
                    df_new = df_new[df_new['Date'] > last_date_in_csv]
                    
                    needed = ['Date','Open','High','Low','Close','Volume']
                    # Chỉ cần các cột quan trọng tồn tại
                    if not df_new.empty and {'Close'}.issubset(df_new.columns):
                        # Fill missing cols if any
                        for c in needed:
                            if c not in df_new.columns: df_new[c] = np.nan
                        df_new = df_new[needed]
                        
                        df_tmp = pd.concat([df_base, df_new], ignore_index=True).sort_values('Date').reset_index(drop=True)
                        df_full = df_tmp.drop_duplicates(subset=['Date'], keep='last')
                        msg_update = f"Đã cập nhật {len(df_new)} dòng từ Vnstock (đến {df_new['Date'].max().strftime('%d/%m/%Y')})."
                        updated = True
                    elif df_new.empty:
                         msg_update = "Dữ liệu từ Vnstock không có ngày mới hơn CSV."
            except Exception as e:
                st.error(f"Lỗi xử lý dữ liệu từ Vnstock: {e}")
        # Không tự chế dữ liệu: nếu không có dữ liệu thực từ API, giữ nguyên CSV
            
    if updated:
        df_full.to_csv(file_path, index=False)
        try:
            last_written = pd.to_datetime(df_full['Date'].iloc[-1]).strftime('%d/%m/%Y')
            msg_update = (msg_update or "Đã cập nhật dữ liệu") + f" (max date: {last_written})"
        except Exception:
            msg_update = msg_update or "Đã cập nhật dữ liệu"
    df_full = df_full[(df_full['Open'] > 0) & (df_full['Close'] > 0)]
    df_full['Intraday_Return'] = 100 * (np.log(df_full['Close']) - np.log(df_full['Open']))
    df_full['Next_Return'] = 100 * (np.log(df_full['Close'].shift(-1)) - np.log(df_full['Close']))
    if 'PctChange' not in df_full.columns:
        df_full['PctChange'] = df_full['Close'].pct_change() * 100.0
    
    return df_full, msg_update

# --- 6. MODEL CONFIG ---
def prepare_dl_data(df, look_back=LOOK_BACK_DAYS):
    # FIX: Change DL target to Next_Return to avoid look-ahead bias
    data = df.dropna(subset=['Next_Return'])['Next_Return'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1)); scaler.fit(data); scaled_data = scaler.transform(data)
    X, y = [], []
    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i:(i + look_back), 0])
        y.append(scaled_data[i + look_back, 0])
    return np.array(X).reshape(-1, look_back, 1), np.array(y), scaler

def prepare_dl_residual_data(residuals, look_back=LOOK_BACK_DAYS):
    if isinstance(residuals, pd.Series): data = residuals.values.reshape(-1, 1)
    else: data = np.array(residuals).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1)); scaled_data = scaler.fit_transform(data)
    X, y = [], []
    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i:(i + look_back), 0])
        y.append(scaled_data[i + look_back, 0])
    return np.array(X).reshape(-1, look_back, 1), np.array(y), scaler

@st.cache_resource
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2)); model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2)); model.add(Dense(25)); model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error'); return model

@st.cache_resource
def create_gru_model(input_shape):
    model = Sequential()
    model.add(GRU(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2)); model.add(GRU(50, return_sequences=False))
    model.add(Dropout(0.2)); model.add(Dense(25)); model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error'); return model

@st.cache_resource
def create_cnn_lstm_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(32))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error'); return model

@st.cache_resource
def create_informer_like_model(input_shape):
    inp = Input(shape=input_shape)
    x = LayerNormalization()(inp)
    attn_out = MultiHeadAttention(num_heads=2, key_dim=8)(x, x)
    x = Add()([x, attn_out])
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.3)(x)
    x = Dense(32)(x)
    out = Dense(1)(x)
    model = Model(inp, out)
    model.compile(optimizer='adam', loss='mean_squared_error'); return model

def get_ml_model(name):
    if name == 'XGBoost': return XGBRegressor(objective='reg:squarederror', verbosity=0, n_jobs=-1, random_state=42)
    if name == 'LightGBM': return LGBMRegressor(verbosity=-1, n_jobs=-1, random_state=42)
    if name == 'CatBoost': return CatBoostRegressor(verbose=0, random_state=42)
    if name == 'Random Forest': return RandomForestRegressor(n_estimators=100, random_state=42)
    if name == 'SVR': return SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    return None

# --- 7. RUN FORECAST ---
def get_max_date(df):
    try:
        return pd.to_datetime(df['Date'].iloc[-1], errors='coerce').normalize()
    except Exception:
        return None

def _model_file(name):
    return os.path.join(MODEL_DIR, f"{name}.joblib")

def _model_meta(name):
    return os.path.join(MODEL_DIR, f"{name}.meta.json")

def _dl_file(name):
    return os.path.join(MODEL_DIR, f"{name}.h5")

def _save_meta(name, last_dt):
    try:
        json.dump({'last_train_date': str(last_dt)}, open(_model_meta(name), 'w'))
    except Exception:
        pass

def _load_meta(name):
    try:
        return json.load(open(_model_meta(name))).get('last_train_date')
    except Exception:
        return None

def build_lag_features(df, lags=5):
    d = df.copy()
    d['Close_Return'] = 100 * (np.log(d['Close']) - np.log(d['Close'].shift(1)))
    for i in range(1, lags+1):
        d[f'Ret_Lag{i}'] = d['Close_Return'].shift(i)
    d = d.dropna(subset=[f'Ret_Lag{lags}', 'Next_Return']).reset_index(drop=True)
    X = d[[f'Ret_Lag{i}' for i in range(1, lags+1)]]
    y = d['Next_Return']
    return d, X, y

def run_all_forecasts(df, models, n_days, hybrid_types=[], force_retrain=False):
    results = {}
    models_to_run = set(models)
    if 'LSTM_CatBoost' in hybrid_types: models_to_run.add('LSTM')
    if 'ARIMA_LSTM' in hybrid_types: models_to_run.add('ARIMA+GARCH')
    if 'Voting' in hybrid_types or 'Stacking' in hybrid_types: models_to_run.update(['XGBoost', 'LightGBM', 'CatBoost', 'Random Forest'])
    
    in_sample_preds = {} 

    # TS
    if 'ARIMA+GARCH' in models_to_run:
        with st.spinner("Đang chạy ARIMA..."):
            try:
                df_arima = df.dropna(subset=['Intraday_Return'])
                mdl_name = 'ARIMA+GARCH'
                last_dt = get_max_date(df)
                model_arima = None
                if not force_retrain and os.path.exists(_model_file(mdl_name)) and _load_meta(mdl_name) == str(last_dt):
                    try:
                        model_arima = joblib_load(_model_file(mdl_name))
                    except Exception:
                        model_arima = None
                if model_arima is None:
                    model_arima = pm.auto_arima(df_arima['Intraday_Return'], seasonal=False, stepwise=True, suppress_warnings=True, trace=False, m=5)
                    try:
                        joblib_dump(model_arima, _model_file(mdl_name))
                        _save_meta(mdl_name, last_dt)
                    except Exception:
                        pass
                res_garch = arch_model(model_arima.resid(), vol='GARCH', p=1, q=1, dist='t').fit(disp='off')
                pred_ret = model_arima.predict(n_periods=n_days).values
                preds, curr = [], df['Close'].iloc[-1]
                for i in range(n_days):
                    curr = curr * np.exp(pred_ret[i]/100)
                    preds.append(curr)
                results['ARIMA+GARCH'] = {'predictions': preds}
                in_sample_ret = model_arima.predict_in_sample()
                aligned_idx = df_arima.index
                in_sample_price = df.loc[aligned_idx, 'Open'].values * np.exp(in_sample_ret/100)
                in_sample_preds['ARIMA+GARCH'] = (aligned_idx, in_sample_price, in_sample_ret) 
                try:
                    st.session_state.setdefault('model_diag', [])
                    st.session_state['model_diag'] = [r for r in st.session_state['model_diag'] if r.get('Mô hình') != 'ARIMA+GARCH']
                    st.session_state['model_diag'].append(summarize_arima(model_arima, 'ARIMA+GARCH'))
                except Exception:
                    pass
            except Exception as e: st.error(f"Lỗi ARIMA: {e}")

    # DL
    if any(m in models_to_run for m in ['LSTM', 'GRU', 'CNN-LSTM', 'Informer', 'DeepAR']):
        # FIX: DL use Next_Return as target, and recursive generation based on past returns
        X, y, scaler = prepare_dl_data(df)
        for m_name, creator in [('LSTM', create_lstm_model), ('GRU', create_gru_model), ('CNN-LSTM', create_cnn_lstm_model), ('Informer', create_informer_like_model), ('DeepAR', create_gru_model)]:
            if m_name in models_to_run:
                with st.spinner(f"Đang chạy {m_name}..."):
                    mdl_name = m_name
                    last_dt = get_max_date(df)
                    model = None
                    if not force_retrain and os.path.exists(_dl_file(mdl_name)) and _load_meta(mdl_name) == str(last_dt):
                        try:
                            model = tf.keras.models.load_model(_dl_file(mdl_name))
                        except Exception:
                            model = None
                    if model is None:
                        model = creator((X.shape[1], 1)); model.fit(X, y, epochs=DL_EPOCHS, batch_size=DL_BATCH_SIZE, verbose=0, callbacks=[EarlyStopping(monitor='loss', patience=3)])
                        try:
                            model.save(_dl_file(mdl_name))
                            _save_meta(mdl_name, last_dt)
                        except Exception:
                            pass
                    curr_in = scaler.transform(df['Next_Return'].dropna().values[-LOOK_BACK_DAYS:].reshape(-1,1)).reshape(1, LOOK_BACK_DAYS, 1)
                    preds_scaled = []
                    for _ in range(n_days):
                        pr = model.predict(curr_in, verbose=0)[0,0]; preds_scaled.append(pr)
                        curr_in = np.append(curr_in[0,1:,0], pr).reshape(1, LOOK_BACK_DAYS, 1)
                    pred_ret = scaler.inverse_transform(np.array(preds_scaled).reshape(-1,1)).flatten()
                    preds, curr = [], float(df['Close'].iloc[-1])
                    for r in pred_ret:
                        curr = curr * np.exp(r/100)
                        preds.append(curr)
                    results[m_name] = {'predictions': preds}
                    is_pred_scaled = model.predict(X, verbose=0)
                    is_pred_ret = scaler.inverse_transform(is_pred_scaled).flatten()
                    idx_nr = df.dropna(subset=['Next_Return']).index
                    valid_idx = idx_nr[LOOK_BACK_DAYS:]
                    is_pred_price = df.loc[valid_idx, 'Open'].values * np.exp(is_pred_ret/100)
                    in_sample_preds[m_name] = (valid_idx, is_pred_price)
                    try:
                        st.session_state.setdefault('model_diag', [])
                        st.session_state['model_diag'] = [r for r in st.session_state['model_diag'] if r.get('Mô hình') != m_name]
                        st.session_state['model_diag'].append(summarize_lstm(DL_EPOCHS, m_name))
                    except Exception:
                        pass

    # ML
    df_ml = df.dropna(subset=['Next_Return'])
    d_feat, X_ml, y_ml = build_lag_features(df)
    last_close = float(df['Close'].iloc[-1])
    # Seed lags cho dự báo đệ quy
    seed_lags = d_feat[[f'Ret_Lag{i}' for i in range(1,6)]].iloc[-1].values.reshape(1,-1) if not d_feat.empty else np.zeros((1,5))
    for name in ['XGBoost', 'LightGBM', 'CatBoost', 'Random Forest', 'SVR']:
        if name in models_to_run: 
            with st.spinner(f"Đang chạy {name}..."):
                mdl_name = name
                last_dt = get_max_date(df)
                pipe = None
                if not force_retrain and os.path.exists(_model_file(mdl_name)) and _load_meta(mdl_name) == str(last_dt):
                    try:
                        pipe = joblib_load(_model_file(mdl_name))
                    except Exception:
                        pipe = None
                if pipe is None:
                    mod = get_ml_model(name)
                    pipe = Pipeline([('scaler', StandardScaler()), ('model', mod)]).fit(X_ml, y_ml)
                    try:
                        joblib_dump(pipe, _model_file(mdl_name))
                        _save_meta(mdl_name, last_dt)
                    except Exception:
                        pass
                # Dự báo đệ quy có kiểm soát với lag returns
                lags = seed_lags.flatten().tolist()
                preds = []; curr = last_close
                for i in range(n_days):
                    feat = np.array(lags[-5:]).reshape(1,-1)
                    ret_pred = float(pipe.predict(feat)[0])
                    # Điều chỉnh nhẹ theo độ biến động gần đây để tránh quá mượt
                    alpha = 0.8
                    if i >= 2:
                        ret_pred = alpha*ret_pred + (1-alpha)*np.mean([ret_pred, pipe.predict(np.array(lags[-5:]).reshape(1,-1))[0]])
                    curr = curr * np.exp(ret_pred / 100)
                    preds.append(curr)
                    lags.append(ret_pred)
                results[name] = {'predictions': preds}
                if 'Voting' in hybrid_types or 'Stacking' in hybrid_types:
                    in_sample_preds[name] = (d_feat.index, pipe.predict(X_ml))
                try:
                    if hasattr(pipe, 'named_steps') and 'model' in pipe.named_steps:
                        base_feats = ['Open','High','Low','Close','Volume']
                        st.session_state.setdefault('model_diag', [])
                        st.session_state['model_diag'] = [r for r in st.session_state['model_diag'] if r.get('Mô hình') != name]
                        st.session_state['model_diag'].append(summarize_xgb(pipe.named_steps['model'], base_feats, name))
                except Exception:
                    pass

    if 'Amazon Chronos (AI)' in models_to_run:
        try:
            context_df = chronos_prepare_series(df['Date'], df['Close'], freq='B', limit=128)
            
            # Giới hạn prediction_length tối đa là 14 nếu n_days > 14
            safe_n_days = min(int(n_days), 14)
            
            pred_df = chronos_predict_safe(context_df, safe_n_days, [0.5])
            
            future_dates = []
            anchor = max(pd.to_datetime(df['Date'].iloc[-1]), pd.to_datetime(datetime.today().strftime('%Y-%m-%d')))
            cur = anchor
            for _ in range(safe_n_days):
                cur += timedelta(days=1)
                while cur.weekday() >= 5:
                    cur += timedelta(days=1)
                future_dates.append(cur)
            y = chronos_align(pred_df, future_dates)
            
            # Nếu n_days thực tế lớn hơn safe_n_days, fill phần còn lại bằng giá trị cuối cùng
            if int(n_days) > safe_n_days:
                last_val = y[-1] if len(y) > 0 else 0
                extras = [last_val] * (int(n_days) - safe_n_days)
                y = np.concatenate([y, extras])
                
            results['Amazon Chronos (AI)'] = {'predictions': y.tolist() if hasattr(y, 'tolist') else list(y)}
        except Exception as e:
            st.error(f"Chronos không chạy được trong dự báo: {e}")

    # --- HYBRIDS ---
    # Prepare last_X for models using OHLCV (Hybrid LSTM+CatBoost, Stacking)
    try:
        last_X = df[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[-1:].values
    except Exception:
        last_X = np.zeros((1, 5))

    for h_type in hybrid_types:
        if h_type == 'LSTM_CatBoost':
            m1, m2 = 'LSTM', 'CatBoost'
            hybrid_name = f"Hybrid ({m1} + {m2})"
            if m1 in in_sample_preds:
                with st.spinner(f"Đang xử lý {hybrid_name}..."):
                    idx_m1, val_m1 = in_sample_preds[m1]
                    common_idx = df.index.intersection(idx_m1)
                    if len(common_idx) > 0:
                        aligned_df = df.loc[common_idx].copy()
                        if len(val_m1) == len(idx_m1):
                            s_m1 = pd.Series(val_m1, index=idx_m1)
                            aligned_df['Pred_M1'] = s_m1.reindex(common_idx)
                            aligned_df['Residual'] = aligned_df['Close'] - aligned_df['Pred_M1']
                            aligned_df = aligned_df.dropna(subset=['Residual'])
                            if not aligned_df.empty:
                                pipe_m2 = Pipeline([('scaler', StandardScaler()), ('model', get_ml_model(m2))])
                                pipe_m2.fit(aligned_df[['Open', 'High', 'Low', 'Close', 'Volume']], aligned_df['Residual'])
                                pred_m1_future = results[m1]['predictions']
                                pred_resid_t1 = pipe_m2.predict(last_X)[0]
                                hybrid_preds = []
                                for i, p_m1 in enumerate(pred_m1_future):
                                    correction = pred_resid_t1 * (0.8 ** i)
                                    hybrid_preds.append(p_m1 + correction)
                                results[hybrid_name] = {'predictions': hybrid_preds}
        
        elif h_type == 'ARIMA_LSTM':
            hybrid_name = "Hybrid (ARIMA + LSTM)"
            if 'ARIMA+GARCH' in in_sample_preds:
                with st.spinner(f"Đang xử lý {hybrid_name}..."):
                    _, _, arima_is_ret = in_sample_preds['ARIMA+GARCH']
                    if isinstance(arima_is_ret, pd.Series): arima_is_ret = arima_is_ret.values
                    actual_ret = df['Intraday_Return'].dropna().values
                    min_len = min(len(actual_ret), len(arima_is_ret))
                    resid_ret = actual_ret[-min_len:] - arima_is_ret[-min_len:]
                    X_res, y_res, scaler_res = prepare_dl_residual_data(resid_ret, LOOK_BACK_DAYS)
                    lstm_res = create_lstm_model((X_res.shape[1], 1))
                    lstm_res.fit(X_res, y_res, epochs=DL_EPOCHS, batch_size=DL_BATCH_SIZE, verbose=0, callbacks=[EarlyStopping(monitor='loss', patience=3)])
                    curr_resid = scaler_res.transform(resid_ret[-LOOK_BACK_DAYS:].reshape(-1,1)).reshape(1, LOOK_BACK_DAYS, 1)
                    pred_resid_scaled = []
                    for _ in range(n_days):
                        pr = lstm_res.predict(curr_resid, verbose=0)[0,0]; pred_resid_scaled.append(pr)
                        curr_resid = np.append(curr_resid[0,1:,0], pr).reshape(1, LOOK_BACK_DAYS, 1)
                    pred_resid_ret = scaler_res.inverse_transform(np.array(pred_resid_scaled).reshape(-1,1)).flatten()
                    arima_preds = results['ARIMA+GARCH']['predictions']
                    hybrid_preds = []
                    for i, p_arima in enumerate(arima_preds):
                        hybrid_preds.append(p_arima * np.exp(pred_resid_ret[i]/100))
                    results[hybrid_name] = {'predictions': hybrid_preds}
        
        elif h_type == 'Voting':
            hybrid_name = "Hybrid (Voting ML)"
            votes = []
            for c in ['XGBoost', 'CatBoost', 'LightGBM', 'Random Forest']:
                if c in results: votes.append(np.array(results[c]['predictions']))
            if votes:
                avg_preds = np.mean(votes, axis=0)
                results[hybrid_name] = {'predictions': avg_preds.tolist()}
        
        elif h_type == 'Stacking' and StackingRegressor is not None:
             hybrid_name = "Hybrid (Stacking ML)"
             estimators = [
                 ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
                 ('xgb', XGBRegressor(objective='reg:squarederror', verbosity=0, n_jobs=-1, random_state=42))
             ]
             stacking_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
             pipe_stack = Pipeline([('scaler', StandardScaler()), ('model', stacking_model)])
             if 'XGBoost' in in_sample_preds: 
                 df_ml = df.dropna(subset=['Next_Return'])
                 X_ml = df_ml[['Open', 'High', 'Low', 'Close', 'Volume']]; y_ml = df_ml['Next_Return']
                 pipe_stack.fit(X_ml, y_ml)
                 pred_ret_t1 = pipe_stack.predict(last_X)[0]
                 preds = []
                 curr = last_close
                 for i in range(n_days):
                     decay = pred_ret_t1 * (0.95 ** i)
                     curr = curr * np.exp(decay / 100) 
                     preds.append(curr)
                 results[hybrid_name] = {'predictions': preds}

    return results

# --- 8. RUN COMPETITION ---
def run_competition(df, models, hybrid_types=[]):
    st.info("Đang chạy Backtest (So sánh RMSE, MAE, R2)...")
    split = int(len(df)*(1-TEST_SPLIT_PCT))
    train, test = df.iloc[:split], df.iloc[split:]
    act_close = test['Close'].values
    models_to_run = set(models)
    
    if 'LSTM_CatBoost' in hybrid_types: models_to_run.add('LSTM')
    if 'ARIMA_LSTM' in hybrid_types: models_to_run.add('ARIMA+GARCH')
    if 'Voting' in hybrid_types or 'Stacking' in hybrid_types: models_to_run.update(['XGBoost', 'CatBoost', 'LightGBM', 'Random Forest'])
    
    res = []; preds_cache = {} 

    def evaluate(name, pred):
        l = min(len(act_close), len(pred))
        true = act_close[:l]; p = pred[:l]
        rmse = np.sqrt(mean_squared_error(true, p))
        r2 = r2_score(true, p)
        mae = mean_absolute_error(true, p)
        return {'Mô hình': name, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

    def metrics_from_history_all_horizons(model_name):
        try:
            hist_df = load_history(df)
            if hist_df is None or hist_df.empty: return None
            h = hist_df[hist_df['model_name'] == model_name].copy()
            if h.empty or 'predicted_for_date' not in h.columns: return None
            h['predicted_for_date'] = pd.to_datetime(h['predicted_for_date'], errors='coerce')
            date_list = df['Date'].tolist(); date_to_idx = {d: i for i, d in enumerate(date_list)}
            y_true_all = []; y_pred_all = []
            for _, r in h.iterrows():
                base = r.get('predicted_for_date')
                if pd.isna(base): continue
                idx0 = date_to_idx.get(base)
                if idx0 is None: continue
                for i in range(1, 15):
                    pred_col = f'pred_T{i}'
                    if pred_col in h.columns:
                        pv = r.get(pred_col)
                        ai = idx0 + i - 1
                        if ai < len(date_list) and pd.notna(pv):
                            y_pred_all.append(float(pv))
                            y_true_all.append(float(df.iloc[ai]['Close']))
            if len(y_true_all) == 0: return None
            y = np.array(y_true_all); p = np.array(y_pred_all)
            rmse_v = np.sqrt(mean_squared_error(y, p))
            mae_v = mean_absolute_error(y, p)
            try:
                r2_v = r2_score(y, p)
            except:
                r2_v = np.nan
            return {'Mô hình': model_name, 'RMSE': rmse_v, 'MAE': mae_v, 'R2': r2_v}
        except:
            return None

    if 'ARIMA+GARCH' in models_to_run:
        try:
            m = pm.auto_arima(train['Intraday_Return'], seasonal=False, stepwise=True, trace=False).predict(len(test))
            p = test['Open'].values * np.exp(m/100); preds_cache['ARIMA+GARCH'] = p
            res.append(evaluate('ARIMA+GARCH', p))
        except: pass
        
    if any(m in models_to_run for m in ['LSTM', 'GRU']):
        # FIX: Use Next_Return and train scaler on TRAIN only; reconstruct price from last TRAIN close
        ret_train = train['Next_Return'].dropna().values.reshape(-1,1)
        if len(ret_train) >= LOOK_BACK_DAYS + 1:
            scaler = MinMaxScaler(); s_train = scaler.fit_transform(ret_train)
            Xt, yt = [], []
            for i in range(len(s_train)-LOOK_BACK_DAYS):
                Xt.append(s_train[i:i+LOOK_BACK_DAYS,0]); yt.append(s_train[i+LOOK_BACK_DAYS,0])
            Xt = np.array(Xt).reshape(-1, LOOK_BACK_DAYS, 1); yt = np.array(yt)
            for nm, cr in [('LSTM', create_lstm_model), ('GRU', create_gru_model)]:
                if nm in models_to_run and len(Xt) > 0:
                    try:
                        md = cr((LOOK_BACK_DAYS,1)); md.fit(Xt, yt, epochs=DL_EPOCHS, verbose=0, callbacks=[EarlyStopping('loss', patience=3)])
                        last_in = s_train[-LOOK_BACK_DAYS:].reshape(1, LOOK_BACK_DAYS, 1)
                        preds_scaled = []
                        for _ in range(len(test)):
                            pr = md.predict(last_in, verbose=0)[0,0]
                            preds_scaled.append(pr)
                            last_in = np.append(last_in[0,1:,0], pr).reshape(1, LOOK_BACK_DAYS, 1)
                        pred_ret = scaler.inverse_transform(np.array(preds_scaled).reshape(-1,1)).flatten()
                        base_price = float(train['Close'].iloc[-1])
                        p = []
                        cur = base_price
                        for r in pred_ret:
                            cur = cur * np.exp(r/100)
                            p.append(cur)
                        p = np.array(p)
                        preds_cache[nm] = p
                        res.append(evaluate(nm, p))
                    except Exception as e:
                        st.warning(f"Bỏ qua {nm} do lỗi DL: {e}")

    df_ml = df.dropna(subset=['Next_Return']); split_ml = int(len(df_ml)*(1-TEST_SPLIT_PCT))
    X_ml = df_ml[['Open', 'High', 'Low', 'Close', 'Volume']]; y_ml = df_ml['Next_Return']
    Xt, Xte, yt, yte = X_ml.iloc[:split_ml], X_ml.iloc[split_ml:], y_ml.iloc[:split_ml], y_ml.iloc[split_ml:]
    
    for n in ['XGBoost', 'LightGBM', 'CatBoost', 'Random Forest', 'SVR']:
        if n in models_to_run:
            mod = get_ml_model(n)
            pipe = Pipeline([('s', StandardScaler()), ('m', mod)])
            pipe.fit(Xt, yt)
            # FIX: Recursive forecasting for ML instead of heuristic decay
            curr_close = float(train['Close'].iloc[-1])
            last_feat = X_ml.iloc[split_ml-1][['Open','High','Low','Close','Volume']].astype(float).values
            preds_seq = []
            for _ in range(len(Xte)):
                pr_ret = float(pipe.predict(last_feat.reshape(1,-1))[0])
                pr_ret = float(clip_returns([pr_ret])[0])
                curr_close = curr_close * np.exp(pr_ret/100)
                preds_seq.append(curr_close)
                # update features with predicted close for next step
                last_feat[0] = curr_close  # Open
                last_feat[1] = curr_close  # High
                last_feat[2] = curr_close  # Low
                last_feat[3] = curr_close  # Close
                # keep Volume as last known
            p_aligned = np.array(preds_seq)
            res.append(evaluate(n, p_aligned))
            preds_cache[n] = p_aligned

    if 'Amazon Chronos (AI)' in models_to_run:
        try:
            context_df = chronos_prepare_series(train['Date'], train['Close'], freq='B')
            pred_df = chronos_predict_safe(context_df, int(len(test)), [0.5])
            p = chronos_align(pred_df, test['Date'])
            res.append(evaluate('Amazon Chronos (AI)', p))
        except Exception as e:
            st.warning(f"Chronos không chạy được trong backtest: {e}")

    

    votes = {}
    for c in ['XGBoost', 'CatBoost', 'LightGBM', 'Random Forest']:
        if c in preds_cache:
            votes[c] = preds_cache[c]
    if 'Voting' in hybrid_types and votes:
        weights = None
        if os.path.exists(CV_WEIGHTS_FILE):
            try:
                weights = json.load(open(CV_WEIGHTS_FILE))
            except Exception:
                weights = None
        min_len = min(len(v) for v in votes.values())
        arr = np.vstack([votes[k][-min_len:] for k in votes.keys()])
        if weights:
            w = np.array([weights.get(k, 1.0) for k in votes.keys()], dtype=float)
            if w.sum() == 0: w = np.ones_like(w)
            avg = (arr.T @ w) / w.sum()
        else:
            avg = np.mean(arr, axis=0)
        res.append(evaluate('Hybrid (Voting ML)', avg))
    if 'Stacking' in hybrid_types and StackingRegressor is not None and len(Xt) > 0 and len(Xte) > 0:
        estimators = [('rf', RandomForestRegressor(n_estimators=50, random_state=42)), ('xgb', XGBRegressor(objective='reg:squarederror', verbosity=0, n_jobs=-1, random_state=42))]
        stacking_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
        pipe_stack = Pipeline([('s', StandardScaler()), ('m', stacking_model)])
        pipe_stack.fit(Xt, yt)
        pr_ret = pipe_stack.predict(Xte)
        p = Xte['Close'].values * np.exp(pr_ret/100)
        res.append(evaluate('Hybrid (Stacking ML)', p))
    # Explicit hybrid metrics
    if 'ARIMA_LSTM' in hybrid_types and ('ARIMA+GARCH' in preds_cache) and ('LSTM' in preds_cache):
        min_len = min(len(preds_cache['ARIMA+GARCH']), len(preds_cache['LSTM']))
        avg = (preds_cache['ARIMA+GARCH'][-min_len:] + preds_cache['LSTM'][-min_len:]) / 2.0
        res.append(evaluate('Hybrid (ARIMA + LSTM)', avg))
    if 'LSTM_CatBoost' in hybrid_types and ('LSTM' in preds_cache) and ('CatBoost' in preds_cache):
        min_len = min(len(preds_cache['LSTM']), len(preds_cache['CatBoost']))
        avg = (preds_cache['LSTM'][-min_len:] + preds_cache['CatBoost'][-min_len:]) / 2.0
        res.append(evaluate('Hybrid (LSTM + CatBoost)', avg))

    # Bổ sung chỉ số cho mô hình chưa chạy được bằng lịch sử dự báo
    present = set([r['Mô hình'] for r in res])
    for m in models_to_run:
        if m not in present:
            alt = metrics_from_history_all_horizons(m)
            if alt is not None:
                res.append(alt)
    if not res:
        st.warning("Không tính được chỉ số cho mô hình thiếu.")
        pdf = pd.DataFrame([])
    else:
        try:
            pdf = pd.DataFrame(res).sort_values('RMSE')
        except Exception:
            pdf = pd.DataFrame(res)
    try:
        with open(RMSE_FILE, 'w') as f:
            json.dump({'date': datetime.today().strftime('%Y-%m-%d'), 'results': pdf.to_dict('records')}, f)
    except Exception:
        pass
    return pdf

# --- 9. BACKTEST (Time Series Split) ---
def run_kfold_cv(df, models):
    st.info("Đang chạy Backtest (Time Series Split)...")
    tscv = TimeSeriesSplit(n_splits=5)
    df_ml = df.dropna(subset=['Next_Return'])
    X = df_ml[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = df_ml['Next_Return']
    results = []
    ml_list = [m for m in models if m in ['XGBoost', 'LightGBM', 'CatBoost', 'Random Forest', 'SVR']]
    include_chronos = 'Amazon Chronos (AI)' in models
    include_arima = 'ARIMA+GARCH' in models
    dl_list = [m for m in models if m in ['LSTM','GRU','CNN-LSTM','Informer','DeepAR']]

    progress_bar = st.progress(0)
    total_tasks = len(ml_list) + (1 if include_chronos else 0) + (1 if include_arima else 0) + len(dl_list)
    idx = 0
    for name in ml_list:
        mod = get_ml_model(name)
        pipe = Pipeline([('scaler', StandardScaler()), ('model', mod)])
        rmses, r2s = [], []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            pipe.fit(X_train, y_train)
            pred_ret = pipe.predict(X_test)
            pred_ret = clip_returns(pred_ret)
            base_price = X_train.iloc[-1]['Close']
            true_p = base_price * (1 + y_test/100)
            pred_p = base_price * (1 + pred_ret/100)
            mse = mean_squared_error(true_p, pred_p)
            rmses.append(np.sqrt(mse))
            r2s.append(r2_score(y_test, pred_ret))
        results.append({'Mô hình': name, 'CV RMSE': np.mean(rmses), 'CV R2': np.mean(r2s), 'CV STD': float(np.std(rmses)), 'Độ ổn định': 'Cao' if np.std(rmses) < 5 else 'Thấp'})
        idx += 1
        progress_bar.progress(idx / total_tasks)
    if include_chronos:
        try:
            rmses, r2s = [], []
            from chronos import ChronosPipeline as CP
            for train_idx, test_idx in tscv.split(df['Close'].values):
                train_df = df.iloc[train_idx]
                test_df = df.iloc[test_idx]
                context_df = chronos_prepare_series(train_df['Date'], train_df['Close'], freq='B')
                pred_df = chronos_predict_safe(context_df, int(len(test_df)), [0.5])
                p = chronos_align(pred_df, test_df['Date'])
                true_p = test_df['Close'].to_numpy()
                mse = mean_squared_error(true_p, p)
                rmses.append(np.sqrt(mse))
                try:
                    r2s.append(r2_score(true_p, p))
                except Exception:
                    r2s.append(np.nan)
            results.append({'Mô hình': 'Amazon Chronos (AI)', 'CV RMSE': np.mean(rmses), 'CV R2': np.nanmean(r2s), 'CV STD': float(np.std(rmses)), 'Độ ổn định': 'Cao' if np.std(rmses) < 5 else 'Thấp'})
        except Exception:
            pass
        progress_bar.progress(1.0)
    if include_arima:
        try:
            rmses, r2s = [] , []
            for train_idx, test_idx in tscv.split(df['Close'].values):
                train_df = df.iloc[train_idx]
                test_df = df.iloc[test_idx]
                try:
                    m = pm.auto_arima(train_df['Intraday_Return'], seasonal=False, stepwise=True, trace=False).predict(len(test_df))
                    p = test_df['Open'].values * np.exp(np.array(m)/100)
                    true_p = test_df['Close'].to_numpy()
                    mse = mean_squared_error(true_p, p)
                    rmses.append(np.sqrt(mse))
                    try:
                        r2s.append(r2_score(true_p, p))
                    except Exception:
                        r2s.append(np.nan)
                except Exception:
                    pass
            if len(rmses) > 0:
                results.append({'Mô hình': 'ARIMA+GARCH', 'CV RMSE': np.mean(rmses), 'CV R2': np.nanmean(r2s), 'CV STD': float(np.std(rmses)), 'Độ ổn định': 'Cao' if np.std(rmses) < 5 else 'Thấp'})
        except Exception:
            pass
        idx += 1
        progress_bar.progress(min(1.0, idx / total_tasks))
    for nm in dl_list:
        try:
            rmses, r2s = [] , []
            for train_idx, test_idx in tscv.split(df['Close'].values):
                train_df = df.iloc[train_idx]
                test_df = df.iloc[test_idx]
                p = dl_cv_predict(train_df, test_df, nm, epochs=CV_EPOCHS)
                if p is None or len(p) == 0:
                    continue
                true_p = test_df['Close'].to_numpy()[:len(p)]
                mse = mean_squared_error(true_p, p)
                rmses.append(np.sqrt(mse))
                try:
                    r2s.append(r2_score(true_p, p))
                except Exception:
                    r2s.append(np.nan)
            if len(rmses) > 0:
                results.append({'Mô hình': nm, 'CV RMSE': np.mean(rmses), 'CV R2': np.nanmean(r2s), 'CV STD': float(np.std(rmses)), 'Độ ổn định': 'Cao' if np.std(rmses) < 5 else 'Thấp'})
        except Exception:
            pass
        idx += 1
        progress_bar.progress(min(1.0, idx / total_tasks))
    res_df = pd.DataFrame(results).sort_values('CV RMSE')
    st.table(res_df.style.format({'CV RMSE': '{:.2f}', 'CV R2': '{:.4f}'}))
    weights = {row['Mô hình']: float(1.0/max(row['CV RMSE'], 1e-6)) for _, row in res_df.iterrows()}
    s = sum(weights.values())
    if s > 0: weights = {k: v/s for k, v in weights.items()}
    try:
        with open(CV_WEIGHTS_FILE, 'w') as f:
            json.dump(weights, f)
        meta = {'date': datetime.today().strftime('%Y-%m-%d %H:%M'), 'results': res_df.to_dict('records')}
        with open(BACKTEST_METRICS_FILE, 'w') as f:
            json.dump(meta, f)
        try:
            os.makedirs('history_logs', exist_ok=True)
            ts = datetime.today().strftime('%Y-%m-%d')
            with open(os.path.join('history_logs', f'backtest_metrics_{ts}.json'), 'w') as h:
                json.dump(meta, h)
        except Exception:
            pass
    except Exception:
        pass

# --- 9b. UPDATE MISSING METRICS ---
def update_missing_rmse(df, models, hybrid_types=[]):
    try:
        existing = []
        if os.path.exists(RMSE_FILE):
            existing = json.load(open(RMSE_FILE)).get('results', [])
        old_df = pd.DataFrame(existing)
        have = set(old_df['Mô hình'].tolist()) if not old_df.empty else set()
        targets = set(models)
        base_needed = set()
        if 'LSTM_CatBoost' in hybrid_types:
            targets.add('Hybrid (LSTM + CatBoost)')
            base_needed.update(['LSTM','CatBoost'])
        if 'ARIMA_LSTM' in hybrid_types:
            targets.add('Hybrid (ARIMA + LSTM)')
            base_needed.update(['ARIMA+GARCH','LSTM'])
        if 'Voting' in hybrid_types: targets.add('Hybrid (Voting ML)')
        if 'Stacking' in hybrid_types: targets.add('Hybrid (Stacking ML)')
        # Hỗ trợ TimeGPT qua lịch sử dự báo nếu có
        unsupported = set()
        targets.update(base_needed)
        missing = [m for m in targets if m not in have]
        if not missing:
            st.toast("Không có mô hình nào thiếu chỉ số.")
            return old_df
        if unsupported:
            st.warning(f"Bỏ qua mô hình không hỗ trợ: {', '.join(sorted(list(unsupported)))}")
        st.info(f"Bổ sung chỉ số cho {len(missing)} mô hình: {', '.join(missing)}")
        split = int(len(df)*(1-TEST_SPLIT_PCT))
        train, test = df.iloc[:split], df.iloc[split:]
        act_close = test['Close'].values
        res = []; preds_cache = {}
        def evaluate(name, pred):
            l = min(len(act_close), len(pred)); true = act_close[:l]; p = pred[:l]
            rmse = np.sqrt(mean_squared_error(true, p)); r2 = r2_score(true, p); mae = mean_absolute_error(true, p)
            return {'Mô hình': name, 'RMSE': rmse, 'MAE': mae, 'R2': r2}
        # ARIMA
        if 'ARIMA+GARCH' in targets:
            try:
                m = pm.auto_arima(train['Intraday_Return'], seasonal=False, stepwise=True, trace=False).predict(len(test))
                p = test['Open'].values * np.exp(m/100); preds_cache['ARIMA+GARCH'] = p
                if 'ARIMA+GARCH' in missing: res.append(evaluate('ARIMA+GARCH', p))
            except: pass
        # DL block
        dl_names = ['LSTM','GRU','CNN-LSTM','Informer','DeepAR']
        if any(n in targets for n in dl_names):
            scaler = MinMaxScaler(); all_s = scaler.fit_transform(df['Intraday_Return'].dropna().values.reshape(-1,1))
            X, y = [], []
            for i in range(len(all_s)-LOOK_BACK_DAYS): X.append(all_s[i:i+60,0]); y.append(all_s[i+60,0])
            X, y = np.array(X).reshape(-1,60,1), np.array(y)
            total_len = len(X); test_len = len(test); train_len = total_len - test_len
            if train_len > 0:
                Xt, yt = X[:train_len], y[:train_len]; Xtest = X[train_len:]
            else:
                Xt, yt, Xtest = None, None, None
            for nm, cr in [('LSTM', create_lstm_model), ('GRU', create_gru_model), ('CNN-LSTM', create_cnn_lstm_model), ('Informer', create_informer_like_model), ('DeepAR', create_gru_model)]:
                if nm in targets:
                    if Xt is None or Xtest is None or len(Xtest) == 0:
                        st.warning(f"Bỏ qua {nm} do dữ liệu không đủ cho khối DL.")
                    else:
                        try:
                            md = cr((60,1))
                            md.fit(Xt, yt, epochs=DL_EPOCHS, verbose=0, callbacks=[EarlyStopping('loss', patience=3)])
                            pr = md.predict(Xtest, verbose=0).flatten()
                            pr = scaler.inverse_transform(pr.reshape(-1,1)).flatten()
                            pr = clip_returns(pr)
                            p = test['Open'].values[-len(pr):] * np.exp(pr/100)
                            preds_cache[nm] = p
                            if nm in missing: res.append(evaluate(nm, p))
                        except Exception as e:
                            st.warning(f"Bỏ qua {nm} do lỗi DL: {e}")
        # ML block
        df_ml = df.dropna(subset=['Next_Return']); split_ml = int(len(df_ml)*(1-TEST_SPLIT_PCT))
        X_ml = df_ml[['Open', 'High', 'Low', 'Close', 'Volume']]; y_ml = df_ml['Next_Return']
        Xt, Xte, yt, yte = X_ml.iloc[:split_ml], X_ml.iloc[split_ml:], y_ml.iloc[:split_ml], y_ml.iloc[split_ml:]
        for n in ['XGBoost','LightGBM','CatBoost','Random Forest','SVR']:
            if n in targets:
                mod = get_ml_model(n)
                pipe = Pipeline([('s', StandardScaler()), ('m', mod)]).fit(Xt, yt)
                pr_ret = pipe.predict(Xte)
                pr_ret = clip_returns(pr_ret)
                pred_price = Xte['Close'].values * np.exp(pr_ret/100)
                l = min(len(act_close), len(pred_price)); p_aligned = pred_price[-l:]
                if len(p_aligned) > 0:
                    preds_cache[n] = p_aligned
                    if n in missing: res.append(evaluate(n, p_aligned))
        # Hybrids
        votes = {}
        for c in ['XGBoost','CatBoost','LightGBM','Random Forest']:
            if c in preds_cache: votes[c] = preds_cache[c]
        if 'Hybrid (Voting ML)' in targets and votes:
            weights = None
            if os.path.exists(CV_WEIGHTS_FILE):
                try: weights = json.load(open(CV_WEIGHTS_FILE))
                except Exception: weights = None
            min_len = min(len(v) for v in votes.values())
            arr = np.vstack([votes[k][-min_len:] for k in votes.keys()])
            if weights:
                w = np.array([weights.get(k, 1.0) for k in votes.keys()], dtype=float)
                if w.sum() == 0: w = np.ones_like(w)
                avg = (arr.T @ w) / w.sum()
            else:
                avg = np.mean(arr, axis=0)
            if 'Hybrid (Voting ML)' in missing: res.append(evaluate('Hybrid (Voting ML)', avg))
        # Hybrid ARIMA + LSTM trung bình đơn giản nếu cả hai sẵn sàng
        if 'Hybrid (ARIMA + LSTM)' in targets and ('ARIMA+GARCH' in preds_cache) and ('LSTM' in preds_cache):
            min_len = min(len(preds_cache['ARIMA+GARCH']), len(preds_cache['LSTM']))
            avg = (preds_cache['ARIMA+GARCH'][-min_len:] + preds_cache['LSTM'][-min_len:]) / 2.0
            if 'Hybrid (ARIMA + LSTM)' in missing: res.append(evaluate('Hybrid (ARIMA + LSTM)', avg))
        # Hybrid LSTM + CatBoost nếu cả hai sẵn sàng
        if 'Hybrid (LSTM + CatBoost)' in targets and ('LSTM' in preds_cache) and ('CatBoost' in preds_cache):
            min_len = min(len(preds_cache['LSTM']), len(preds_cache['CatBoost']))
            avg = (preds_cache['LSTM'][-min_len:] + preds_cache['CatBoost'][-min_len:]) / 2.0
            if 'Hybrid (LSTM + CatBoost)' in missing: res.append(evaluate('Hybrid (LSTM + CatBoost)', avg))
        if 'Hybrid (Stacking ML)' in targets and StackingRegressor is not None and len(Xt) > 0 and len(Xte) > 0:
            estimators = [('rf', RandomForestRegressor(n_estimators=50, random_state=42)), ('xgb', XGBRegressor(objective='reg:squarederror', verbosity=0, n_jobs=-1, random_state=42))]
            stacking_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
            pipe_stack = Pipeline([('s', StandardScaler()), ('m', stacking_model)])
            pipe_stack.fit(Xt, yt)
            pr_ret = pipe_stack.predict(Xte)
            p = Xte['Close'].values * np.exp(pr_ret/100)
            if 'Hybrid (Stacking ML)' in missing: res.append(evaluate('Hybrid (Stacking ML)', p))

        # Amazon Chronos (AI)
        if 'Amazon Chronos (AI)' in targets and 'Amazon Chronos (AI)' in missing:
            try:
                context_df = chronos_prepare_series(train['Date'], train['Close'], freq='B')
                pred_df = chronos_predict_safe(context_df, int(len(test)), [0.5])
                p = chronos_align(pred_df, test['Date'])
                res.append(evaluate('Amazon Chronos (AI)', p))
            except Exception:
                pass
        add_df = pd.DataFrame(res)
        if add_df.empty:
            st.warning("Không tính được chỉ số cho mô hình thiếu.")
            return old_df
        merged = pd.concat([old_df[~old_df['Mô hình'].isin(add_df['Mô hình'].tolist())], add_df], ignore_index=True)
        merged = merged.sort_values('RMSE')
        meta = {'date': datetime.today().strftime('%Y-%m-%d'), 'results': merged.to_dict('records')}
        with open(RMSE_FILE, 'w') as f:
            json.dump(meta, f)
        try:
            os.makedirs('history_logs', exist_ok=True)
            ts = datetime.today().strftime('%Y-%m-%d')
            with open(os.path.join('history_logs', f'in_sample_rmse_{ts}.json'), 'w') as h:
                json.dump(meta, h)
        except Exception:
            pass
        st.success("Đã bổ sung chỉ số cho mô hình thiếu.")
        return merged
    except Exception as e:
        st.error(f"Lỗi khi bổ sung chỉ số: {e}")
        return None

# --- 10. HISTORY & UI ---
def _init_db():
    try:
        conn = sqlite3.connect(HISTORY_DB)
        conn.execute("PRAGMA journal_mode=WAL;")
        # Create table with full schema if not exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                date_predicted TEXT,
                model_name TEXT,
                predicted_for_date TEXT,
                actual_close REAL,
                pred_T1 REAL, pred_T2 REAL, pred_T3 REAL, pred_T4 REAL, pred_T5 REAL,
                pred_T6 REAL, pred_T7 REAL, pred_T8 REAL, pred_T9 REAL, pred_T10 REAL,
                pred_T11 REAL, pred_T12 REAL, pred_T13 REAL, pred_T14 REAL
            )
        """)
        
        # Schema migration: check for missing columns (e.g. T6..T14 for old DBs)
        try:
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(predictions)")
            existing_cols = [r[1] for r in cur.fetchall()]
            for i in range(1, 15):
                cname = f'pred_T{i}'
                if cname not in existing_cols:
                    try:
                        conn.execute(f"ALTER TABLE predictions ADD COLUMN {cname} REAL")
                    except Exception:
                        pass
        except Exception:
            pass

        conn.execute("CREATE INDEX IF NOT EXISTS idx_pred_k1 ON predictions(model_name, date_predicted)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pred_k2 ON predictions(model_name, predicted_for_date)")
        conn.close()
    except Exception as e:
        # st.error(f"DB Init Error: {e}")
        pass

def save_history(results, next_days, df):
    rows = []
    run_date = datetime.today().strftime('%Y-%m-%d')
    pred_date = next_days[0].strftime('%Y-%m-%d')
    if isinstance(results, dict):
        for m, r in results.items():
            if not r or 'predictions' not in r: continue
            row = {
                'date_predicted': run_date,
                'model_name': m,
                'predicted_for_date': pred_date,
                'actual_close': np.nan
            }
            for i, p in enumerate(r['predictions']):
                if i < 14: row[f'pred_T{i+1}'] = p
            rows.append(row)
    if len(rows) == 0:
        st.warning("Không có kết quả để lưu lịch sử.")
        return pd.DataFrame()
    new_df = pd.DataFrame(rows)
    os.makedirs('history_logs', exist_ok=True)
    per_day_path = os.path.join('history_logs', f'prediction_{run_date}.csv')
    new_df.to_csv(per_day_path, mode='w', header=True, index=False)
    try:
        _init_db()
        conn = sqlite3.connect(HISTORY_DB)
        conn.execute("BEGIN")
        for _, r in new_df.iterrows():
            # Xóa trùng theo (model_name, date_predicted) hoặc (model_name, predicted_for_date)
            conn.execute("DELETE FROM predictions WHERE model_name=? AND date_predicted=?", (str(r['model_name']), str(r['date_predicted'])))
            conn.execute("DELETE FROM predictions WHERE model_name=? AND predicted_for_date=?", (str(r['model_name']), str(r['predicted_for_date'])))
            cols = ['date_predicted','model_name','predicted_for_date','actual_close'] + [f'pred_T{i}' for i in range(1,15)]
            vals = [r.get('date_predicted'), r.get('model_name'), r.get('predicted_for_date'), r.get('actual_close')] + [r.get(f'pred_T{i}') for i in range(1,15)]
            placeholders = ",".join(["?"]*len(cols))
            conn.execute(f"INSERT INTO predictions ({','.join(cols)}) VALUES ({placeholders})", vals)
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Lỗi lưu lịch sử vào SQLite: {e}")
        # Không pass nữa để debug
    try:
        rows_by_date = []
        for _, r in new_df.iterrows():
            base = pd.to_datetime(r.get('predicted_for_date'), errors='coerce')
            td = [base] + bdays_from(base, 13) if pd.notna(base) else []
            for i in range(1, 15):
                d = td[i-1] if i-1 < len(td) else None
                pred = r.get(f'pred_T{i}')
                rows_by_date.append({
                    'date_predicted': run_date,
                    'model_name': r.get('model_name'),
                    'date': d.strftime('%Y-%m-%d') if d is not None else f'T{i}',
                    'predicted': pred
                })
        df_by_date = pd.DataFrame(rows_by_date)
        by_date_path = os.path.join('history_logs', f'forecast_by_date_{run_date}.csv')
        df_by_date.to_csv(by_date_path, mode='w', header=True, index=False)
    except Exception:
        pass


def load_history(df):
    if not os.path.exists(HISTORY_DB): 
        return pd.DataFrame()
    try:
        _init_db()
        conn = sqlite3.connect(HISTORY_DB)
        hist = pd.read_sql_query("SELECT * FROM predictions", conn)
        conn.close()
        if 'date_predicted' not in hist.columns and 'predicted_for_date' in hist.columns:
            hist['date_predicted'] = hist['predicted_for_date']
        hist['date_predicted'] = pd.to_datetime(hist['date_predicted'], errors='coerce')
        hist['predicted_for_date'] = pd.to_datetime(hist['predicted_for_date'], errors='coerce')
        df_norm = df.copy(); df_norm['Date'] = pd.to_datetime(df_norm['Date']).dt.normalize()
        lookup = df_norm.set_index('Date')['Close']
        if 'actual_close' in hist.columns:
            mask = hist['actual_close'].isna()
            for i in hist[mask].index:
                d = pd.to_datetime(hist.loc[i, 'predicted_for_date'], errors='coerce')
                if pd.notna(d):
                    dn = d.normalize()
                    if dn in lookup.index:
                        hist.loc[i, 'actual_close'] = lookup[dn]
        return hist
    except: return pd.DataFrame()

def calculate_hist_perf(hist_df, df=None, horizon=1):
    if hist_df.empty:
        st.info("Chưa có dữ liệu lịch sử.")
        return
    col_pred = f'pred_T{int(horizon)}'
    col_actual = f'actual_T{int(horizon)}'
    if df is not None and col_actual not in hist_df.columns and 'predicted_for_date' in hist_df.columns:
        df_dates = pd.to_datetime(df['Date'])
        date_to_idx = {d: i for i, d in enumerate(df_dates.tolist())}
        def get_actual_for_row(row):
            try:
                d = pd.to_datetime(row['predicted_for_date'], errors='coerce')
                if pd.isna(d): return np.nan
                idx = date_to_idx.get(d)
                if idx is None: return np.nan
                ai = idx + int(horizon) - 1
                if ai < len(df_dates):
                    return float(df.iloc[ai]['Close'])
            except:
                return np.nan
            return np.nan
        hist_df[col_actual] = hist_df.apply(get_actual_for_row, axis=1)
    cl = hist_df.dropna(subset=[col_pred, col_actual])
    if cl.empty:
        st.info("Chưa có dữ liệu đối chiếu cho horizon đã chọn.")
        return
    p = []
    for m in cl['model_name'].unique():
        sub = cl[cl['model_name']==m]
        rmse = np.sqrt(mean_squared_error(sub[col_actual], sub[col_pred]))
        mae = mean_absolute_error(sub[col_actual], sub[col_pred])
        try:
            r2 = r2_score(sub[col_actual], sub[col_pred])
        except:
            r2 = np.nan
        if pd.isna(r2) or not np.isfinite(r2):
            try:
                if len(sub) >= 2:
                    corr = np.corrcoef(sub[col_actual], sub[col_pred])[0,1]
                    r2 = float(corr*corr)
                else:
                    r2 = 0.0
            except:
                r2 = 0.0
        p.append({'Mô hình': m, 'RMSE': rmse, 'MAE': mae, 'R2': r2})
    df_p = pd.DataFrame(p).sort_values('RMSE')
    if 'R2' in df_p.columns:
        df_p['R2'] = pd.to_numeric(df_p['R2'], errors='coerce').fillna(0.0)
    df_p.index = np.arange(1, len(df_p) + 1)
    st.dataframe(
        df_p,
        column_config={
            "RMSE": st.column_config.NumberColumn(format="%.2f"),
            "MAE": st.column_config.NumberColumn(format="%.2f"),
            "R2": st.column_config.NumberColumn(format="%.4f")
        },
        width='stretch'
    )

def get_recommendation(pct):
    if pct > 1.5: return "MUA MẠNH", "success"
    elif pct > 0.3: return "MUA", "success"
    elif pct < -1.5: return "BÁN MẠNH", "error"
    elif pct < -0.3: return "BÁN", "error"
    else: return "NẮM GIỮ", "warning"

# --- MAIN UI ---
st.sidebar.header("🔧 Điều khiển")
# Nhóm input quan trọng
models = st.sidebar.multiselect("Chọn Mô hình:", ['ARIMA+GARCH', 'LSTM', 'GRU', 'CNN-LSTM', 'Informer', 'DeepAR', 'Amazon Chronos (AI)', 'XGBoost', 'LightGBM', 'CatBoost', 'Random Forest', 'SVR'], default=['ARIMA+GARCH', 'XGBoost'])
hybrid_labels = st.sidebar.multiselect("Chế độ Hybrid:", ["LSTM + CatBoost", "ARIMA + LSTM", "Voting", "Stacking"])
hybrid_map = {"LSTM + CatBoost": "LSTM_CatBoost", "ARIMA + LSTM": "ARIMA_LSTM", "Voting": "Voting", "Stacking": "Stacking"}
hybrid_types = [hybrid_map[l] for l in hybrid_labels]
days = st.sidebar.slider("Ngày chạy dự báo :", 1, 14, 5)
force_retrain = st.sidebar.checkbox("Force Retrain (huấn luyện lại)", value=False)
use_csv_only = st.sidebar.checkbox("Chỉ dùng CSV (không gọi API)", value=True)
show_model_prices = st.sidebar.checkbox("Hiển thị đường giá dự báo", value=False)

# Nút hành động chính nổi bật
cta_run = st.sidebar.button("🚀 CHẠY DỰ BÁO NGAY", type="primary")

st.sidebar.markdown("---")
with st.sidebar.expander("⚙️ Quản lý Dữ liệu & Chỉ số", expanded=False):
    if st.button("📥 Cập nhật Vnstock"):
        try:
            with st.spinner("Đang cập nhật dữ liệu từ Vnstock..."):
                try:
                    prev_df = pd.read_csv(CSV_DATA_FILE)
                    prev_df['Date'] = pd.to_datetime(prev_df['Date'], errors='coerce')
                    prev_last = prev_df['Date'].max()
                except Exception:
                    prev_last = None
                _df_tmp, _msg_tmp = load_data_internal(CSV_DATA_FILE, mtime=None, force=True)
                st.cache_data.clear(); st.cache_resource.clear()
                if _df_tmp is not None:
                    try:
                        new_last = pd.to_datetime(_df_tmp['Date'], errors='coerce').max()
                        added = int((_df_tmp['Date'] > prev_last).sum()) if prev_last is not None else 0
                        st.toast(f"Đã cập nhật {added} dòng. Max date: {new_last.strftime('%d/%m/%Y')}")
                    except Exception:
                        if _msg_tmp: st.toast(_msg_tmp)
                elif _msg_tmp:
                    st.toast(_msg_tmp)
        except Exception:
            pass
        st.rerun()
    up_file = st.file_uploader("Tải CSV VN-INDEX để cập nhật", type=["csv"], key="upload_vnindex")
    if up_file is not None:
        try:
            df_up = pd.read_csv(up_file)
            cols_norm = [str(c).strip().strip('"').strip("'") for c in df_up.columns]; df_up.columns = cols_norm
            cmap = {'ngày':'Date','date':'Date','lần cuối':'Close','close':'Close','đóng cửa':'Close','mở':'Open','open':'Open','cao':'High','high':'High','thấp':'Low','low':'Low','kl':'Volume','khối lượng':'Volume','volume':'Volume','% thay đổi':'PctChange','pct change':'PctChange','% change':'PctChange'}
            df_up = df_up.rename(columns=lambda c: cmap.get(c.lower(), c))
            df_up['Date'] = pd.to_datetime(df_up['Date'], format='mixed', dayfirst=True, errors='coerce')
            for c in ['Open','High','Low','Close','Volume','PctChange']:
                if c in df_up.columns: df_up[c] = pd.to_numeric(clean_series_to_float(df_up[c]), errors='coerce')
            needed = ['Date','Open','High','Low','Close','Volume']
            df_up = df_up.dropna(subset=['Date']).sort_values('Date')
            df_up = df_up[[c for c in needed if c in df_up.columns]]
            prev = pd.read_csv(CSV_DATA_FILE)
            prev['Date'] = pd.to_datetime(prev['Date'], errors='coerce')
            merged = pd.concat([prev, df_up], ignore_index=True).sort_values('Date').drop_duplicates(subset=['Date'], keep='last')
            merged.to_csv(CSV_DATA_FILE, index=False)
            st.toast(f"Đã ghi {len(merged) - len(prev)} dòng mới vào 15_VNINDEX.csv. Max date: {pd.to_datetime(merged['Date']).max().strftime('%d/%m/%Y')}")
            st.cache_data.clear(); st.cache_resource.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Không thể cập nhật từ file CSV: {e}")
    if st.button("🔄 Xóa Cache"):
        st.cache_data.clear(); st.cache_resource.clear()
        if os.path.exists(RMSE_FILE): os.remove(RMSE_FILE)
        st.rerun()
    if st.button("➕ Bổ sung chỉ số mô hình thiếu"):
        if 'df' in locals():
            update_missing_rmse(df, models, hybrid_types)
    if st.button("🧮 Làm mới chỉ số toàn bộ (RMSE/MAE/R2)"):
        if 'df' in locals():
            run_competition(df, models, hybrid_types)
    try:
        run_day = datetime.today().strftime('%Y-%m-%d')
        txt_content = ""
        if os.path.exists('models_metrics.txt'):
            try:
                txt_content = open('models_metrics.txt', 'r', encoding='utf-8').read()
            except Exception:
                txt_content = ""
        elif os.path.exists(RMSE_FILE):
            try:
                r_df_tmp = pd.DataFrame(json.load(open(RMSE_FILE)).get('results', []))
                lines = []
                lines.append("Mo hinh | RMSE | MAE | R2")
                for _, rr in r_df_tmp.iterrows():
                    lines.append(f"{rr.get('Mô hình','-')} | {fmt_num(rr.get('RMSE'),2)} | {fmt_num(rr.get('MAE'),2)} | {fmt_num(rr.get('R2'),4)}")
                txt_content = "\n".join(lines)
            except Exception:
                txt_content = ""
        if txt_content:
            st.download_button("📄 Tải danh sách mô hình & độ đo (.txt)", data=txt_content.encode('utf-8'), file_name=f"models_metrics_{run_day}.txt", mime="text/plain", key="download_metrics_txt")
        else:
            st.info("Chưa có dữ liệu để tải danh sách mô hình & độ đo.")
    except Exception:
        pass

st.sidebar.markdown("---")

mtime = os.path.getmtime(CSV_DATA_FILE) if os.path.exists(CSV_DATA_FILE) else None
df, msg = load_data_internal(CSV_DATA_FILE, mtime, prefer_csv_only=use_csv_only)
if df is None:
    st.error(msg or "Không thể tải dữ liệu.")
    st.info("Vui lòng bấm nút Chạy Dự Báo hoặc Backtest bên Sidebar để bắt đầu")
    st.stop()
if msg: st.toast(msg)

# Auto cập nhật CSV khi đang bật chế độ "Chỉ dùng CSV" nhưng dữ liệu không phải mới nhất
try:
    if use_csv_only and df is not None:
        today_anchor = pd.to_datetime(datetime.today().strftime('%Y-%m-%d'))
        last_data_date = pd.to_datetime(df['Date'], errors='coerce').max()
        if pd.notna(last_data_date) and last_data_date.normalize() < today_anchor:
            cooldown_sec = 120
            last_fail = st.session_state.get('api_fail_ts')
            if not last_fail or (time.time() - last_fail) >= cooldown_sec:
                with st.spinner("Đang kiểm tra dữ liệu VN-INDEX mới..."):
                    _df_auto, _msg_auto = load_data_internal(CSV_DATA_FILE, mtime=None, force=True, prefer_csv_only=True)
                    if _df_auto is not None:
                        try:
                            prev_last = pd.to_datetime(df['Date'], errors='coerce').max()
                            added_rows = int((_df_auto['Date'] > prev_last).sum())
                        except Exception:
                            added_rows = int(len(_df_auto) - len(df)) if df is not None else 0
                        df = _df_auto
                        st.cache_data.clear(); st.cache_resource.clear()
                        try:
                            last_written = pd.to_datetime(df['Date'].iloc[-1]).strftime('%d/%m/%Y')
                            msg_show = _msg_auto or "Đã kiểm tra dữ liệu"
                            if added_rows > 0:
                                st.toast(f"{msg_show} (+{added_rows} dòng mới, max date: {last_written})")
                            else:
                                st.toast(f"{msg_show} (không có dòng mới, max date: {last_written})")
                        except Exception:
                            if _msg_auto: st.toast(_msg_auto)
                        try:
                            if added_rows == 0 and (last_data_date.normalize() < today_anchor) and isinstance(_msg_auto, str) and ("Không thể lấy dữ liệu VN-INDEX" in _msg_auto):
                                st.session_state['api_fail_ts'] = time.time()
                        except Exception:
                            pass
except Exception:
    pass

if not use_csv_only:
    try:
        with st.spinner("Đang kiểm tra dữ liệu VN-INDEX mới..."):
            _df_auto, _msg_auto = load_data_internal(CSV_DATA_FILE, mtime=None, force=True)
            if _df_auto is not None:
                try:
                    prev_last = pd.to_datetime(df['Date'], errors='coerce').max()
                    added_rows = int((_df_auto['Date'] > prev_last).sum())
                except Exception:
                    added_rows = int(len(_df_auto) - len(df)) if df is not None else 0
                df = _df_auto
                st.cache_data.clear(); st.cache_resource.clear()
                try:
                    last_written = pd.to_datetime(df['Date'].iloc[-1]).strftime('%d/%m/%Y')
                    msg_show = _msg_auto or "Đã kiểm tra dữ liệu"
                    if added_rows > 0:
                        st.toast(f"{msg_show} (+{added_rows} dòng mới, max date: {last_written})")
                    else:
                        st.toast(f"{msg_show} (không có dòng mới, max date: {last_written})")
                except Exception:
                    if _msg_auto: st.toast(_msg_auto)
    except Exception:
        pass

if df is not None:
    hist = load_history(df)
    next_dates = []
    today_anchor = pd.to_datetime(datetime.today().strftime('%Y-%m-%d'))
    last_data_date = pd.to_datetime(df['Date'], errors='coerce').max()
    curr = max(last_data_date, today_anchor)
    for _ in range(days):
        curr += timedelta(days=1)
        while curr.weekday() >= 5:
            curr += timedelta(days=1)
        next_dates.append(curr)

    if cta_run:
        try:
            st.toast("Bắt đầu chạy dự báo...")
        except Exception:
            pass
        if os.path.exists(RMSE_FILE): os.remove(RMSE_FILE)
        run_competition(df, models, hybrid_types)
        res = run_all_forecasts(df, models, days, hybrid_types, force_retrain=force_retrain)
        save_history(res, next_dates, df)
        try:
            st.session_state['just_ran'] = True
        except Exception:
            pass
        st.rerun()
        
with st.sidebar.expander("📊 Backtest", expanded=False):
    all_models_avail = ['ARIMA+GARCH', 'LSTM', 'GRU', 'CNN-LSTM', 'Informer', 'DeepAR', 'Amazon Chronos (AI)', 'XGBoost', 'LightGBM', 'CatBoost', 'Random Forest', 'SVR']
    if 'LSTM_CatBoost' in hybrid_types: all_models_avail.append("Hybrid (LSTM + CatBoost)")
    if 'ARIMA_LSTM' in hybrid_types: all_models_avail.append("Hybrid (ARIMA + LSTM)")
    if 'Voting' in hybrid_types: all_models_avail.append("Hybrid (Voting ML)")
    if 'Stacking' in hybrid_types: all_models_avail.append("Hybrid (Stacking ML)")
    with st.form("backtest_form"):
        picks = st.multiselect("Chọn mô hình để backtest:", all_models_avail, default=all_models_avail)
        mode = st.radio("Kiểu backtest:", ["Chia Train/Test (tất cả)", "TimeSeriesSplit (ML)"])
        run_bt = st.form_submit_button("🔬 Chạy Backtest đã chọn")
        if run_bt:
            pick_hybrids = []
            if any("Hybrid (LSTM + CatBoost)" == p for p in picks): pick_hybrids.append('LSTM_CatBoost')
            if any("Hybrid (ARIMA + LSTM)" == p for p in picks): pick_hybrids.append('ARIMA_LSTM')
            if any("Hybrid (Voting ML)" == p for p in picks): pick_hybrids.append('Voting')
            if any("Hybrid (Stacking ML)" == p for p in picks): pick_hybrids.append('Stacking')
            if mode.startswith("Chia"):
                run_competition(df, picks, pick_hybrids)
            else:
                run_kfold_cv(df, picks)
                update_missing_rmse(df, picks, pick_hybrids)

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Dự báo & Khuyến nghị", "🏆 Hiệu suất & Chỉ số", "📝 Dữ liệu chi tiết"])

with tab1:
        today_str = datetime.today().strftime('%Y-%m-%d')
        if st.session_state.get('just_ran'):
            st.success("Đã cập nhật dự báo và lịch sử.")
            st.session_state['just_ran'] = False
        display_models = list(models)
        if 'LSTM_CatBoost' in hybrid_types: display_models.append("Hybrid (LSTM + CatBoost)")
        if 'ARIMA_LSTM' in hybrid_types: display_models.append("Hybrid (ARIMA + LSTM)")
        if 'Voting' in hybrid_types: display_models.append("Hybrid (Voting ML)")
        if 'Stacking' in hybrid_types: display_models.append("Hybrid (Stacking ML)")
        
        current_preds = pd.DataFrame()
        if not hist.empty:
            hist['date_predicted'] = pd.to_datetime(hist['date_predicted'], errors='coerce')
            hist['predicted_for_date'] = pd.to_datetime(hist['predicted_for_date'], errors='coerce')
            today_dt = pd.to_datetime(today_str)
            current_preds = hist[(hist['date_predicted'].dt.date == today_dt.date()) & (hist['model_name'].isin(display_models))]
            if current_preds.empty:
                last_avail = hist[hist['model_name'].isin(display_models)]['date_predicted'].max()
                if pd.notna(last_avail):
                    current_preds = hist[(hist['date_predicted'] == last_avail) & (hist['model_name'].isin(display_models))]

        st.subheader("Kết quả Dự báo & Khuyến nghị")
        
        if not current_preds.empty:
            backtest_text = {}
            if os.path.exists(BACKTEST_METRICS_FILE):
                try:
                    cv_data = json.load(open(BACKTEST_METRICS_FILE))
                    for rec in cv_data.get('results', []):
                        mname = rec.get('Mô hình')
                        rmse = rec.get('CV RMSE')
                        if mname is not None and rmse is not None:
                            backtest_text[mname] = f"RMSE: {rmse:.2f}"
                except Exception as e:
                    print(f"[Backtest metrics read error] {e}")
                    st.info("Chưa có dữ liệu kiểm thử, vui lòng bấm chạy")
                    backtest_text = {}
            else:
                st.info("Chưa có dữ liệu kiểm thử, vui lòng bấm chạy")
            rows = []
            for _, row in current_preds.iterrows():
                delta = row['pred_T1'] - df['Close'].iloc[-1]
                pct = (delta/df['Close'].iloc[-1])*100
                rec_text, _ = get_recommendation(pct)
                reliability = backtest_text.get(row['model_name'], 'Chưa kiểm thử')
                try:
                    run_dt = pd.to_datetime(row.get('date_predicted'), errors='coerce')
                    pred_dt = pd.to_datetime(row.get('predicted_for_date'), errors='coerce')
                    if pd.notna(run_dt):
                        run_str = run_dt.strftime('%d/%m/%Y %H:%M')
                    else:
                        rv = row.get('date_predicted')
                        run_str = str(rv) if rv is not None and str(rv).strip() else '-'
                    if pd.notna(pred_dt):
                        pred_str = pred_dt.strftime('%d/%m/%Y')
                    else:
                        pv = row.get('predicted_for_date')
                        pred_str = str(pv) if pv is not None and str(pv).strip() else '-'
                except Exception as e:
                    print(f"[Date format error] {e}")
                    rv = row.get('date_predicted'); pv = row.get('predicted_for_date')
                    run_str = str(rv) if rv is not None and str(rv).strip() else '-'
                    pred_str = str(pv) if pv is not None and str(pv).strip() else '-'
                rows.append({'Ngày chạy': run_str, 'Mô hình': row['model_name'], 'Dự báo T1': row['pred_T1'], 'Δ': delta, '%': pct, 'Khuyến nghị': rec_text, 'Độ tin cậy (Backtest)': reliability})
            df_summary = pd.DataFrame(rows).sort_values('%', ascending=False)
            df_summary.index = np.arange(1, len(df_summary) + 1)
            df_summary.index.name = 'STT'
            df_summary_view = df_summary.copy()
            if 'Dự báo T1' in df_summary_view.columns:
                df_summary_view['Dự báo T1'] = df_summary_view['Dự báo T1'].apply(lambda x: fmt_num(x, 2))
            if 'Δ' in df_summary_view.columns:
                df_summary_view['Δ'] = df_summary_view['Δ'].apply(lambda x: fmt_num(x, 2))
            if '%' in df_summary_view.columns:
                df_summary_view['%'] = df_summary_view['%'].apply(lambda x: fmt_pct(x, 2))
            master_models = list(dict.fromkeys(display_models))
            df_master = pd.DataFrame({'Mô hình': master_models})
            df_join = df_master.merge(df_summary_view, on='Mô hình', how='left')
            r_df_cur = pd.DataFrame()
            if os.path.exists(RMSE_FILE):
                try:
                    r_df_cur = pd.DataFrame(json.load(open(RMSE_FILE))['results'])
                except Exception:
                    r_df_cur = pd.DataFrame()
            if not r_df_cur.empty:
                df_join = df_join.merge(r_df_cur[['Mô hình','RMSE','MAE','R2']], on='Mô hình', how='left')
            cv_df = pd.DataFrame()
            if os.path.exists(BACKTEST_METRICS_FILE):
                try:
                    cv_df = pd.DataFrame(json.load(open(BACKTEST_METRICS_FILE))['results'])
                except Exception:
                    cv_df = pd.DataFrame()
            if not cv_df.empty and 'CV RMSE' in cv_df.columns:
                try:
                    df_join = df_join.merge(cv_df[['Mô hình','CV RMSE','Độ ổn định','CV STD']], on='Mô hình', how='left')
                    df_join['RMSE'] = df_join.apply(lambda r: r['CV RMSE'] if pd.notna(r.get('CV RMSE')) else r.get('RMSE'), axis=1)
                    if 'CV STD' in df_join.columns:
                        df_join['CV STD'] = df_join['CV STD'].apply(lambda x: fmt_num(x, 2) if pd.notna(x) else 'Chưa có số liệu')
                    if 'CV RMSE' in df_join.columns:
                        df_join = df_join.drop(columns=['CV RMSE'])
                except Exception:
                    pass
            if 'Độ tin cậy (Backtest)' in df_join.columns and 'RMSE' in df_join.columns:
                try:
                    df_join['Độ tin cậy (Backtest)'] = df_join.apply(lambda r: (f"RMSE: {float(r['RMSE']):.2f}" if pd.notna(r['RMSE']) else 'Chưa kiểm thử'), axis=1)
                except Exception:
                    pass
            fill_cols = ['Ngày chạy','Dự báo T1','Δ','%','Khuyến nghị','Độ tin cậy (Backtest)']
            for c in fill_cols:
                if c in df_join.columns:
                    df_join[c] = df_join[c].fillna('Chưa có số liệu')
            for c in ['RMSE','MAE','R2']:
                if c in df_join.columns:
                    df_join[c] = df_join[c].apply(lambda x: fmt_num(x, 4) if pd.notna(x) else 'Chưa có số liệu')
            df_join.index = np.arange(1, len(df_join) + 1)
            df_view = df_join.copy()
            if 'RMSE' in df_view.columns:
                try:
                    df_view = df_view.drop(columns=['RMSE'])
                except Exception:
                    pass
            st.dataframe(
                df_view,
                column_config={
                    'Ngày chạy': st.column_config.TextColumn('Ngày chạy'),
                    'Mô hình': st.column_config.TextColumn('Mô hình'),
                    'Dự báo T1': st.column_config.TextColumn('Dự báo T1'),
                    'Δ': st.column_config.TextColumn('Δ'),
                    '%': st.column_config.TextColumn('%'),
                    'Khuyến nghị': st.column_config.TextColumn('Khuyến nghị'),
                    'Độ tin cậy (Backtest)': st.column_config.TextColumn('Độ tin cậy (Backtest)'),
                    'MAE': st.column_config.TextColumn('MAE'),
                    'R2': st.column_config.TextColumn('R2'),
                    'Độ ổn định': st.column_config.TextColumn('Độ ổn định'),
                    'CV STD': st.column_config.TextColumn('Biến động RMSE (CV σ)')
                },
                width='stretch'
            )
            try:
                os.makedirs('history_logs', exist_ok=True)
                txt_lines = []
                txt_lines.append(f"Ngay chay: {today_str}")
                txt_lines.append("Mo hinh | RMSE | MAE | R2 | Do on dinh | CV sigma")
                for _, r in df_join.iterrows():
                    txt_lines.append(
                        f"{r.get('Mô hình','-')} | {r.get('RMSE','--')} | {r.get('MAE','--')} | {r.get('R2','--')} | {r.get('Độ ổn định','--')} | {r.get('CV STD','--')}"
                    )
                txt_content = "\n".join(txt_lines)
                run_day = datetime.today().strftime('%Y-%m-%d')
                with open(os.path.join('history_logs', f"models_metrics_{run_day}.txt"), 'w', encoding='utf-8') as f:
                    f.write(txt_content)
                with open('models_metrics.txt', 'w', encoding='utf-8') as f:
                    f.write(txt_content)
            except Exception:
                pass
            fig_t1 = px.bar(df_summary, x='Mô hình', y='%', text='%', title='Tỷ lệ % T1 theo mô hình<br><sup>Biến động dự kiến so với giá hiện tại</sup>', color_discrete_sequence=['#42a5f5'])
            fig_t1.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            ymax_t1 = float(df_summary['%'].max()) if not df_summary.empty else 0.0
            ymin_t1 = float(df_summary['%'].min()) if not df_summary.empty else 0.0
            fig_t1.update_traces(cliponaxis=False)
            fig_t1.update_layout(
                yaxis=dict(range=[min(0, ymin_t1*1.15), ymax_t1*1.15], automargin=True),
                margin=dict(t=60, b=40, l=60, r=20),
                xaxis_tickangle=-20,
                template='plotly_white',
                uniformtext_minsize=8,
                uniformtext_mode='hide',
                showlegend=False
            )
            st.plotly_chart(fig_t1, width='stretch')
            st.markdown("### Biểu đồ theo ngày T")
            selected_T = st.slider("Chọn ngày T (1-14)", 1, min(14, days), 1)
            close_last = float(df['Close'].iloc[-1])
            t_vals = []
            cols_ok = (not current_preds.empty) and ('model_name' in current_preds.columns)
            for m in display_models:
                r = current_preds[current_preds['model_name']==m] if cols_ok else pd.DataFrame()
                v = None
                if not r.empty:
                    v = r.iloc[0].get(f'pred_T{selected_T}')
                pct = None
                if v is not None and pd.notna(v):
                    try:
                        pct = (float(v) - close_last) / close_last * 100.0
                    except Exception:
                        pct = None
                t_vals.append({'Mô hình': m, 'Giá dự báo': v if v is not None and pd.notna(v) else np.nan, '%': pct if pct is not None else np.nan})
            df_t = pd.DataFrame(t_vals)
            base_dates = current_preds['predicted_for_date'] if 'predicted_for_date' in current_preds.columns else pd.Series([])
            base_dt = pd.to_datetime(base_dates.iloc[0], errors='coerce') if len(base_dates) > 0 else None
            t_dates = ([base_dt] + bdays_from(base_dt, days - 1)) if base_dt is not None else []
            lbl_date = (t_dates[selected_T-1].strftime('%d/%m/%Y') if selected_T-1 < len(t_dates) else f'T{selected_T}')
            fig_t_price = px.bar(df_t, x='Mô hình', y='Giá dự báo', text='Giá dự báo', title=f'Giá dự báo theo mô hình - {lbl_date}<br><sup>Giá dự báo tuyệt đối theo từng mô hình</sup>', color_discrete_sequence=['#42a5f5'])
            fig_t_price.update_traces(texttemplate='%{text:.2f}', textposition='outside', cliponaxis=False)
            fig_t_price.update_layout(margin=dict(t=60,b=40,l=60,r=20), xaxis_tickangle=-20, template='plotly_white', uniformtext_minsize=8, uniformtext_mode='hide', showlegend=False)
            
            st.plotly_chart(fig_t_price, width='stretch')
            fig_t_pct = px.bar(df_t, x='Mô hình', y='%', text='%', title=f'Tỷ lệ biến động (%) - {lbl_date}<br><sup>Biến động (%) dự kiến vào ngày đã chọn</sup>', color_discrete_sequence=['#42a5f5'])
            fig_t_pct.update_traces(texttemplate='%{text:.2f}%', textposition='outside', cliponaxis=False)
            ymax_t = float(df_t['%'].max()) if not df_t['%'].isna().all() else 0.0
            ymin_t = float(df_t['%'].min()) if not df_t['%'].isna().all() else 0.0
            fig_t_pct.update_layout(yaxis=dict(range=[min(0, ymin_t*1.15), ymax_t*1.15], automargin=True), margin=dict(t=60,b=40,l=60,r=20), xaxis_tickangle=-20, template='plotly_white', uniformtext_minsize=8, uniformtext_mode='hide', showlegend=False)
            fig_t_pct.add_annotation(x=0, y=1.05, xref='paper', yref='paper', text='📊 Biến động (%) theo mô hình vào ngày đã chọn', showarrow=False)
            st.plotly_chart(fig_t_pct, width='stretch')
            # Build charts: aggregate + per-model; and forecast table by date for report
            fig_line = None
            figs_by_model = {}
            df_forecast_report = pd.DataFrame()
            if not current_preds.empty:
                df_view = df.tail(300)
                best_one = []
                try:
                    if os.path.exists(RMSE_FILE):
                        r_df_best = pd.DataFrame(json.load(open(RMSE_FILE))['results'])
                        if not r_df_best.empty and 'RMSE' in r_df_best.columns and 'Mô hình' in r_df_best.columns:
                            best_one = r_df_best.sort_values('RMSE')['Mô hình'].head(1).tolist()
                except Exception:
                    best_one = []
                if not best_one:
                    hybrids = [m for m in display_models if "Hybrid" in m]
                    best_one = hybrids[:1] if hybrids else display_models[:1]
                fig_line = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
                fig_line.add_trace(go.Candlestick(x=df_view['Date'], open=df_view['Open'], high=df_view['High'], low=df_view['Low'], close=df_view['Close'], name='OHLC', showlegend=False), row=1, col=1)
                fig_line.add_trace(go.Bar(x=df_view['Date'], y=df_view['Volume'] if 'Volume' in df_view.columns else [np.nan]*len(df_view), name='Khối lượng', showlegend=False), row=2, col=1)
                fig_line.add_trace(go.Scatter(x=df_view['Date'], y=df_view['Close'], mode='lines', name='Giá thực tế', line=dict(color='#ffffff', width=3), opacity=0.99, visible=True), row=1, col=1)
                if show_model_prices:
                    for m in display_models:
                        r = current_preds[current_preds['model_name']==m]
                        if not r.empty:
                            c = model_color(m)
                            y_vals = []
                            for k in range(1, days + 1):
                                val = r.iloc[0].get(f'pred_T{k}')
                                if pd.notna(val): y_vals.append(val)
                            if y_vals:
                                base_dt_m = pd.to_datetime(r.iloc[0].get('predicted_for_date'), errors='coerce') if 'predicted_for_date' in r.columns else None
                                x_vals = ([df['Date'].iloc[-1]] + bdays_from(base_dt_m, len(y_vals))) if base_dt_m is not None else []
                                vis = True if m in best_one else 'legendonly'
                                if len(x_vals) > 0 and len(y_vals) > 0:
                                    fig_line.add_trace(go.Scatter(x=x_vals, y=[df['Close'].iloc[-1]] + y_vals, mode='lines', name=m, line=dict(color=c, width=2), opacity=0.85 if m in best_one else 0.6, visible=vis), row=1, col=1)
                                fm = go.Figure()
                                fm.add_trace(go.Scatter(x=df_view['Date'], y=df_view['Close'], name='Giá thực tế', mode='lines', line=dict(color='#9aa4b2', width=3)))
                                if len(x_vals) > 0 and len(y_vals) > 0:
                                    fm.add_trace(go.Scatter(x=x_vals, y=[df['Close'].iloc[-1]] + y_vals, name='Dự báo', mode='lines+markers', line=dict(color=c, width=3)))
                                fm.update_layout(title=f'Dự báo {m}', template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e6e8eb', margin=dict(t=40,b=40,l=40,r=20), height=500, xaxis=dict(gridcolor='#2a2f3a', tickfont=dict(color='#e6e8eb')), yaxis=dict(gridcolor='#2a2f3a', tickfont=dict(color='#e6e8eb')))
                                fm.update_xaxes(tickformat='%d/%m')
                                figs_by_model[m] = fm
                fig_line.update_layout(template='plotly_dark', hovermode='x unified', legend=dict(orientation='h', yanchor='top', y=-0.15), height=700, margin=dict(t=40,b=40,l=20,r=20))
                fig_line.update_xaxes(rangeslider_visible=False)
                st.plotly_chart(fig_line, width='stretch')
                try:
                    start_dt = df['Date'].iloc[-1] - pd.Timedelta(days=90)
                    df_vol = df[df['Date'] >= start_dt].copy()
                    df_vol = df_vol[pd.to_datetime(df_vol['Date']).dt.weekday < 5]
                    dates_v = pd.to_datetime(df_vol['Date'], errors='coerce')
                    vol_v = pd.to_numeric(df_vol['Volume'], errors='coerce').fillna(0).values
                    fig_vol = go.Figure()
                    fig_vol.add_trace(go.Bar(x=dates_v, y=vol_v, name='Khối lượng', marker=dict(color='#42a5f5'), opacity=0.35))
                    tmp_v = pd.DataFrame({'Date': dates_v, 'Volume': vol_v})
                    tmp_v['Week'] = tmp_v['Date'].dt.to_period('W')
                    wk_idx = tmp_v.groupby('Week')['Volume'].idxmax().dropna().astype(int).tolist()
                    if wk_idx:
                        peaks_x = dates_v.iloc[wk_idx]
                        peaks_y = vol_v[wk_idx]
                        fig_vol.add_trace(go.Scatter(x=peaks_x, y=peaks_y, mode='lines+markers', name='Đỉnh tuần', line=dict(color='#58a6ff', width=2), marker=dict(size=6, color='#58a6ff')))
                        top_idx = np.argsort(peaks_y)[-3:] if len(peaks_y) >= 3 else range(len(peaks_y))
                        for ti in top_idx:
                            fig_vol.add_annotation(x=peaks_x.iloc[int(ti)], y=peaks_y[int(ti)], text='🔺 Đỉnh tuần', showarrow=True, arrowhead=2, font=dict(color='#e6e8eb', size=12))
                    fig_vol.update_yaxes(tickformat='~s', gridcolor='#2a2f3a')
                    fig_vol.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e6e8eb', height=320, margin=dict(t=40,b=30,l=40,r=20))
                    st.plotly_chart(fig_vol, width='stretch')
                except Exception:
                    pass
                # Build forecast table labeled by dates
                rows_by_date = []
                for _, r in current_preds.iterrows():
                    base = r.get('predicted_for_date')
                    item = {
                        'Ngày chạy': fmt_dt_str(r.get('date_predicted')),
                        'Ngày dự báo': fmt_date_str(base),
                        'Mô hình': str(r.get('model_name') or '-')
                    }
                    td = ([base] + bdays_from(base, 13)) if base is not None else []
                    for i in range(1, 15):
                        col_lbl = (td[i-1].strftime('%d/%m/%Y') if i-1 < len(td) else f'T{i}')
                        val = r.get(f'pred_T{i}')
                        item[col_lbl] = fmt_num(val, 2)
                    rows_by_date.append(item)
                df_forecast_report = pd.DataFrame(rows_by_date)
            # Load metrics for report
            r_df_report = pd.DataFrame()
            fig_rmse_report = None
            if os.path.exists(RMSE_FILE):
                try:
                    r_df_report = pd.DataFrame(json.load(open(RMSE_FILE))['results'])
                    if not r_df_report.empty:
                        fig_rmse_report = px.bar(r_df_report, x='Mô hình', y='RMSE', text='RMSE', title='RMSE theo mô hình<br><sup>Sai số gốc trong báo cáo, càng thấp càng tốt</sup>', color_discrete_sequence=['#42a5f5'])
                        fig_rmse_report.update_traces(texttemplate='%{text:.2f}', textposition='outside', cliponaxis=False)
                        rmse_max = float(r_df_report['RMSE'].max())
                        fig_rmse_report.update_layout(margin=dict(t=60,b=40,l=60,r=20), xaxis_tickangle=-20, template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e6e8eb', xaxis=dict(gridcolor='#2a2f3a', tickfont=dict(color='#e6e8eb')), yaxis=dict(range=[0, rmse_max*1.15], gridcolor='#2a2f3a', tickfont=dict(color='#e6e8eb')), showlegend=False)
                except Exception:
                    r_df_report = pd.DataFrame()
            try:
                fname, html = generate_report(df_join, r_df_report, fig_rmse_report, fig_line, df_forecast_report, models_list=master_models, figs_by_model=figs_by_model)
            except Exception:
                pass
            # FIX: Khu vực xuất dữ liệu cuối trang
            try:
                with st.sidebar.expander("📥 Xuất dữ liệu", expanded=False):
                    excel_bytes = export_comprehensive_excel(df_join, df_forecast_report, r_df_report, hist, df)
                    xlsx_name = f"Du_lieu_VNINDEX_{datetime.today().strftime('%d%m%Y')}.xlsx"
                    st.download_button("Tải Excel tổng hợp", data=excel_bytes, file_name=xlsx_name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="download_excel_all")
                    try:
                        imgs = []
                        # Thêm trang thông số cấu hình và môi trường
                        try:
                            params_items = [
                                ('Thời điểm xuất', datetime.today().strftime('%d/%m/%Y %H:%M')),
                                ('Số ngày dự báo', str(days)),
                                ('Train/Test', f"{int((1-TEST_SPLIT_PCT)*100)}%/{int(TEST_SPLIT_PCT*100)}%"),
                                ('Scaler (DL)', 'MinMaxScaler (fit trên Train)'),
                                ('Scaler (ML)', 'StandardScaler (fit trên Train)'),
                                ('LookBack (DL)', str(LOOK_BACK_DAYS)),
                                ('DL Epochs', str(DL_EPOCHS)),
                                ('DL Batch Size', str(DL_BATCH_SIZE)),
                                ('Số dòng dữ liệu', str(len(df))),
                                ('Khoảng ngày', f"{fmt_date_str(df['Date'].iloc[0])} → {fmt_date_str(df['Date'].iloc[-1])}"),
                                ('Giá Close gần nhất', fmt_num(df['Close'].iloc[-1], 2)),
                                ('Mô hình chọn', ', '.join(display_models)),
                                ('Hybrid chọn', ', '.join(hybrid_labels) if hybrid_labels else 'Không')
                            ]
                            params_df = pd.DataFrame(params_items, columns=['Tham số', 'Giá trị'])
                            f_params = table_to_fig(params_df, 'Thông số cấu hình & môi trường')
                            imgs.append(pio.to_image(f_params, format='png', engine='kaleido', width=900, height=520))
                        except Exception:
                            pass
                        # Thêm bảng độ đo dự báo (RMSE/MAE/R2) nếu có
                        try:
                            if r_df_report is not None and not r_df_report.empty:
                                cols = []
                                for c in ['Mô hình','RMSE','MAE','R2','CV RMSE','CV R2']:
                                    if c in r_df_report.columns: cols.append(c)
                                metrics_tbl = r_df_report[cols].copy()
                                f_metrics = table_to_fig(metrics_tbl, 'Độ đo dự báo theo mô hình')
                                imgs.append(pio.to_image(f_metrics, format='png', engine='kaleido', width=900, height=520))
                        except Exception:
                            pass
                        if fig_rmse_report is not None:
                            imgs.append(pio.to_image(fig_rmse_report, format='png', engine='kaleido', width=900, height=520))
                        if fig_line is not None:
                            imgs.append(pio.to_image(fig_line, format='png', engine='kaleido', width=900, height=520))
                        eval_rows = []
                        if current_preds is not None and not current_preds.empty and df is not None:
                            price_map = df.set_index('Date')['Close'].to_dict()
                            date_list = df['Date'].tolist()
                            date_to_idx = {d: i for i, d in enumerate(date_list)}
                            for _, r in current_preds.iterrows():
                                base = r.get('predicted_for_date')
                                pred = r.get('pred_T1')
                                actual = None
                                try:
                                    if base in date_to_idx:
                                        idx = date_to_idx[base]
                                        actual = df.iloc[idx]['Close']
                                except Exception:
                                    actual = None
                                eval_rows.append({
                                    'Ngày dự báo': fmt_date_str(base),
                                    'Mô hình': str(r.get('model_name') or '-'),
                                    'Giá dự báo': fmt_num(pred, 2),
                                    'Giá thực tế': fmt_num(actual, 2)
                                })
                        if eval_rows:
                            df_eval_pdf = pd.DataFrame(eval_rows)
                            chunks = [df_eval_pdf.iloc[i:i+30].reset_index(drop=True) for i in range(0, len(df_eval_pdf), 30)]
                            for idx, ch in enumerate(chunks):
                                f = table_to_fig(ch, f"Đánh giá ngày dự báo ({idx+1}/{len(chunks)})")
                                imgs.append(pio.to_image(f, format='png', engine='kaleido', width=900, height=520))
                        if show_model_prices and figs_by_model:
                            for m in list(figs_by_model.keys()):
                                fig_m = figs_by_model.get(m)
                                if fig_m is not None:
                                    imgs.append(pio.to_image(fig_m, format='png', engine='kaleido', width=900, height=520))
                        pil_imgs = []
                        for b in imgs:
                            try:
                                pil_imgs.append(Image.open(io.BytesIO(b)).convert("RGB"))
                            except Exception:
                                pass
                        if pil_imgs:
                            buf_pdf = io.BytesIO()
                            pil_imgs[0].save(buf_pdf, format='PDF', save_all=True, append_images=pil_imgs[1:])
                            pdf_bytes = buf_pdf.getvalue()
                            pdf_name = f"Bao_cao_VNINDEX_{datetime.today().strftime('%d%m%Y')}.pdf"
                            st.download_button("Tải báo cáo PDF đầy đủ", data=pdf_bytes, file_name=pdf_name, mime="application/pdf", key="download_pdf_full")
                    except Exception:
                        st.info("Không thể xuất PDF. Vui lòng cài 'kaleido' (pip install -U kaleido).")
            except Exception:
                st.warning("Không thể tạo mục tải dữ liệu trong Sidebar.")
        else:
            st.info("Vui lòng bấm nút Chạy Dự Báo hoặc Backtest bên Sidebar để bắt đầu")
            with st.expander(f"Lộ trình giá {days} ngày tới", expanded=False):
                st.markdown("""
                <style>
                div[data-testid="stDataFrame"] table { font-size: 12px; }
                div[data-testid="stDataFrame"] table th, 
                div[data-testid="stDataFrame"] table td { padding: 6px 8px; white-space: nowrap; }
                </style>
                """, unsafe_allow_html=True)
                forecast_data = []
                for _, row in current_preds.iterrows():
                    item = {'Mô hình': row['model_name']}
                    base = pd.to_datetime(row.get('predicted_for_date'), errors='coerce')
                    td = bdays_from(base, days) if pd.notna(base) else []
                    for i in range(days): 
                        col_name = (td[i].strftime('%d/%m') if i < len(td) else f'T{i+1}')
                        val = row.get(f'pred_T{i+1}')
                        item[col_name] = f"{val:,.2f}" if pd.notna(val) else "-"
                    forecast_data.append(item)
                df_forecast = pd.DataFrame(forecast_data)
                df_forecast.index = np.arange(1, len(df_forecast) + 1)
                st.dataframe(df_forecast, width='stretch', height=420)

            with st.expander("Biểu đồ Xu hướng & Khối lượng", expanded=True):
                range_opt = st.selectbox("Phạm vi hiển thị", ["90 ngày", "180 ngày", "365 ngày", "Tất cả"], index=0)
                if range_opt == "Tất cả":
                    df_view = df.copy()
                else:
                    try:
                        days_win = int(range_opt.split()[0])
                    except Exception:
                        days_win = 90
                    start_dt = df['Date'].iloc[-1] - pd.Timedelta(days=days_win)
                    df_view = df[df['Date'] >= start_dt]
                try:
                    df_view = df_view[pd.to_datetime(df_view['Date']).dt.weekday < 5]
                except Exception:
                    pass
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3], subplot_titles=("Biến động Giá", "Khối lượng"))
                fig.add_trace(go.Candlestick(x=df_view['Date'], open=df_view['Open'], high=df_view['High'], low=df_view['Low'], close=df_view['Close'], name='Giá'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_view['Date'], y=df_view['Close'], mode='lines', name='Giá thực tế', line=dict(color='#9aa4b2', width=3)), row=1, col=1)
                cols_ok = (not current_preds.empty) and ('model_name' in current_preds.columns)
                best_one = []
                try:
                    if os.path.exists(RMSE_FILE):
                        r_df_best = pd.DataFrame(json.load(open(RMSE_FILE))['results'])
                        if not r_df_best.empty and 'RMSE' in r_df_best.columns and 'Mô hình' in r_df_best.columns:
                            best_one = r_df_best.sort_values('RMSE')['Mô hình'].head(1).tolist()
                except Exception:
                    best_one = []
                if not best_one:
                    hybrids = [m for m in display_models if "Hybrid" in m]
                    best_one = hybrids[:1] if hybrids else display_models[:1]
                if show_model_prices:
                    for m in display_models:
                        r = current_preds[current_preds['model_name']==m] if cols_ok else pd.DataFrame()
                        if not r.empty:
                            c = model_color(m)
                            y_vals = []
                            for k in range(1, days + 1):
                                val = r.iloc[0].get(f'pred_T{k}')
                                if pd.notna(val): y_vals.append(val)
                            if y_vals:
                                base_dt = pd.to_datetime(r.iloc[0].get('predicted_for_date'), errors='coerce') if 'predicted_for_date' in r.columns else None
                                x_vals = ([df['Date'].iloc[-1]] + bdays_from(base_dt, len(y_vals))) if base_dt is not None else []
                                vis = True if m in best_one else 'legendonly'
                                fig.add_trace(
                                    go.Scatter(
                                        x=x_vals,
                                        y=[df['Close'].iloc[-1]] + y_vals,
                                        mode='lines+markers',
                                        name=m,
                                        line=dict(color=c, width=3),
                                        marker=dict(size=5),
                                        hovertemplate='%{x|%d/%m/%Y}: %{y:.2f}<extra></extra>',
                                        visible=vis,
                                        opacity=0.9 if m in best_one else 0.6
                                    ),
                                    row=1,
                                    col=1
                                )
                                if m in best_one:
                                    try:
                                        fig.add_annotation(x=x_vals[-1], y=y_vals[-1], xref='x', yref='y', text=m, showarrow=False, yshift=-10, font=dict(color='#e6e8eb', size=10))
                                    except Exception:
                                        pass
                try:
                    dates = pd.to_datetime(df_view['Date'], errors='coerce')
                    vol = pd.to_numeric(df_view['Volume'], errors='coerce').fillna(0).values
                    fig.add_trace(
                        go.Bar(
                            x=dates,
                            y=vol,
                            name='Khối lượng',
                            marker=dict(color='#42a5f5'),
                            opacity=0.35
                        ),
                        row=2, col=1
                    )
                    peaks_idx = []
                    for i in range(1, len(vol)-1):
                        if vol[i] >= vol[i-1] and vol[i] >= vol[i+1]:
                            peaks_idx.append(i)
                    if peaks_idx:
                        peaks_x = dates.iloc[peaks_idx]
                        peaks_y = vol[peaks_idx]
                        fig.add_trace(
                            go.Scatter(
                                x=peaks_x,
                                y=peaks_y,
                                mode='lines+markers',
                                name='Đường nối đỉnh',
                                line=dict(color='#58a6ff', width=2),
                                marker=dict(size=5, color='#58a6ff'),
                                hovertemplate='Đỉnh: %{x|%d/%m/%Y}<br>Volume: %{y:.0f}<extra></extra>'
                            ),
                            row=2, col=1
                        )
                        top_idx = np.argsort(peaks_y)[-3:] if len(peaks_y) >= 3 else range(len(peaks_y))
                        for ti in top_idx:
                            fig.add_annotation(x=peaks_x.iloc[int(ti)], y=peaks_y[int(ti)], xref='x', yref='y2', text='🔺 Đỉnh', showarrow=True, arrowhead=2, font=dict(color='#e6e8eb', size=12))
                except Exception:
                    fig.add_trace(go.Bar(x=df_view['Date'], y=pd.to_numeric(df_view['Volume'], errors='coerce'), name='Khối lượng', marker_color='#42a5f5', opacity=0.35), row=2, col=1)
                fig.update_yaxes(tickformat='~s', row=2, col=1, gridcolor='#2a2f3a')
                fig.update_yaxes(tickformat=',.2f', row=1, col=1, gridcolor='#2a2f3a')
                fig.update_layout(legend_title_text="Mô hình", legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1), font_color='#e6e8eb')
                fig.add_annotation(x=0, y=1.06, xref='paper', yref='paper', text='📈 Giá & dự báo (tất cả mô hình)', showarrow=False, font=dict(color='#e6e8eb', size=12))
                fig.add_annotation(x=0, y=0.38, xref='paper', yref='paper', text='🔺 Khối lượng: cột + nối các đỉnh liên tiếp', showarrow=False, font=dict(color='#e6e8eb', size=12))
                fig.update_layout(xaxis_rangeslider_visible=False, template="plotly_dark", height=700, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, width='stretch')

        with st.expander("Biểu đồ trực quan hóa mô hình", expanded=False):
            heat_rows = []
            if not current_preds.empty:
                for _, row in current_preds.iterrows():
                    vals = []
                    for k in range(1, days + 1):
                        v = row.get(f'pred_T{k}')
                        vals.append(v if pd.notna(v) else np.nan)
                    heat_rows.append(vals)
            if heat_rows and ('model_name' in current_preds.columns):
                heat_arr = np.array(heat_rows)
                base0 = None
                try:
                    base0 = pd.to_datetime(current_preds.iloc[0].get('predicted_for_date'), errors='coerce') if not current_preds.empty else None
                except Exception:
                    base0 = None
                tds = bdays_from(base0, days) if base0 is not None else next_dates[:days]
                day_labels = [d.strftime('%d/%m') for d in tds]
                fig_hm = px.imshow(heat_arr, labels=dict(x='Ngày', y='Mô hình', color='Giá dự báo'), x=day_labels, y=current_preds['model_name'].tolist(), aspect='auto', color_continuous_scale='Viridis')
                fig_hm.update_traces(hovertemplate='Ngày: %{x}<br>Mô hình: %{y}<br>Giá dự báo: %{z:.2f}<extra></extra>')
                fig_hm.add_annotation(x=0, y=1.05, xref='paper', yref='paper', text='🔥 Heatmap dự báo theo ngày & mô hình', showarrow=False)
                st.plotly_chart(fig_hm, width='stretch')
            else:
                st.info("Chưa có dữ liệu dự báo để trực quan hóa.")

with tab2:
        st.subheader("1. Bảng Xếp Hạng (In-Sample)")
        if os.path.exists(RMSE_FILE):
            r_df = pd.DataFrame(json.load(open(RMSE_FILE))['results'])
            if not r_df.empty:
                if 'CV RMSE' in r_df.columns and 'RMSE' not in r_df.columns: r_df['RMSE'] = r_df['CV RMSE']
                if 'CV R2' in r_df.columns and 'R2' not in r_df.columns: r_df['R2'] = r_df['CV R2']
                for c in ['Mô hình','RMSE','MAE','R2']:
                    if c not in r_df.columns: r_df[c] = np.nan
            # Hợp nhất để hiển thị đủ 15 mô hình (kể cả mô hình chưa backtest)
            display_models2 = list(models)
            if 'LSTM_CatBoost' in hybrid_types: display_models2.append("Hybrid (LSTM + CatBoost)")
            if 'ARIMA_LSTM' in hybrid_types: display_models2.append("Hybrid (ARIMA + LSTM)")
            if 'Voting' in hybrid_types: display_models2.append("Hybrid (Voting ML)")
            if 'Stacking' in hybrid_types: display_models2.append("Hybrid (Stacking ML)")
            full_df = pd.DataFrame({'Mô hình': display_models2})
            r_sel = r_df[['Mô hình','RMSE','MAE','R2']].copy() if not r_df.empty else pd.DataFrame(columns=['Mô hình','RMSE','MAE','R2'])
            full_df = full_df.merge(r_sel, on='Mô hình', how='left')
            full_df['RMSE'] = full_df['RMSE'].apply(lambda x: fmt_num(x, 2))
            full_df['MAE'] = full_df['MAE'].apply(lambda x: fmt_num(x, 2))
            full_df['R2'] = full_df['R2'].apply(lambda x: fmt_num(x, 4))
            full_df.index = np.arange(1, len(full_df) + 1)
            st.dataframe(
                full_df,
                column_config={
                    "Mô hình": st.column_config.TextColumn("Mô hình"),
                    "RMSE": st.column_config.TextColumn("RMSE"),
                    "MAE": st.column_config.TextColumn("MAE"),
                    "R2": st.column_config.TextColumn("R2")
                },
                width='stretch'
            )
            # Biểu đồ chỉ thể hiện các mô hình có số đo
            if not r_df.empty:
                fig_rmse = px.bar(r_df, x='Mô hình', y='RMSE', text='RMSE', title='RMSE theo mô hình', color_discrete_sequence=['#42a5f5'])
                fig_rmse.update_traces(texttemplate='%{text:.2f}', textposition='outside', cliponaxis=False)
                rmse_max = float(r_df['RMSE'].max()) if not r_df.empty else 0.0
                mae_max = float(r_df['MAE'].max()) if not r_df.empty else 0.0
                fig_rmse.update_layout(yaxis=dict(range=[0, rmse_max*1.15], automargin=True), margin=dict(t=60,b=40,l=60,r=20), xaxis_tickangle=-20, template='plotly_white', uniformtext_minsize=8, uniformtext_mode='hide', showlegend=False)
                #fig_rmse.add_annotation(x=0, y=1.05, xref='paper', yref='paper', text='📊 RMSE theo mô hình', showarrow=False)
                st.plotly_chart(fig_rmse, width='stretch')
                if 'R2' in r_df.columns and not r_df['R2'].isna().all():
                    fig_r2 = px.bar(r_df, x='Mô hình', y='R2', text='R2', title='R2 theo mô hình', color_discrete_sequence=['#42a5f5'])
                    fig_r2.update_traces(texttemplate='%{text:.4f}', textposition='outside', cliponaxis=False)
                    r2_max = float(r_df['R2'].max()) if not r_df.empty else 0.0
                    r2_min = float(r_df['R2'].min()) if not r_df.empty else 0.0
                    fig_r2.update_layout(yaxis=dict(range=[min(0, r2_min*1.15), r2_max*1.15], automargin=True), margin=dict(t=60,b=40,l=60,r=20), xaxis_tickangle=-20, template='plotly_white', uniformtext_minsize=8, uniformtext_mode='hide', showlegend=False)
                    st.plotly_chart(fig_r2, width='stretch')
                if 'MAE' in r_df.columns and not r_df['MAE'].isna().all():
                    fig_mae = px.bar(r_df, x='Mô hình', y='MAE', text='MAE', title='MAE theo mô hình', color_discrete_sequence=['#42a5f5'])
                    fig_mae.update_traces(texttemplate='%{text:.2f}', textposition='outside', cliponaxis=False)
                    mae_max = float(r_df['MAE'].max()) if not r_df.empty else 0.0
                    fig_mae.update_layout(yaxis=dict(range=[0, mae_max*1.15], automargin=True), margin=dict(t=60,b=40,l=60,r=20), xaxis_tickangle=-20, template='plotly_white', uniformtext_minsize=8, uniformtext_mode='hide', showlegend=False)
                    st.plotly_chart(fig_mae, width='stretch')
        else: st.warning("Chưa có dữ liệu.")
        
        st.subheader("2. Chẩn đoán Mô hình (Model Diagnostics)")
        diag_rows = st.session_state.get('model_diag', [])
        if isinstance(diag_rows, list) and len(diag_rows) > 0:
            fig_diag = build_summary_figure(diag_rows)
            st.plotly_chart(fig_diag, width='stretch')
        else:
            st.info("Chưa có thông số mô hình. Hãy chạy dự báo để xem chẩn đoán.")
        
        st.subheader("3. Diễn biến Sai số (Error Trend)")
        hist_t = st.slider("Chọn ngày T (1-14)", 1, 14, 1, key="err_hist_T")
        if not hist.empty:
            err_col = f'error_T{hist_t}'
            pred_col = f'pred_T{hist_t}'
            act_col = f'actual_T{hist_t}'
            if act_col not in hist.columns and 'predicted_for_date' in hist.columns:
                df_norm = df.copy()
                df_norm['Date'] = pd.to_datetime(df_norm['Date'], errors='coerce').dt.normalize()
                lookup = df_norm.set_index('Date')['Close'].to_dict()
                def add_bd(d, k):
                    try:
                        cur = pd.to_datetime(d, errors='coerce')
                        if pd.isna(cur): return None
                        cur = cur.normalize()
                        steps = int(k)
                        while steps > 0:
                            cur = cur + pd.Timedelta(days=1)
                            if cur.weekday() < 5:
                                steps -= 1
                        return cur
                    except:
                        return None
                def get_future_actual(row):
                    tgt = add_bd(row['predicted_for_date'], hist_t - 1)
                    if tgt is None: return np.nan
                    return float(lookup.get(tgt, np.nan))
                hist[act_col] = hist.apply(lambda r: get_future_actual(r), axis=1)
            hist[err_col] = hist[act_col] - hist[pred_col]
            err_data = hist.dropna(subset=[err_col])
            if not err_data.empty:
                try:
                    err_data = err_data.copy()
                    err_data['x_label'] = err_data['model_name'].astype(str)
                except Exception:
                    err_data['x_label'] = err_data['model_name'].astype(str)
                fig_err = px.bar(err_data, x='x_label', y=err_col, title=f"Biến động sai số T{hist_t}<br><sup>Chênh lệch Thực tế - Dự báo</sup>", color_discrete_sequence=['#42a5f5'])
                fig_err.update_traces(texttemplate='%{y:.2f}', textposition='outside', cliponaxis=False)
                ymax_e = float(err_data[err_col].max()); ymin_e = float(err_data[err_col].min())
                fig_err.update_layout(showlegend=False, yaxis=dict(range=[ymin_e*1.15, ymax_e*1.15]), xaxis_tickangle=-30, template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e6e8eb')
                fig_err.add_hline(y=0, line_dash="dash", line_color="#999")
                st.plotly_chart(fig_err, width='stretch')

with tab3:
        st.subheader("Lịch sử chi tiết")
        if not hist.empty:
            h_show = hist.copy()
            df_norm = df.copy()
            df_norm['Date'] = pd.to_datetime(df_norm['Date'], errors='coerce').dt.normalize()
            lookup = df_norm.set_index('Date')['Close'].to_dict()
            def get_future_actual(row, steps):
                try:
                    cur = pd.to_datetime(row['predicted_for_date'], errors='coerce')
                    if pd.isna(cur): return np.nan
                    cur = cur.normalize()
                    k = int(steps) - 1
                    while k > 0:
                        cur = cur + pd.Timedelta(days=1)
                        if cur.weekday() < 5:
                            k -= 1
                    return float(lookup.get(cur, np.nan))
                except: pass
                return np.nan
            for i in range(1, 15): h_show[f'actual_T{i}'] = h_show.apply(lambda r: get_future_actual(r, i), axis=1)
            
            # Recalculate errors for display
            for i in range(1, 15):
                if f'pred_T{i}' in h_show.columns:
                    h_show[f'error_T{i}'] = h_show[f'actual_T{i}'] - h_show[f'pred_T{i}']
                    h_show[f'error_pct_T{i}'] = (h_show[f'error_T{i}'] / h_show[f'pred_T{i}']) * 100
            
            h_show['date_predicted_str'] = h_show['date_predicted'].apply(fmt_dt_str)
            if 'predicted_for_date' in h_show.columns:
                h_show['predicted_for_date_str'] = h_show['predicted_for_date'].apply(fmt_date_str)
            else:
                h_show['predicted_for_date_str'] = '-'
            h_show['model_name'] = h_show['model_name'].astype(str).fillna('-')

            cols_to_show = ['date_predicted_str', 'predicted_for_date_str', 'model_name'] + [f'pred_T{i}' for i in range(1,15)] + [f'error_T{i}' for i in range(1,15)] + [f'error_pct_T{i}' for i in range(1,15)]
            cols_exist = [c for c in cols_to_show if c in h_show.columns]
            
            for c in cols_exist:
                if c.startswith('pred_T') or c.startswith('actual_T') or c.startswith('error_T') or c.startswith('error_pct_T'):
                    h_show[c] = pd.to_numeric(h_show[c], errors='coerce')
            
            col_cfg = {
                "date_predicted_str": st.column_config.TextColumn("Ngày chạy", help="Ngày bạn chạy dự báo"),
                "predicted_for_date_str": st.column_config.TextColumn("Ngày dự báo", help="Ngày áp dụng T1"),
                "model_name": st.column_config.TextColumn("Mô hình", help="Tên mô hình")
            }
            for i in range(1, 15):
                col_cfg[f'pred_T{i}'] = st.column_config.NumberColumn(f"Dự báo T{i}", format="%.2f", help=f"Giá dự báo ngày T{i}")
                if f'error_T{i}' in cols_exist:
                    col_cfg[f'error_T{i}'] = st.column_config.NumberColumn(f"Sai số T{i}", format="%.2f", help=f"Sai số giá T{i}")
                if f'error_pct_T{i}' in cols_exist:
                    col_cfg[f'error_pct_T{i}'] = st.column_config.TextColumn(f"Sai số % T{i}", help=f"Sai số % T{i}")

            df_hist_view = h_show[cols_exist].copy()
            for i in range(1, 15):
                c = f'error_pct_T{i}'
                if c in df_hist_view.columns:
                    df_hist_view[c] = df_hist_view[c].apply(lambda x: fmt_pct(x, 2))
            df_hist_view.index = np.arange(1, len(df_hist_view) + 1)
            st.dataframe(df_hist_view.tail(50), column_config=col_cfg, width='stretch')

            with st.expander("Bảng dự báo theo ngày (T1..T14 gắn nhãn ngày)", expanded=False):
                def add_business_days(d, k):
                    try:
                        cur = pd.to_datetime(d, errors='coerce')
                        if pd.isna(cur):
                            return None
                        steps = int(k)
                        while steps > 0:
                            cur = cur + pd.Timedelta(days=1)
                            if cur.weekday() < 5:
                                steps -= 1
                        return cur
                    except:
                        return None
                rows_by_date = []
                subset = h_show.tail(50).copy()
                for _, r in subset.iterrows():
                    base = r.get('predicted_for_date')
                    item = {
                        'Ngày chạy': fmt_dt_str(r.get('date_predicted')),
                        'Ngày dự báo': fmt_date_str(base),
                        'Mô hình': str(r.get('model_name') or '-')
                    }
                    for i in range(1, 15):
                        lbl_dt = add_business_days(base, i-1)
                        col_lbl = fmt_date_str(lbl_dt) if lbl_dt is not None else f'T{i}'
                        val = r.get(f'pred_T{i}')
                        item[col_lbl] = fmt_num(val, 2)
                    rows_by_date.append(item)
                df_hist_by_date = pd.DataFrame(rows_by_date)
            
            with st.expander("Bảng dự báo theo ngày (file mới hôm nay)", expanded=False):
                try:
                    fpath = os.path.join('history_logs', f"forecast_by_date_{datetime.today().strftime('%Y-%m-%d')}.csv")
                    if os.path.exists(fpath):
                        df_fd = pd.read_csv(fpath)
                        df_fd['date'] = pd.to_datetime(df_fd['date'], errors='coerce')
                        look = df.copy()
                        look['Date'] = pd.to_datetime(look['Date'], errors='coerce').dt.normalize()
                        mp = look.set_index('Date')['Close'].to_dict()
                        def _act(d):
                            try:
                                if pd.isna(d): return np.nan
                                dn = d.normalize()
                                v = mp.get(dn)
                                return float(v) if v is not None else np.nan
                            except:
                                return np.nan
                        df_fd['actual_rt'] = df_fd['date'].apply(lambda x: _act(x))
                        st.dataframe(df_fd, width='stretch')
                    else:
                        st.info("Chưa có file hôm nay.")
                except Exception:
                    st.info("Không thể hiển thị file hôm nay.")
                df_hist_by_date.index = np.arange(1, len(df_hist_by_date) + 1)
                st.dataframe(df_hist_by_date, width='stretch')

            st.subheader("Biểu đồ sai lệch theo từng mô hình")
            err_t = st.slider("Chọn ngày T (1-14)", 1, 14, 1, key="hist_err_T")
            err_df = h_show.copy()
            col_pred = f'pred_T{err_t}'
            col_actual = f'actual_T{err_t}'
            col_error = f'error_T{err_t}'
            if {'model_name','date_predicted', col_pred}.issubset(err_df.columns):
                err_df[col_error] = pd.to_numeric(err_df.get(col_error, np.nan), errors='coerce')
                err_df[col_pred] = pd.to_numeric(err_df[col_pred], errors='coerce')
                err_df[col_actual] = pd.to_numeric(err_df.get(col_actual, np.nan), errors='coerce')
                err_df = err_df.dropna(subset=[col_pred, col_actual])
                err_df[col_error] = err_df[col_actual] - err_df[col_pred]
                try:
                    err_df['date_predicted'] = pd.to_datetime(err_df['date_predicted'], errors='coerce')
                    last_dt = err_df['date_predicted'].max()
                    df_day = err_df[err_df['date_predicted'] == last_dt] if pd.notna(last_dt) else err_df
                except Exception:
                    df_day = err_df
                fig_err = px.bar(df_day, x='model_name', y=col_error, title=f'Sai lệch T{err_t} theo mô hình<br><sup>Chênh lệch Thực tế - Dự báo (ngày chạy gần nhất)</sup>', labels={col_error:'Sai lệch (điểm)'}, color_discrete_sequence=['#42a5f5'])
                fig_err.update_traces(cliponaxis=False, texttemplate='%{y:.2f}', textposition='outside')
                try:
                    ht = "Mô hình: %{x}<br>Sai lệch: %{y:.2f}<extra></extra>"
                    for tr in fig_err.data:
                        tr.update(hovertemplate=ht)
                except Exception:
                    pass
                ymax_e = float(df_day[col_error].max()) if not df_day.empty else 0.0
                ymin_e = float(df_day[col_error].min()) if not df_day.empty else 0.0
                fig_err.update_layout(yaxis=dict(range=[ymin_e*1.15, ymax_e*1.15], automargin=True), margin=dict(t=40,b=40,l=60,r=20), xaxis_tickangle=-20, template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e6e8eb')
                fig_err.add_hline(y=0, line_dash='dash', line_color='#999')
                st.plotly_chart(fig_err, width='stretch')

            st.download_button("📥 Tải Excel", to_excel(h_show), "history.xlsx", key="download_history_excel", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
