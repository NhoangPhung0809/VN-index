import pandas as pd
import plotly.graph_objects as go
import numpy as np

def summarize_arima(model, name='ARIMA+GARCH'):
    try:
        p, d, q = model.order
    except Exception:
        p, d, q = (np.nan, np.nan, np.nan)
    def _metric(m, attr):
        try:
            v = getattr(m, attr, np.nan)
            if callable(v):
                v = v()
            try:
                return float(v)
            except Exception:
                return np.nan
        except Exception:
            return np.nan
    aic = _metric(model, 'aic')
    bic = _metric(model, 'bic')
    return {'Mô hình': name, 'Tham số': 'ARIMA(p,d,q)', 'Giá trị': f'({p},{d},{q})', 'AIC': aic, 'BIC': bic}

def summarize_xgb(model, feature_names=None, name='XGBoost'):
    imp = None
    try:
        imp = model.feature_importances_
    except Exception:
        imp = None
    if imp is not None and feature_names is not None and len(feature_names) == len(imp):
        order = np.argsort(imp)[::-1]
        top = [(feature_names[i], float(imp[i])) for i in order[:5]]
        txt = ', '.join([f'{k}:{v:.3f}' for k, v in top])
    elif imp is not None:
        order = np.argsort(imp)[::-1]
        top = [(str(i), float(imp[i])) for i in order[:5]]
        txt = ', '.join([f'{k}:{v:.3f}' for k, v in top])
    else:
        txt = '-'
    return {'Mô hình': name, 'Tham số': 'Top Features', 'Giá trị': txt, 'AIC': np.nan, 'BIC': np.nan}

def summarize_lstm(epochs, name='LSTM'):
    return {'Mô hình': name, 'Tham số': 'Epochs', 'Giá trị': int(epochs), 'AIC': np.nan, 'BIC': np.nan}

def build_summary_figure(rows):
    df = pd.DataFrame(rows)
    df = df[['Mô hình','Tham số','Giá trị','AIC','BIC']]
    header = dict(values=list(df.columns), fill_color='#1f2937', font=dict(color='#e6e8eb'))
    cells = dict(values=[df[c].tolist() for c in df.columns], fill_color='#0b1220', font=dict(color='#e6e8eb'))
    fig = go.Figure(data=[go.Table(header=header, cells=cells)])
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=30,b=20,l=20,r=20))
    return fig
