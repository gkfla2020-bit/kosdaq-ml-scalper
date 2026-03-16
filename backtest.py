"""
코스닥 ML 백테스트 최종판
==========================
3가지 전략 동시 비교:
A) 오버나이트: 종가 매수 → 익일 시가 청산 (갭 수익)
B) 인트라데이: 시가 매수 → 종가 청산 (양봉 예측)
C) 하이브리드: 시가 매수 → 장중 고가 50% 청산 (부분익절)

XGBoost + LightGBM 앙상블 | Walk-Forward | 롱 온리
"""
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import time, json, os, warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

INVEST = 80_000_000
COMM = 0.0015
START, END = '2024-06-01', '2026-03-14'
TRADE_START = '2025-03-01'
TOP_N = 5
RETRAIN = 15

print("=" * 70)
print("🧠 코스닥 ML 백테스트 최종판")
print("=" * 70)
print(f"전략 A: 오버나이트 (종가매수→시가청산)")
print(f"전략 B: 인트라데이 (시가매수→종가청산)")
print(f"전략 C: 하이브리드 (시가매수→고가50%+종가50%)")
print(f"모델: XGB+LGBM | 재학습: {RETRAIN}일 | TOP {TOP_N}\n")

# ── 데이터 ──
print("[1] 데이터 수집...")
listing = fdr.StockListing('KOSDAQ')
listing = listing[listing['Volume'] > 0]
codes = listing.sort_values('Volume', ascending=False)['Code'].head(100).tolist()
names = dict(zip(listing['Code'], listing['Name']))
raw = {}
for i, c in enumerate(codes):
    try:
        df = fdr.DataReader(c, START, END)
        if df is not None and len(df) > 80: raw[c] = df
    except: pass
    if (i+1) % 25 == 0: print(f"  {i+1}/100")
    time.sleep(0.02)
print(f"  {len(raw)}개 종목")

# ── 피처 ──
print("\n[2] 피처...")
def features(df):
    f = pd.DataFrame(index=df.index)
    c, h, l, o, v = df['Close'], df['High'], df['Low'], df['Open'], df['Volume']
    for d in [1,3,5,10,20]: f[f'ret_{d}d'] = c.pct_change(d)
    f['gap'] = (o - c.shift(1)) / c.shift(1)
    for w in [5,10,20,60]: f[f'ma{w}_r'] = c / c.rolling(w).mean() - 1
    f['ma_cross'] = (c.rolling(5).mean()-c.rolling(20).mean())/c.rolling(20).mean()
    delta = c.diff()
    for p in [7,14]:
        g = delta.where(delta>0,0).rolling(p).mean()
        lo = (-delta.where(delta<0,0)).rolling(p).mean()
        f[f'rsi_{p}'] = 100-100/(1+g/lo.replace(0,np.nan))
    macd = c.ewm(span=12).mean()-c.ewm(span=26).mean()
    sig = macd.ewm(span=9).mean()
    f['macd'] = macd/c; f['macd_hist'] = (macd-sig)/c
    ma20 = c.rolling(20).mean(); std20 = c.rolling(20).std()
    f['bb_pct'] = (c-(ma20-2*std20))/(4*std20); f['bb_w'] = 4*std20/ma20
    tr = pd.concat([h-l,(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    f['atr_14'] = tr.rolling(14).mean()/c
    f['vol_r5'] = v/v.rolling(5).mean(); f['vol_r20'] = v/v.rolling(20).mean()
    f['body_r'] = (c-o).abs()/(h-l).replace(0,np.nan)
    f['bullish'] = (c>o).astype(int)
    f['vol_5'] = c.pct_change().rolling(5).std()
    f['vol_20'] = c.pct_change().rolling(20).std()
    l14,h14 = l.rolling(14).min(),h.rolling(14).max()
    f['stoch_k'] = (c-l14)/(h14-l14).replace(0,np.nan)*100
    f['will_r'] = (h14-c)/(h14-l14).replace(0,np.nan)*-100
    tp = (h+l+c)/3
    f['cci'] = (tp-tp.rolling(20).mean())/(0.015*tp.rolling(20).std())
    f['dow'] = pd.to_datetime(df.index).dayofweek
    f['bullish_3d'] = f['bullish'].rolling(3).mean()
    f['high_dist'] = (h.rolling(20).max()-c)/c
    f['low_dist'] = (c-l.rolling(20).min())/c
    # 3가지 타겟
    f['ret_intra'] = (c-o)/o                    # B: 시가→종가
    f['ret_overnight'] = (o.shift(-1)-c)/c      # A: 종가→익일시가
    f['ret_high'] = (h-o)/o                     # C: 시가→고가
    f['ret_hybrid'] = 0.5*(h-o)/o + 0.5*(c-o)/o # C: 고가50%+종가50%
    f['tgt_intra'] = (f['ret_intra']>0).astype(int)
    f['tgt_overnight'] = (f['ret_overnight']>0).astype(int)
    # Shift 피처 (타겟 제외)
    skip = {'tgt_intra','tgt_overnight','ret_intra','ret_overnight','ret_high','ret_hybrid','dow'}
    for col in f.columns:
        if col not in skip: f[col] = f[col].shift(1)
    return f.dropna()

feats = {}
for code, df in raw.items():
    ft = features(df)
    if len(ft) > 50: feats[code] = ft
print(f"  {len(feats)}개 종목")

rows = []
for code, ft in feats.items():
    for date in ft.index:
        r = ft.loc[date].to_dict(); r['code']=code; r['date']=date; rows.append(r)
master = pd.DataFrame(rows).sort_values('date')
dates = sorted(master['date'].unique())
trade_start_dt = pd.Timestamp(TRADE_START)
trade_dates = [d for d in dates if d >= trade_start_dt]
fcols = [c for c in master.columns if c not in [
    'tgt_intra','tgt_overnight','ret_intra','ret_overnight','ret_high','ret_hybrid','code','date']]
print(f"  {len(master):,}행 | 매매일: {len(trade_dates)}일 | 피처: {len(fcols)}개")

# ── Walk-Forward 함수 ──
def walk_forward(target_col, ret_col, threshold, label):
    print(f"\n{'='*60}")
    print(f"전략 {label} (임계: {threshold})")
    print(f"{'='*60}")
    xm = lm = None
    sc = StandardScaler()
    eq = [INVEST]; tr = []; dr = []; lt = -RETRAIN

    for ti, today in enumerate(trade_dates):
        if ti - lt >= RETRAIN:
            td = master[master['date'] < today]
            if len(td) < 500:
                eq.append(eq[-1]); dr.append(0); continue
            X = np.clip(np.nan_to_num(td[fcols].values.astype(float)), -1e6, 1e6)
            y = td[target_col].values.astype(int)
            sc.fit(X)
            Xs = np.clip(sc.transform(X), -10, 10)
            xm = xgb.XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.03,
                subsample=0.7, colsample_bytree=0.6, min_child_weight=10,
                gamma=0.2, reg_alpha=0.5, reg_lambda=2.0,
                eval_metric='logloss', verbosity=0, random_state=42, n_jobs=-1)
            xm.fit(Xs, y)
            lm = lgb.LGBMClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.03,
                subsample=0.7, colsample_bytree=0.6, min_child_weight=10,
                reg_alpha=0.5, reg_lambda=2.0, verbosity=-1, random_state=42, n_jobs=-1)
            lm.fit(Xs, y)
            lt = ti

        td_today = master[master['date'] == today]
        if len(td_today) == 0 or xm is None:
            eq.append(eq[-1]); dr.append(0); continue
        Xt = np.clip(np.nan_to_num(td_today[fcols].values.astype(float)), -1e6, 1e6)
        Xts = np.clip(sc.transform(Xt), -10, 10)
        prob = 0.5*xm.predict_proba(Xts)[:,1] + 0.5*lm.predict_proba(Xts)[:,1]

        td_today = td_today.copy()
        td_today['prob'] = prob
        top = td_today.nlargest(TOP_N, 'prob')
        top = top[top['prob'] > threshold]

        if len(top) > 0:
            avg_r = top[ret_col].mean() - 2*COMM
            eq.append(eq[-1]*(1+avg_r))
            dr.append(avg_r)
            for _, row in top.iterrows():
                tr.append({'date': str(today)[:10], 'code': row['code'],
                           'prob': round(row['prob'],3), 'ret': round(row[ret_col],4)})
        else:
            eq.append(eq[-1]); dr.append(0)

        if (ti+1) % 80 == 0:
            print(f"  [{ti+1}/{len(trade_dates)}] {(eq[-1]/INVEST-1)*100:+.1f}%")

    # 통계
    e = np.array(eq, dtype=float)
    ret = (e[-1]/e[0]-1)*100
    pk = np.maximum.accumulate(e)
    mdd = ((e-pk)/pk*100).min()
    w = sum(1 for t in tr if t['ret']>0)
    wr = w/max(len(tr),1)*100
    d = np.array(dr)
    sh = np.mean(d)/max(np.std(d),1e-10)*np.sqrt(252)
    active = sum(1 for r in dr if r!=0)
    print(f"  수익률: {ret:+.1f}% | MDD: {mdd:.1f}% | 샤프: {sh:.2f} | 거래: {len(tr)} | 승률: {wr:.0f}% | 매매일: {active}")
    return {'label': label, 'ret': ret, 'mdd': mdd, 'sharpe': sh, 'trades': len(tr),
            'wr': wr, 'final': e[-1], 'eq': eq, 'imp': xm.feature_importances_ if xm else None}

# ── 3가지 전략 실행 ──
print("\n[3] Walk-Forward 백테스트...")

# A: 오버나이트 (종가매수→시가청산)
rA = walk_forward('tgt_overnight', 'ret_overnight', 0.55, 'A: 오버나이트')

# B: 인트라데이 (시가매수→종가청산)
rB = walk_forward('tgt_intra', 'ret_intra', 0.58, 'B: 인트라데이')

# C: 하이브리드 (시가매수→고가50%+종가50%)
rC = walk_forward('tgt_intra', 'ret_hybrid', 0.58, 'C: 하이브리드')

# ── 최종 비교 ──
print("\n" + "=" * 70)
print("📊 최종 비교")
print("=" * 70)
print(f"{'전략':<30} {'수익률':>9} {'MDD':>9} {'샤프':>7} {'거래':>6} {'승률':>5}")
print("-" * 70)
for r in [rA, rB, rC]:
    print(f"{r['label']:<30} {r['ret']:>+8.1f}% {r['mdd']:>8.1f}% {r['sharpe']:>6.2f} {r['trades']:>6} {r['wr']:>4.0f}%")

best = max([rA, rB, rC], key=lambda x: x['sharpe'])
print(f"\n🏆 최고 (샤프): {best['label']}")
print(f"   수익률: {best['ret']:+.1f}% | MDD: {best['mdd']:.1f}% | 최종: {best['final']:,.0f}원")

if best['imp'] is not None:
    imp = sorted(zip(fcols, best['imp']), key=lambda x:x[1], reverse=True)
    print(f"\n  🔑 피처 TOP 10:")
    for fn, iv in imp[:10]:
        print(f"    {fn:<20} {iv:.4f}")

out = {'strategies': []}
for r in [rA, rB, rC]:
    out['strategies'].append({k:v for k,v in r.items() if k not in ['eq','imp']})
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml_backtest_results.json'), 'w') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"\n저장: ml_backtest_results.json")
