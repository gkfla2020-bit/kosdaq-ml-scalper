"""
코스닥 ML 스캘핑 봇 v4 — 초고위험 풀시드
==========================================
전략: XGBoost+LightGBM 앙상블 → 1종목 풀시드 올인
  - 09:05 ML 예측 → 확률 최고 1종목 풀시드 매수
  - 장중 트레일링 스탑 (+1.5% 활성화, 고점-1% 청산)
  - 부분익절: +3% 시 절반 매도
  - 손절: -4%
  - 15:20 강제 청산 (인트라데이)
  - 매일 자동 실행, 주말 스킵, 텔레그램 알림

사용법:
  python kosdaq_scalper.py              # 모의투자
  python kosdaq_scalper.py --live       # 실전
  python kosdaq_scalper.py --dry        # 드라이런

환경변수 (.env):
  API_KEY=한국투자증권_앱키
  API_SECRET=한국투자증권_앱시크릿
  ACCOUNT_NO=계좌번호-01
  TG_TOKEN=텔레그램_봇_토큰
  TG_CHAT=텔레그램_채팅_ID
"""
import os, sys, json, time, logging, argparse
import urllib3, requests
import numpy as np
import pandas as pd
import FinanceDataReader as fdr
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime, timedelta
from pathlib import Path

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
_orig_request = requests.Session.request
def _no_verify(self, method, url, **kwargs):
    kwargs.setdefault('verify', False)
    return _orig_request(self, method, url, **kwargs)
requests.Session.request = _no_verify

# ═══════════════════════════════════════
#  CLI
# ═══════════════════════════════════════
parser = argparse.ArgumentParser()
g = parser.add_mutually_exclusive_group()
g.add_argument('--live', action='store_true')
g.add_argument('--dry', action='store_true')
args = parser.parse_args()
MODE = 'live' if args.live else ('dry' if args.dry else 'paper')

# ═══════════════════════════════════════
#  설정
# ═══════════════════════════════════════
DIR = Path(__file__).parent
ENV = {}
if (DIR/'.env').exists():
    for line in (DIR/'.env').read_text().splitlines():
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1); ENV[k.strip()] = v.strip()

PAPER_KEY    = ENV.get('API_KEY', '---')
PAPER_SECRET = ENV.get('API_SECRET', '---')
PAPER_ACCT   = ENV.get('ACCOUNT_NO', '00000000-01')
LIVE_KEY     = ENV.get('LIVE_API_KEY', '')
LIVE_SECRET  = ENV.get('LIVE_API_SECRET', '')
LIVE_ACCT    = ENV.get('LIVE_ACCOUNT_NO', '')

TG_TOKEN = ENV.get('TG_TOKEN', '')
TG_CHAT  = ENV.get('TG_CHAT', '')

if MODE == 'live':
    APP_KEY, APP_SECRET, ACCT = LIVE_KEY, LIVE_SECRET, LIVE_ACCT
    BASE = "https://openapi.koreainvestment.com:9443"
    MKT  = BASE
    TR_BUY, TR_SELL = "TTTC0802U", "TTTC0801U"
else:
    APP_KEY, APP_SECRET, ACCT = PAPER_KEY, PAPER_SECRET, PAPER_ACCT
    BASE = "https://openapivts.koreainvestment.com:9443"
    MKT  = "https://openapi.koreainvestment.com:9443"
    TR_BUY, TR_SELL = "VTTC0802U", "VTTC0801U"

CANO, ACNT = ACCT.split("-")
INVEST       = 80_000_000
STOP_LOSS    = 0.04       # -4% 손절
PARTIAL_TP   = 0.03       # +3% 절반 익절
TRAIL_ACT    = 0.015      # +1.5% 트레일링 활성화
TRAIL_STOP   = 0.01       # 고점 대비 -1% 트레일링 청산
CLOSE_TIME   = "152000"   # 15:20 강제 청산
BUY_TIME     = "090500"   # 09:05 매수
LIMIT_WAIT   = 8          # 지정가 대기 초
MON_INTERVAL = 5          # 모니터링 간격 초
RETRAIN_DAYS = 15         # ML 재학습 주기
DAILY_MAX_LOSS = -3_200_000  # 일일 최대 손실 (-4% of 80M)
API_TIMEOUT  = 10
API_RETRIES  = 3

# ═══════════════════════════════════════
#  로깅 + 텔레그램
# ═══════════════════════════════════════
log = logging.getLogger('bot')
log.setLevel(logging.INFO)
_f = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
log.addHandler(logging.StreamHandler(sys.stdout))
log.addHandler(logging.FileHandler(DIR/'scalper.log', encoding='utf-8'))
for h in log.handlers: h.setFormatter(_f)

def tg(msg):
    if not TG_TOKEN: return
    try:
        for i in range(0, len(msg), 4000):
            requests.post(f'https://api.telegram.org/bot{TG_TOKEN}/sendMessage',
                json={'chat_id': TG_CHAT, 'text': msg[i:i+4000], 'parse_mode': 'HTML'}, timeout=10)
    except: pass

# ═══════════════════════════════════════
#  토큰
# ═══════════════════════════════════════
TOKEN = None
TOKEN_EXP = None
HDR = {}

def get_token():
    global TOKEN, TOKEN_EXP, HDR
    r = requests.post(f"{BASE}/oauth2/tokenP",
        json={"grant_type":"client_credentials","appkey":APP_KEY,"appsecret":APP_SECRET}).json()
    TOKEN = r.get("access_token","")
    if not TOKEN: raise Exception(f"토큰 실패: {r}")
    TOKEN_EXP = datetime.now() + timedelta(hours=23)
    HDR.update({"authorization":f"Bearer {TOKEN}","appkey":APP_KEY,"appsecret":APP_SECRET,"Content-Type":"application/json"})
    log.info(f"토큰 발급 (만료: {TOKEN_EXP:%H:%M})")

def check_token():
    if TOKEN is None or datetime.now() >= TOKEN_EXP - timedelta(minutes=30): get_token()

# ═══════════════════════════════════════
#  안전 요청 헬퍼 (재시도 + 타임아웃)
# ═══════════════════════════════════════
def _safe_get(url, headers, params=None):
    for attempt in range(1, API_RETRIES + 1):
        try:
            res = requests.get(url, headers=headers, params=params, timeout=API_TIMEOUT)
            res.raise_for_status()
            return res
        except Exception as e:
            log.warning(f"[API] GET 재시도 {attempt}/{API_RETRIES}: {e}")
            if attempt < API_RETRIES: time.sleep(2)
    return None

def _safe_post(url, headers, json_body):
    try:
        res = requests.post(url, headers=headers, json=json_body, timeout=API_TIMEOUT)
        res.raise_for_status()
        return res
    except Exception as e:
        log.error(f"[API] POST 실패: {e}")
        return None

# ═══════════════════════════════════════
#  KIS API
# ═══════════════════════════════════════
def price(code):
    res = _safe_get(f"{MKT}/uapi/domestic-stock/v1/quotations/inquire-price",
        {**HDR,"tr_id":"FHKST01010100"},
        {"FID_COND_MRKT_DIV_CODE":"J","FID_INPUT_ISCD":code})
    if res is None:
        log.warning(f"시세 조회 실패: {code}")
        return {"price":0,"ask1":0,"open":0,"high":0,"prev":0,"vol":0,"chg":0.0,"name":code}
    r = res.json().get("output",{})
    return {"price":int(r.get("stck_prpr",0)),"ask1":int(r.get("askp1",0)),
            "open":int(r.get("stck_oprc",0)),"high":int(r.get("stck_hgpr",0)),
            "prev":int(r.get("stck_sdpr",0)),"vol":int(r.get("acml_vol",0)),
            "chg":float(r.get("prdy_ctrt",0)),"name":r.get("hts_kor_isnm",code)}

def vol_rank():
    res = _safe_get(f"{MKT}/uapi/domestic-stock/v1/quotations/volume-rank",
        {**HDR,"tr_id":"FHPST01710000"},
        {"FID_COND_MRKT_DIV_CODE":"J","FID_COND_SCR_DIV_CODE":"20171",
                "FID_INPUT_ISCD":"0000","FID_DIV_CLS_CODE":"0","FID_BLNG_CLS_CODE":"0",
                "FID_TRGT_CLS_CODE":"111111111","FID_TRGT_EXLS_CLS_CODE":"000000",
                "FID_INPUT_PRICE_1":"500","FID_INPUT_PRICE_2":"100000",
                "FID_VOL_CNT":"0","FID_INPUT_DATE_1":""})
    if res is None: return []
    r = res.json().get("output",[])
    return [i for i in r if len(i.get("mksc_shrn_iscd",""))==6 and not i.get("mksc_shrn_iscd","").startswith("0")]

def pending(code):
    tr = "TTTC8036R" if MODE=="live" else "VTTC8036R"
    res = _safe_get(f"{BASE}/uapi/domestic-stock/v1/trading/inquire-psbl-rvsecncl",
        {**HDR,"tr_id":tr},
        {"CANO":CANO,"ACNT_PRDT_CD":ACNT,"CTX_AREA_FK100":"","CTX_AREA_NK100":"",
                "INQR_DVSN_1":"0","INQR_DVSN_2":"0"})
    if res is None: return []
    return [o for o in res.json().get("output",[]) if o.get("pdno")==code]

def cancel(odno, code, qty):
    tr = "TTTC0803U" if MODE=="live" else "VTTC0803U"
    res = _safe_post(f"{BASE}/uapi/domestic-stock/v1/trading/order-rvsecncl",
        {**HDR,"tr_id":tr},
        {"CANO":CANO,"ACNT_PRDT_CD":ACNT,"KRX_FWDG_ORD_ORGNO":"","ORGN_ODNO":odno,
              "ORD_DVSN":"00","RVSE_CNCL_DVSN_CD":"02","ORD_QTY":str(qty),"ORD_UNPR":"0","QTY_ALL_ORD_YN":"Y"})
    if res is None: return {"rt_cd":"E","msg1":"통신 오류"}
    return res.json()

def order(code, side, qty, px=0):
    if MODE=='dry':
        log.info(f"[DRY] {side} {code} {qty}주 {'시장가' if px==0 else f'{px:,}원'}")
        return {"rt_cd":"0","output":{"ODNO":"DRY000"}}
    tr = TR_BUY if side=="BUY" else TR_SELL
    res = _safe_post(f"{BASE}/uapi/domestic-stock/v1/trading/order-cash",
        {**HDR,"tr_id":tr},
        {"CANO":CANO,"ACNT_PRDT_CD":ACNT,"PDNO":code,
              "ORD_DVSN":"01" if px==0 else "00","ORD_QTY":str(qty),"ORD_UNPR":str(px)})
    if res is None:
        log.error(f"[주문] ❌ {side} {code} — 통신 오류")
        return {"rt_cd":"E","msg1":"주문 통신 오류"}
    r = res.json()
    ok = "✅" if r.get("rt_cd")=="0" else "❌"
    log.info(f"[주문] {ok} {side} {code} {qty}주 → {r.get('msg1','')}")
    return r

def tick(px):
    if px<1000: return 1
    if px<5000: return 5
    if px<10000: return 10
    if px<50000: return 50
    if px<100000: return 100
    if px<500000: return 500
    return 1000

def balance():
    tr = "TTTC8434R" if MODE=="live" else "VTTC8434R"
    res = _safe_get(f"{BASE}/uapi/domestic-stock/v1/trading/inquire-balance",
        {**HDR,"tr_id":tr},
        {"CANO":CANO,"ACNT_PRDT_CD":ACNT,"AFHR_FLPR_YN":"N",
                "OFL_YN":"","INQR_DVSN":"02","UNPR_DVSN":"01",
                "FUND_STTL_ICLD_YN":"N","FNCG_AMT_AUTO_RDPT_YN":"N",
                "PRCS_DVSN":"01","CTX_AREA_FK100":"","CTX_AREA_NK100":""})
    if res is None: return [], 0
    r = res.json()
    holdings = []
    for h in r.get("output1",[]):
        if int(h.get("hldg_qty",0)) > 0:
            holdings.append({"code":h["pdno"],"name":h.get("prdt_name",""),
                "qty":int(h["hldg_qty"]),"avg":float(h.get("pchs_avg_pric",0)),
                "cur":int(h.get("prpr",0)),"pnl":float(h.get("evlu_pfls_rt",0))})
    cash = int(r.get("output2",[{}])[0].get("dnca_tot_amt",0))
    return holdings, cash

# ═══════════════════════════════════════
#  ML 피처 엔지니어링 (31개 피처)
# ═══════════════════════════════════════
FCOLS = None

def compute_features(df):
    """31개 피처 계산 (shift 1일 적용, 미래정보 없음)"""
    f = pd.DataFrame(index=df.index)
    c, h, l, o, v = df['Close'], df['High'], df['Low'], df['Open'], df['Volume']
    for d in [1,3,5,10,20]: f[f'ret_{d}d'] = c.pct_change(d)
    f['gap'] = (o - c.shift(1)) / c.shift(1)
    for w in [5,10,20,60]: f[f'ma{w}_r'] = c / c.rolling(w).mean() - 1
    f['ma_cross'] = (c.rolling(5).mean() - c.rolling(20).mean()) / c.rolling(20).mean()
    delta = c.diff()
    for p in [7,14]:
        g = delta.where(delta>0,0).rolling(p).mean()
        lo = (-delta.where(delta<0,0)).rolling(p).mean()
        f[f'rsi_{p}'] = 100 - 100/(1 + g/lo.replace(0, np.nan))
    macd_line = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    sig = macd_line.ewm(span=9).mean()
    f['macd'] = macd_line / c
    f['macd_hist'] = (macd_line - sig) / c
    ma20 = c.rolling(20).mean(); std20 = c.rolling(20).std()
    f['bb_pct'] = (c - (ma20 - 2*std20)) / (4*std20)
    f['bb_w'] = 4*std20 / ma20
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    f['atr_14'] = tr.rolling(14).mean() / c
    f['vol_r5'] = v / v.rolling(5).mean()
    f['vol_r20'] = v / v.rolling(20).mean()
    f['body_r'] = (c-o).abs() / (h-l).replace(0, np.nan)
    f['bullish'] = (c > o).astype(int)
    f['vol_5'] = c.pct_change().rolling(5).std()
    f['vol_20'] = c.pct_change().rolling(20).std()
    l14, h14 = l.rolling(14).min(), h.rolling(14).max()
    f['stoch_k'] = (c - l14) / (h14 - l14).replace(0, np.nan) * 100
    f['will_r'] = (h14 - c) / (h14 - l14).replace(0, np.nan) * -100
    tp = (h + l + c) / 3
    f['cci'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
    f['dow'] = pd.to_datetime(df.index).dayofweek
    f['bullish_3d'] = f['bullish'].rolling(3).mean()
    f['high_dist'] = (h.rolling(20).max() - c) / c
    f['low_dist'] = (c - l.rolling(20).min()) / c
    f['tgt'] = ((c - o) / o > 0).astype(int)
    skip = {'tgt', 'dow'}
    for col in f.columns:
        if col not in skip: f[col] = f[col].shift(1)
    return f.dropna()

# ═══════════════════════════════════════
#  ML 모델 학습 + 예측
# ═══════════════════════════════════════
XGB_MODEL = None
LGB_MODEL = None
SCALER = StandardScaler()
LAST_TRAIN = None

def get_universe():
    listing = fdr.StockListing('KOSDAQ')
    listing = listing[listing['Volume'] > 0]
    return listing.sort_values('Volume', ascending=False)['Code'].head(100).tolist()

def train_models():
    global XGB_MODEL, LGB_MODEL, SCALER, FCOLS, LAST_TRAIN
    log.info("🧠 ML 모델 학습 시작...")
    tg("🧠 ML 모델 학습 시작")
    codes = get_universe()
    end_dt = datetime.now().strftime('%Y-%m-%d')
    start_dt = (datetime.now() - timedelta(days=300)).strftime('%Y-%m-%d')
    all_rows, ok = [], 0
    for i, c in enumerate(codes):
        try:
            df = fdr.DataReader(c, start_dt, end_dt)
            if df is not None and len(df) > 80:
                ft = compute_features(df)
                if len(ft) > 30:
                    for date in ft.index:
                        r = ft.loc[date].to_dict(); r['code'] = c; r['date'] = date; all_rows.append(r)
                    ok += 1
        except: pass
        if (i+1) % 25 == 0: log.info(f"  데이터 수집 {i+1}/100 ({ok}개 성공)")
        time.sleep(0.03)
    if len(all_rows) < 1000:
        log.warning(f"학습 데이터 부족: {len(all_rows)}행"); return False
    master = pd.DataFrame(all_rows)
    FCOLS = [c for c in master.columns if c not in ['tgt','code','date']]
    X = np.clip(np.nan_to_num(master[FCOLS].values.astype(float)), -1e6, 1e6)
    y = master['tgt'].values.astype(int)
    SCALER.fit(X)
    Xs = np.clip(SCALER.transform(X), -10, 10)
    XGB_MODEL = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.6, min_child_weight=10,
        gamma=0.2, reg_alpha=0.5, reg_lambda=2.0,
        eval_metric='logloss', verbosity=0, random_state=42, n_jobs=-1)
    XGB_MODEL.fit(Xs, y)
    LGB_MODEL = lgb.LGBMClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.6, min_child_weight=10,
        reg_alpha=0.5, reg_lambda=2.0, verbosity=-1, random_state=42, n_jobs=-1)
    LGB_MODEL.fit(Xs, y)
    LAST_TRAIN = datetime.now()
    log.info(f"✅ 학습 완료: {len(all_rows):,}행, {ok}종목, 피처 {len(FCOLS)}개")
    tg(f"✅ ML 학습 완료\n{len(all_rows):,}행 | {ok}종목")
    return True

def predict_best_stock():
    if XGB_MODEL is None or LGB_MODEL is None: return None
    check_token()
    vr = vol_rank()
    candidates = []
    for item in vr[:30]:
        code = item.get("mksc_shrn_iscd","")
        if not code or len(code) != 6: continue
        name = item.get("hts_kor_isnm", code)
        try:
            end_dt = datetime.now().strftime('%Y-%m-%d')
            start_dt = (datetime.now() - timedelta(days=120)).strftime('%Y-%m-%d')
            df = fdr.DataReader(code, start_dt, end_dt)
            if df is None or len(df) < 60: continue
            ft = compute_features(df)
            if len(ft) == 0: continue
            last = ft.iloc[-1]
            X = np.clip(np.nan_to_num(last[FCOLS].values.astype(float).reshape(1,-1)), -1e6, 1e6)
            Xs = np.clip(SCALER.transform(X), -10, 10)
            prob = 0.5 * XGB_MODEL.predict_proba(Xs)[:,1][0] + 0.5 * LGB_MODEL.predict_proba(Xs)[:,1][0]
            p = price(code)
            candidates.append({"code": code, "name": name, "prob": float(prob),
                "price": p["price"], "ask1": p["ask1"], "vol": p["vol"]})
        except Exception as e:
            log.debug(f"  {code} 스킵: {e}")
        time.sleep(0.05)
    if not candidates: return None
    candidates.sort(key=lambda x: x["prob"], reverse=True)
    best = candidates[0]
    log.info(f"🎯 ML 예측 TOP: {best['name']}({best['code']}) prob={best['prob']:.3f}")
    top5 = "\n".join([f"  {c['name']}({c['code']}) p={c['prob']:.3f}" for c in candidates[:5]])
    tg(f"🎯 ML 예측 TOP 5\n{top5}")
    if best["prob"] < 0.50:
        log.info(f"확률 {best['prob']:.3f} < 0.50 → 매수 스킵"); return None
    return best

# ═══════════════════════════════════════
#  포지션 관리
# ═══════════════════════════════════════
class Position:
    def __init__(self, code, name, entry_px, qty):
        self.code, self.name = code, name
        self.entry_px, self.qty, self.init_qty = entry_px, qty, qty
        self.high_px = entry_px
        self.trail_active, self.partial_done = False, False
        self.entry_time = datetime.now()
    def pnl(self, px): return (px - self.entry_px) / self.entry_px
    def update_high(self, px):
        if px > self.high_px: self.high_px = px
        if not self.trail_active and self.pnl(px) >= TRAIL_ACT:
            self.trail_active = True
            log.info(f"🔔 트레일링 활성화 (고점: {self.high_px:,})")

def enter_position(stock):
    code, name = stock["code"], stock["name"]
    px, ask1 = stock["price"], stock["ask1"]
    qty = max(1, INVEST // px)
    limit_px = (ask1 if ask1 > 0 else px) + tick(px)
    log.info(f"[매수] {name}({code}) 지정가 {limit_px:,}원 × {qty:,}주")
    tg(f"🟢 매수 시도\n{name}({code})\n{limit_px:,}원 × {qty:,}주 = {limit_px*qty:,.0f}원")
    r = order(code, "BUY", qty, px=limit_px)
    if r.get("rt_cd") != "0":
        r2 = order(code, "BUY", qty, px=0)
        if r2.get("rt_cd") != "0":
            tg(f"❌ 매수 실패\n{name}({code})"); return None
        return Position(code, name, px, qty)
    odno = r.get("output",{}).get("ODNO","")
    log.info(f"  지정가 대기 {LIMIT_WAIT}초...")
    time.sleep(LIMIT_WAIT)
    pend = pending(code)
    if pend:
        log.info("  미체결 → 취소 후 시장가 전환")
        cancel(odno, code, qty); time.sleep(0.3)
        r2 = order(code, "BUY", qty, px=0)
        if r2.get("rt_cd") != "0": return None
        actual_px = px
    else:
        actual_px = limit_px
        log.info(f"  ✅ 지정가 체결 {limit_px:,}원")
    pos = Position(code, name, actual_px, qty)
    tg(f"✅ 매수 체결\n{name}({code})\n{actual_px:,}원 × {qty:,}주")
    return pos

def monitor_position(pos):
    log.info(f"📡 모니터링 시작: {pos.name}({pos.code})")
    while True:
        now = datetime.now()
        now_str = now.strftime("%H%M%S")
        try:
            p = price(pos.code); cur = p["price"]
        except:
            time.sleep(MON_INTERVAL); continue
        if cur == 0:
            log.warning(f"⚠️ {pos.name} 시세 0 — 대기"); time.sleep(MON_INTERVAL); continue
        pnl = pos.pnl(cur); pos.update_high(cur); reason = None
        if pnl <= -STOP_LOSS: reason = f"❌ 손절 {pnl*100:.2f}%"
        elif not pos.partial_done and pnl >= PARTIAL_TP:
            half = pos.qty // 2
            if half > 0:
                log.info(f"💰 부분익절 +{pnl*100:.1f}% → {half}주 매도")
                order(pos.code, "SELL", half, px=0)
                pos.qty -= half; pos.partial_done = True
                tg(f"💰 부분익절\n{pos.name} +{pnl*100:.1f}%\n{half}주 매도 (잔여 {pos.qty}주)")
        elif pos.trail_active:
            drop = (pos.high_px - cur) / pos.high_px
            if drop >= TRAIL_STOP:
                reason = f"📉 트레일링 청산 (고점 {pos.high_px:,} → {cur:,}, -{drop*100:.1f}%)"
        if now_str >= CLOSE_TIME: reason = f"⏰ 장마감 청산 {pnl*100:.2f}%"
        if reason:
            log.info(f"[청산] {reason}")
            order(pos.code, "SELL", pos.qty, px=0)
            pnl_amt = int((cur - pos.entry_px) * pos.init_qty)
            result = {"date": now.strftime("%Y-%m-%d"), "code": pos.code, "name": pos.name,
                "entry": pos.entry_px, "exit": cur, "qty": pos.init_qty,
                "pnl_pct": round(pnl*100, 2), "pnl_krw": pnl_amt, "reason": reason}
            tg(f"{'🟢' if pnl>=0 else '🔴'} 청산\n{pos.name}({pos.code})\n"
               f"{pos.entry_px:,} → {cur:,}\n{pnl*100:+.2f}% ({pnl_amt:+,}원)\n{reason}")
            return result
        elapsed = (now - pos.entry_time).seconds
        if elapsed % 30 < MON_INTERVAL + 1:
            log.info(f"  {pos.name} {cur:,}원 {pnl*100:+.2f}% 잔여:{pos.qty}주")
        time.sleep(MON_INTERVAL)

# ═══════════════════════════════════════
#  일일 리포트
# ═══════════════════════════════════════
def daily_report(result):
    log_file = DIR / "trade_log.json"
    logs = []
    if log_file.exists():
        try: logs = json.loads(log_file.read_text())
        except: logs = []
    if result: logs.append(result)
    log_file.write_text(json.dumps(logs, ensure_ascii=False, indent=2))
    if logs:
        wins = sum(1 for t in logs if t.get("pnl_krw",0) > 0)
        total_pnl = sum(t.get("pnl_krw",0) for t in logs)
        msg = (f"📊 누적 성적\n총 {len(logs)}거래 | 승 {wins} | 패 {len(logs)-wins}\n"
               f"승률: {wins/len(logs)*100:.0f}%\n누적 손익: {total_pnl:+,}원")
        log.info(msg); tg(msg)

def is_trading_day():
    return datetime.now().weekday() < 5

def wait_until(target_hhmm):
    while True:
        now = datetime.now()
        if now.strftime("%H%M") >= target_hhmm: return
        remain = (int(target_hhmm[:2])*60 + int(target_hhmm[2:])) - (now.hour*60 + now.minute)
        if remain > 5:
            log.info(f"⏳ {target_hhmm} 대기 중... (약 {remain}분 남음)"); time.sleep(60)
        else: time.sleep(10)

# ═══════════════════════════════════════
#  메인 루프
# ═══════════════════════════════════════
def main():
    log.info("=" * 60)
    log.info(f"🚀 코스닥 ML 스캘핑 봇 v4 시작 [{MODE.upper()}]")
    log.info(f"   투자금: {INVEST:,}원 | 손절: -{STOP_LOSS*100:.0f}% | 익절: +{PARTIAL_TP*100:.0f}%")
    log.info(f"   트레일링: +{TRAIL_ACT*100:.1f}% 활성화, 고점-{TRAIL_STOP*100:.0f}% 청산")
    log.info(f"   ML 재학습: {RETRAIN_DAYS}일 주기")
    log.info("=" * 60)
    tg(f"🚀 봇 시작 [{MODE.upper()}]\n투자금: {INVEST:,}원")

    while True:
        try:
            if not is_trading_day():
                now = datetime.now()
                days_until_mon = (7 - now.weekday()) % 7 or 7
                next_mon = now.replace(hour=8, minute=30, second=0) + timedelta(days=days_until_mon)
                sleep_sec = (next_mon - now).total_seconds()
                log.info(f"🌙 주말 — {next_mon:%m/%d %H:%M}까지 대기")
                time.sleep(min(sleep_sec, 3600)); continue

            wait_until("0850")
            check_token()

            if LAST_TRAIN is None or (datetime.now() - LAST_TRAIN).days >= RETRAIN_DAYS:
                if not train_models():
                    log.error("ML 학습 실패 → 오늘 스킵"); time.sleep(3600); continue

            wait_until("0905")
            check_token()
            best = predict_best_stock()
            if best is None:
                log.info("오늘 매수 종목 없음"); wait_until("1530"); time.sleep(60); continue

            pos = enter_position(best)
            if pos is None:
                log.error("매수 실패"); wait_until("1530"); time.sleep(60); continue

            result = monitor_position(pos)

            if result and result.get("pnl_krw", 0) <= DAILY_MAX_LOSS:
                log.warning(f"🛑 일일 최대 손실 한도: {result['pnl_krw']:+,}원")
                daily_report(result); wait_until("1530"); time.sleep(60); continue

            daily_report(result)
            now = datetime.now()
            tomorrow = (now + timedelta(days=1)).replace(hour=8, minute=30, second=0, microsecond=0)
            sleep_sec = (tomorrow - now).total_seconds()
            log.info(f"💤 오늘 매매 종료 — {tomorrow:%m/%d %H:%M}까지 대기")
            time.sleep(max(sleep_sec, 60))

        except KeyboardInterrupt:
            log.info("🛑 사용자 중단"); tg("🛑 봇 수동 중단")
            try:
                holdings, _ = balance()
                if holdings:
                    msg = "⚠️ 보유 종목:\n" + "\n".join(f"  {h['name']}({h['code']}) {h['qty']}주" for h in holdings)
                    log.warning(msg); tg(msg)
            except: pass
            break
        except Exception as e:
            log.exception(f"에러: {e}"); tg(f"⚠️ 에러\n{str(e)[:500]}")
            time.sleep(300)

if __name__ == "__main__":
    main()
