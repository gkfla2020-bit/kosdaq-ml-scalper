# 🧠 KOSDAQ ML Scalping Bot

코스닥 ML 기반 자동매매 봇. XGBoost + LightGBM 앙상블로 당일 최고 확률 1종목을 선정하고, 풀시드 인트라데이 매매를 수행합니다.

> ⚠️ **경고**: 이 봇은 초고위험 전략입니다. 반드시 모의투자로 충분히 테스트한 후 사용하세요.

## 전략 개요

```
09:05  ML 예측 → 거래량 상위 30종목 중 확률 최고 1종목 선정
       ↓ 풀시드 매수 (지정가 → 시장가 fallback)
       ↓ 실시간 모니터링
       ├── +3%  → 절반 익절
       ├── +1.5% → 트레일링 스탑 활성화 (고점 -1% 청산)
       ├── -4%  → 손절
       └── 15:20 → 강제 청산 (인트라데이)
```

## ML 모델

| 항목 | 내용 |
|------|------|
| 모델 | XGBoost + LightGBM 앙상블 (50:50) |
| 피처 | 31개 (수익률, 이평선, RSI, MACD, 볼린저, ATR, 거래량 등) |
| 타겟 | 당일 양봉 여부 (시가 < 종가) |
| 학습 데이터 | 코스닥 거래량 상위 100종목 × 300일 |
| 재학습 주기 | 15일 |
| Look-ahead 방지 | 모든 피처 shift(1) 적용 |

### 피처 목록

| 카테고리 | 피처 |
|----------|------|
| 수익률 | ret_1d, ret_3d, ret_5d, ret_10d, ret_20d, gap |
| 이평선 | ma5_r, ma10_r, ma20_r, ma60_r, ma_cross |
| 모멘텀 | rsi_7, rsi_14, macd, macd_hist, stoch_k, will_r, cci |
| 변동성 | bb_pct, bb_w, atr_14, vol_5, vol_20 |
| 거래량 | vol_r5, vol_r20 |
| 캔들 | body_r, bullish, bullish_3d |
| 기타 | high_dist, low_dist, dow |

## 백테스트 결과

3가지 전략 비교 (2025.03 ~ 2026.03, 코스닥 상위 100종목):

| 전략 | 수익률 | MDD | 샤프 | 거래 | 승률 |
|------|--------|-----|------|------|------|
| A: 오버나이트 (종가→시가) | +11.8% | -28.6% | 0.51 | - | - |
| B: 인트라데이 (시가→종가) | +42.3% | -15.2% | 1.82 | - | - |
| C: 하이브리드 (부분익절) | +163.1% | -8.7% | 3.35 | - | - |

> 백테스트에는 look-ahead bias가 포함될 수 있습니다. 실전 성과와 다를 수 있습니다.

## 설치

```bash
git clone https://github.com/gkfla2020/kosdaq-ml-scalper.git
cd kosdaq-ml-scalper
pip install -r requirements.txt
```

## 설정

1. [한국투자증권](https://securities.koreainvestment.com/) 모의투자 API 발급
2. `.env` 파일 생성:

```bash
cp .env.example .env
# .env 파일에 API 키 입력
```

3. (선택) 텔레그램 봇 생성 후 토큰/채팅ID 입력

## 실행

```bash
# 모의투자 (기본)
python kosdaq_scalper.py

# 드라이런 (주문 없이 로직만 테스트)
python kosdaq_scalper.py --dry

# 실전투자
python kosdaq_scalper.py --live
```

## 백테스트

```bash
python backtest.py
```

## 리스크 관리

- 손절: -4% (포지션 단위)
- 일일 최대 손실: -320만원 (투자금의 4%)
- 부분익절: +3% 시 절반 매도
- 트레일링 스탑: +1.5% 활성화, 고점 대비 -1% 청산
- 15:20 강제 청산 (오버나이트 리스크 제거)

## 파일 구조

```
kosdaq-ml-scalper/
├── kosdaq_scalper.py    # 메인 봇
├── backtest.py          # ML 백테스트
├── .env.example         # 환경변수 템플릿
├── .gitignore
├── requirements.txt
└── README.md
```

## 주의사항

- 이 봇은 교육/연구 목적입니다
- 실전 투자 시 발생하는 손실에 대해 책임지지 않습니다
- 반드시 모의투자로 충분히 검증 후 사용하세요
- KIS API 호출 제한에 주의하세요 (초당 20회)

## License

MIT
