"""
generate_dashboard.py
=====================
Run once to fetch real market data and bake it into dashboard.html.

    pip3 install yfinance pandas numpy
    python3 generate_dashboard.py

Opens dashboard.html in your browser automatically when done.
Re-run any morning to get fresh data.
"""

import json, webbrowser, os, sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

# ── CONFIG ────────────────────────────────────────────────────────────────────
MAG7_W = {"MSFT":0.25,"NVDA":0.25,"GOOGL":0.20,"META":0.15,"AMZN":0.10,"AAPL":0.05}
FWD_CAGR = {"SPY":0.08,"MAG7":0.11,"TSLA":0.15,"BTC":0.20}
VOLS     = {"SPY":0.16,"MAG7":0.22,"TSLA":0.65,"BTC":0.80}
ALL_TICKERS = list(MAG7_W.keys()) + ["SPY","TSLA","BTC-USD"]

print("Fetching market data via yfinance...")

# ── FETCH ─────────────────────────────────────────────────────────────────────
end   = datetime.today()
start = end - timedelta(days=365*11)

raw = yf.download(
    ALL_TICKERS,
    start=start.strftime("%Y-%m-%d"),
    end=end.strftime("%Y-%m-%d"),
    auto_adjust=True,
    progress=False,
)["Close"]

prices = {}
for col in raw.columns:
    s = raw[col].dropna()
    if len(s) > 20:
        key = "BTC" if col == "BTC-USD" else str(col)
        prices[key] = s
        print(f"  {key}: {len(s)} rows  ${s.iloc[0]:.2f} → ${s.iloc[-1]:.2f}")

# ── MAG7 BASKET ───────────────────────────────────────────────────────────────
components = []
for t, w in MAG7_W.items():
    s = prices.get(t)
    if s is not None:
        components.append((s / s.iloc[0] * 100) * w)
basket = pd.concat(components, axis=1).sum(axis=1)
prices["MAG7"] = basket / basket.iloc[0] * 100

# ── PROJECTIONS ───────────────────────────────────────────────────────────────
def biz_range(s, e):
    return pd.bdate_range(start=s + timedelta(days=1), end=e)

def proj_cagr(lv, ld, ed, rate):
    dates = biz_range(ld, ed)
    if dates.empty: return []
    days = np.arange(1, len(dates)+1)
    return [(str(d.date()), round(float(lv * (1+rate)**(i/252)), 2))
            for d, i in zip(dates, days)]

def proj_gbm(lv, ld, ed, rate, vol, seed):
    dates = biz_range(ld, ed)
    if dates.empty: return []
    rng = np.random.default_rng(seed=seed)
    dt = 1/252
    shocks = rng.normal((rate - 0.5*vol**2)*dt, vol*np.sqrt(dt), len(dates))
    vals = lv * np.exp(np.cumsum(shocks))
    return [(str(d.date()), round(float(v), 2)) for d, v in zip(dates, vals)]

def vol_bands(proj_pts, last_val, vol):
    upper, lower = [], []
    for i, (d, v) in enumerate(proj_pts):
        t = (i+1)/252
        f = np.exp(1.645 * vol * np.sqrt(t))
        upper.append((d, round(v*f, 2)))
        lower.append((d, round(v/f, 2)))
    return upper, lower

# ── SERIALIZE ALL DATA FOR EACH HORIZON ───────────────────────────────────────
today = pd.Timestamp.today().normalize()
LONG_H = {"10Y","5Y"}

horizons_cfg = {
    "10Y": {"back":pd.DateOffset(years=10),  "fwd":pd.DateOffset(years=10)},
    "5Y":  {"back":pd.DateOffset(years=5),   "fwd":pd.DateOffset(years=5)},
    "1Y":  {"back":pd.DateOffset(years=1),   "fwd":pd.DateOffset(years=1)},
    "6M":  {"back":pd.DateOffset(months=6),  "fwd":pd.DateOffset(months=6)},
    "30D": {"back":pd.DateOffset(days=30),   "fwd":pd.DateOffset(days=30)},
}

all_data = {}
for hkey, hcfg in horizons_cfg.items():
    hist_start = today - hcfg["back"]
    proj_end   = today + hcfg["fwd"]
    is_long    = hkey in LONG_H
    h_data     = {}

    for key in ["SPY","MAG7","TSLA","BTC"]:
        raw_s = prices.get(key)
        if raw_s is None: continue

        sliced = raw_s.loc[(raw_s.index >= hist_start) & (raw_s.index <= today)].dropna()
        if len(sliced) < 2: continue

        normed = sliced / sliced.iloc[0] * 100
        hist_pts = [(str(d.date()), round(float(v),2)) for d,v in normed.items()]

        lv   = float(normed.iloc[-1])
        ld   = normed.index[-1].to_pydatetime()
        rate = FWD_CAGR[key]
        vol  = VOLS[key]
        seed = abs(hash(f"{key}:{hkey}")) % (2**31)

        proj_pts = (proj_cagr(lv, ld, proj_end, rate) if is_long
                    else proj_gbm(lv, ld, proj_end, rate, vol, seed))

        entry = {
            "hist": hist_pts,
            "proj": proj_pts,
            "ret":  round(lv - 100, 2),
            "base_price": round(float(sliced.iloc[0]), 2),
            "base_date":  str(sliced.index[0].date()),
        }

        # CAGR
        yrs = (normed.index[-1] - normed.index[0]).days / 365.25
        entry["cagr"] = round((float(normed.iloc[-1]/normed.iloc[0])**(1/yrs)-1)*100, 1) if yrs > 0.1 else None

        # Vol bands for TSLA / BTC
        if key in ("TSLA","BTC") and proj_pts:
            u, l = vol_bands(proj_pts, lv, vol)
            entry["band_upper"] = u
            entry["band_lower"] = l

        h_data[key] = entry

    all_data[hkey] = h_data

fetched_at = datetime.now().strftime("%b %d, %Y %H:%M")
payload = json.dumps({"horizons": all_data, "fetched_at": fetched_at}, separators=(",",":"))

print(f"\nData ready. Generating dashboard.html...")

# ── HTML TEMPLATE ─────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Beta · Alpha · Asymmetric Bets</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@800&display=swap" rel="stylesheet"/>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{--bg:#07070f;--surf:#0f0f1a;--bdr:#1c1c2e;--bdr2:#2a2a40;--txt:#dcdcec;--mut:#5a5a80;--dim:#252538;--grn:#4ade80;--red:#f87171;--ylw:#fbbf24;--spy:#378ADD;--mag:#EF9F27;--tsl:#D85A30;--btc:#D4537E}}
html,body{{background:var(--bg);color:var(--txt);font-family:'Syne',sans-serif;font-size:14px;min-height:100vh}}
.page{{max-width:1200px;margin:0 auto;padding:28px 24px 48px}}
.hdr{{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:16px;flex-wrap:wrap;gap:12px}}
h1{{font-size:26px;font-weight:800;letter-spacing:-.5px;color:#eeeefa}}
.sub{{font-family:'DM Mono',monospace;font-size:10px;color:var(--mut);letter-spacing:1.5px;margin-top:3px}}
.fetch-btn{{display:inline-flex;align-items:center;gap:8px;font-family:'DM Mono',monospace;font-size:11px;font-weight:500;padding:9px 18px;border-radius:6px;border:1.5px solid var(--mag);background:transparent;color:var(--mag);cursor:pointer}}
.fetch-btn:hover{{background:rgba(239,159,39,.1)}}
.fetch-btn:disabled{{opacity:.45;cursor:not-allowed}}
.src{{font-family:'DM Mono',monospace;font-size:10px;padding:8px 12px;border-radius:5px;border:.5px solid;margin-bottom:14px;display:flex;align-items:center;gap:8px}}
.src.live{{background:rgba(74,222,128,.07);border-color:rgba(74,222,128,.3);color:var(--grn)}}
.src.synth{{background:rgba(251,191,36,.07);border-color:rgba(251,191,36,.3);color:var(--ylw)}}
.src-dot{{width:7px;height:7px;border-radius:50%;flex-shrink:0}}
.src.live .src-dot{{background:var(--grn)}}
.src.synth .src-dot{{background:var(--ylw)}}
.ctrls{{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:14px}}
.cl{{font-family:'DM Mono',monospace;font-size:10px;color:var(--mut)}}
.tog{{font-family:'DM Mono',monospace;font-size:10px;padding:4px 11px;border-radius:4px;border:.5px solid var(--bdr2);background:transparent;color:var(--mut);cursor:pointer}}
.tog.on{{background:var(--dim);color:var(--txt);border-color:var(--bdr)}}
.vsep{{width:1px;height:18px;background:var(--bdr);margin:0 4px}}
.tabs{{display:flex;border-bottom:1px solid var(--bdr);margin-bottom:14px}}
.tab{{font-family:'DM Mono',monospace;font-size:11px;padding:8px 18px;border:none;background:transparent;color:var(--mut);cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-1px}}
.tab.on{{color:var(--txt);border-bottom-color:var(--txt)}}
.cards{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-bottom:14px}}
.card{{background:var(--surf);border-radius:6px;border:.5px solid var(--bdr);border-top:2px solid transparent;padding:12px 14px}}
.card-lbl{{font-family:'DM Mono',monospace;font-size:9px;color:var(--mut);letter-spacing:1.2px;margin-bottom:5px}}
.card-val{{font-family:'DM Mono',monospace;font-size:22px;font-weight:500;margin-bottom:3px}}
.card-sub{{font-family:'DM Mono',monospace;font-size:10px;color:var(--mut)}}
.card-base{{font-family:'DM Mono',monospace;font-size:9px;color:var(--mut);opacity:.55;margin-top:3px}}
.pos{{color:var(--grn)}} .neg{{color:var(--red)}}
.legend{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:8px}}
.li{{display:flex;align-items:center;gap:5px;font-family:'DM Mono',monospace;font-size:10px;color:var(--mut)}}
.ln{{width:18px;height:2px;flex-shrink:0}}
.ld{{width:18px;height:2px;background-image:repeating-linear-gradient(90deg,currentColor 0,currentColor 4px,transparent 4px,transparent 8px);flex-shrink:0}}
.note{{font-family:'DM Mono',monospace;font-size:10px;color:var(--mut);opacity:.6;margin-bottom:8px;line-height:1.5}}
.chart-wrap{{position:relative;width:100%;height:440px}}
.outperf{{display:flex;gap:16px;flex-wrap:wrap;margin-top:8px;min-height:18px}}
.op{{font-family:'DM Mono',monospace;font-size:10px;color:var(--mut)}}
hr{{border:none;border-top:1px solid var(--bdr);margin:22px 0 10px}}
.footer{{font-family:'DM Mono',monospace;font-size:9px;color:var(--dim);line-height:1.9;text-align:center}}
@media(max-width:700px){{.cards{{grid-template-columns:repeat(2,minmax(0,1fr))}}.tab{{padding:7px 10px;font-size:10px}}}}
</style>
</head>
<body>
<div class="page">
  <div class="hdr">
    <div>
      <h1>Beta · Alpha · Asymmetric Bets</h1>
      <div class="sub">MULTI-HORIZON PERFORMANCE DASHBOARD</div>
    </div>
    <button class="fetch-btn" onclick="alert('To refresh data:\\n\\npython3 generate_dashboard.py\\n\\nThis re-fetches live prices and regenerates this file.')">
      ↻ How to refresh
    </button>
  </div>

  <div class="src live" id="srcBanner">
    <span class="src-dot"></span>
    <span id="srcTxt">Yahoo Finance via yfinance · Fetched {fetched_at} · Data baked in</span>
  </div>

  <div class="ctrls">
    <span class="cl">Scale</span>
    <button class="tog on" id="bl" onclick="setScale('linear')">Linear</button>
    <button class="tog"    id="bg" onclick="setScale('log')">Log</button>
    <div class="vsep"></div>
    <span class="cl">Vol bands</span>
    <button class="tog on" id="bbon"  onclick="setBands(true)">Show</button>
    <button class="tog"    id="bboff" onclick="setBands(false)">Hide</button>
  </div>

  <div class="tabs">
    <button class="tab on" onclick="setTab('10Y')">10Y / 10Y</button>
    <button class="tab"    onclick="setTab('5Y')">5Y / 5Y</button>
    <button class="tab"    onclick="setTab('1Y')">1Y / 1Y</button>
    <button class="tab"    onclick="setTab('6M')">6M / 6M</button>
    <button class="tab"    onclick="setTab('30D')">30D / 30D</button>
  </div>

  <div class="cards" id="cards"></div>

  <div class="legend">
    <span class="li"><span class="ln" style="background:var(--spy)"></span>SPY · Beta</span>
    <span class="li"><span class="ln" style="background:var(--mag)"></span>Mag7 · Alpha</span>
    <span class="li"><span class="ln" style="background:var(--tsl)"></span>Tesla</span>
    <span class="li"><span class="ln" style="background:var(--btc)"></span>Bitcoin</span>
    <span class="li"><span class="ld" style="color:#555"></span>Projection</span>
    <span class="li" id="bandLeg"><span style="display:inline-block;width:18px;height:7px;background:rgba(150,150,150,.15);border-radius:1px"></span>90% band</span>
  </div>

  <div class="note" id="note"></div>
  <div class="chart-wrap"><canvas id="chart" role="img" aria-label="Multi-horizon performance chart."></canvas></div>
  <div class="outperf" id="outperf"></div>

  <hr/>
  <div class="footer">
    Yahoo Finance via yfinance · MAG7: MSFT 25% / NVDA 25% / GOOGL 20% / META 15% / AMZN 10% / AAPL 5%<br>
    Long-term: CAGR geometric compounding · Short-term: Geometric Brownian Motion · Not financial advice
  </div>
</div>

<script>
const DATA = {payload};
const COLORS = {{SPY:'#378ADD',MAG7:'#EF9F27',TSLA:'#D85A30',BTC:'#D4537E'}};
const FWD    = {{SPY:8,MAG7:11,TSLA:15,BTC:20}};
const LONG_H = new Set(['10Y','5Y']);
const LABELS = {{SPY:'SPY · BETA',MAG7:'MAG7 · ALPHA',TSLA:'TESLA · BET',BTC:'BITCOIN · BET'}};
const NAMES  = {{SPY:'SPY (Beta)',MAG7:'Mag7 Alpha',TSLA:'Tesla',BTC:'Bitcoin'}};

let curTab='10Y', useLog=false, showBands=true, chartInst=null;

function setTab(h){{
  curTab=h;
  document.querySelectorAll('.tab').forEach(b=>b.classList.toggle('on',b.textContent.trim().startsWith(h)));
  render();
}}
function setScale(s){{
  useLog=s==='log';
  document.getElementById('bl').classList.toggle('on',!useLog);
  document.getElementById('bg').classList.toggle('on',useLog);
  render();
}}
function setBands(v){{
  showBands=v;
  document.getElementById('bbon').classList.toggle('on',v);
  document.getElementById('bboff').classList.toggle('on',!v);
  document.getElementById('bandLeg').style.opacity=v?1:0.3;
  render();
}}

function render(){{
  const hd = DATA.horizons[curTab];
  if(!hd) return;

  // Cards
  const keys = ['SPY','MAG7','TSLA','BTC'];
  document.getElementById('cards').innerHTML = keys.map(k=>{{
    const d = hd[k];
    const col = COLORS[k];
    if(!d) return `<div class="card" style="border-top-color:${{col}}"><div class="card-lbl">${{LABELS[k]}}</div><div class="card-val" style="color:var(--mut)">—</div></div>`;
    const vc = d.ret>=0?'pos':'neg', sign=d.ret>=0?'+':'';
    const cagr = d.cagr!=null?`${{d.cagr}}%`:'—';
    return `<div class="card" style="border-top-color:${{col}}">
      <div class="card-lbl">${{LABELS[k]}}</div>
      <div class="card-val ${{vc}}">${{sign}}${{d.ret.toFixed(1)}}%</div>
      <div class="card-sub">CAGR ${{cagr}} · Fwd ${{FWD[k]}}%</div>
      <div class="card-base">Base $${{d.base_price}} · ${{d.base_date}}</div>
    </div>`;
  }}).join('');

  // Outperf
  const spy = hd['SPY'];
  if(spy){{
    document.getElementById('outperf').innerHTML = ['MAG7','TSLA','BTC'].map(k=>{{
      const d=hd[k]; if(!d) return '';
      const a = d.ret - spy.ret, col=a>=0?'var(--grn)':'var(--red)', sign=a>=0?'+':'';
      const names={{MAG7:'Mag7',TSLA:'Tesla',BTC:'Bitcoin'}};
      return `<span class="op">${{names[k]}}: <span style="color:${{col}}">${{sign}}${{a.toFixed(1)}}pts vs SPY</span></span>`;
    }}).join('');
  }}

  // Note
  const method = LONG_H.has(curTab)?'CAGR geometric compounding':'Geometric Brownian Motion';
  document.getElementById('note').textContent = `Projection: ${{method}} · Dashed = forward · 90% vol bands on Tesla & Bitcoin`;

  // Chart
  if(chartInst){{chartInst.destroy();chartInst=null;}}
  const datasets=[];

  keys.forEach(k=>{{
    const d=hd[k]; if(!d) return;
    const col=COLORS[k], nm=NAMES[k];

    // Historical
    datasets.push({{
      label:nm,
      data:d.hist.map(([dt,v])=>{{return{{x:new Date(dt+'T12:00:00'),y:v}};}}),
      borderColor:col,backgroundColor:'transparent',borderWidth:2.5,pointRadius:0,tension:.12,borderDash:[]
    }});

    // Projection (stitched)
    if(d.proj&&d.proj.length){{
      const lastH = d.hist[d.hist.length-1];
      const stitch = [{{x:new Date(lastH[0]+'T12:00:00'),y:lastH[1]}}];
      datasets.push({{
        label:nm+' →',
        data:[...stitch,...d.proj.map(([dt,v])=>{{return{{x:new Date(dt+'T12:00:00'),y:v}};}})] ,
        borderColor:col,backgroundColor:'transparent',borderWidth:1.8,pointRadius:0,tension:.12,borderDash:[6,4]
      }});
    }}

    // Vol bands
    if(showBands && d.band_upper && d.band_lower){{
      const hex=col.replace('#','');
      const r=parseInt(hex.slice(0,2),16),g=parseInt(hex.slice(2,4),16),b=parseInt(hex.slice(4,6),16);
      const fc=`rgba(${{r}},${{g}},${{b}},0.08)`;
      const lastH=d.hist[d.hist.length-1];
      const stU=[{{x:new Date(lastH[0]+'T12:00:00'),y:lastH[1]}},...d.band_upper.map(([dt,v])=>{{return{{x:new Date(dt+'T12:00:00'),y:v}};}})] ;
      const stL=[{{x:new Date(lastH[0]+'T12:00:00'),y:lastH[1]}},...d.band_lower.map(([dt,v])=>{{return{{x:new Date(dt+'T12:00:00'),y:v}};}})] ;
      datasets.push({{label:`_${{k}}_u`,data:stU,borderColor:'transparent',backgroundColor:fc,fill:'+1',borderWidth:0,pointRadius:0,tension:.12}});
      datasets.push({{label:`_${{k}}_l`,data:stL,borderColor:'transparent',backgroundColor:fc,fill:false,borderWidth:0,pointRadius:0,tension:.12}});
    }}
  }});

  const todayX = new Date();
  const todayPlugin={{id:'tl',afterDraw(c){{
    const xs=c.scales.x;if(!xs)return;
    const xp=xs.getPixelForValue(todayX);if(!xp||xp<xs.left||xp>xs.right)return;
    c.ctx.save();c.ctx.strokeStyle='rgba(160,160,200,.25)';c.ctx.lineWidth=1;c.ctx.setLineDash([5,5]);
    c.ctx.beginPath();c.ctx.moveTo(xp,c.chartArea.top);c.ctx.lineTo(xp,c.chartArea.bottom);c.ctx.stroke();
    c.ctx.fillStyle='rgba(160,160,200,.4)';c.ctx.font='9px monospace';c.ctx.fillText('today',xp+5,c.chartArea.top+12);
    c.ctx.restore();
  }}}};

  chartInst=new Chart(document.getElementById('chart').getContext('2d'),{{
    type:'line',data:{{datasets}},
    options:{{
      responsive:true,maintainAspectRatio:false,animation:{{duration:300}},
      interaction:{{mode:'index',intersect:false}},
      plugins:{{
        legend:{{display:false}},
        tooltip:{{
          backgroundColor:'rgba(8,8,16,.97)',borderColor:'rgba(255,255,255,.07)',borderWidth:.5,
          titleColor:'#ddd',bodyColor:'#888',
          titleFont:{{size:10,family:'monospace'}},bodyFont:{{size:10,family:'monospace'}},
          callbacks:{{
            title(i){{const d=i[0]?.raw?.x;return d?d.toLocaleDateString('en',{{month:'short',day:'numeric',year:'numeric'}}):''}},
            label(i){{if((i.dataset.label||'').startsWith('_'))return null;return` ${{i.dataset.label}}: ${{i.raw.y?.toFixed(1)}}`}}
          }}
        }}
      }},
      scales:{{
        x:{{type:'time',time:{{tooltipFormat:'MMM d yyyy'}},ticks:{{color:'#2e2e48',font:{{size:9,family:'monospace'}},maxTicksLimit:9}},grid:{{color:'#0e0e1c'}}}},
        y:{{type:useLog?'logarithmic':'linear',ticks:{{color:'#2e2e48',font:{{size:9,family:'monospace'}},callback:v=>parseFloat(v.toFixed(0)),maxTicksLimit:9}},grid:{{color:'#0e0e1c'}},title:{{display:true,text:'Index · base = 100',color:'#2e2e48',font:{{size:9,family:'monospace'}}}}}}
      }}
    }},
    plugins:[todayPlugin]
  }});
}}

render();
</script>
</body>
</html>"""

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")
with open(out_path, "w") as f:
    f.write(html)

print(f"✓ Saved: {out_path}")
print(f"  Opening in browser...")
webbrowser.open(f"file://{out_path}")
