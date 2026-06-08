import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import torch
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
import time
import io
import base64

# ═══════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════

st.set_page_config(
    page_title="AnimeGAN Studio",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════
# THEME STATE
# ═══════════════════════════════════════════════
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "result_image" not in st.session_state:
    st.session_state.result_image = None
if "generation_done" not in st.session_state:
    st.session_state.generation_done = False

dark = st.session_state.dark_mode

# ═══════════════════════════════════════════════
# THEME VARIABLES
# ═══════════════════════════════════════════════
if dark:
    BG          = "#0d0d14"
    BG2         = "#13131f"
    CARD        = "rgba(255,255,255,0.04)"
    CARD_BORDER = "rgba(180,130,255,0.14)"
    CARD_HOVER  = "rgba(180,130,255,0.28)"
    TEXT        = "#ede9f8"
    TEXT_MUTED  = "rgba(237,233,248,0.45)"
    TEXT_FAINT  = "rgba(237,233,248,0.18)"
    ACCENT      = "#c084fc"
    ACCENT2     = "#38bdf8"
    ACCENT_RGB  = "192,132,252"
    ACCENT2_RGB = "56,189,248"
    LABEL_C     = "rgba(237,233,248,0.55)"
    INPUT_BG    = "rgba(255,255,255,0.05)"
    INPUT_BOR   = "rgba(192,132,252,0.2)"
    STEP_DONE   = "#38bdf8"
    BTN_GRAD    = "linear-gradient(135deg,#7c3aed,#a855f7)"
    BTN_HOV     = "linear-gradient(135deg,#9333ea,#c084fc)"
    BTN_SHADOW  = "rgba(168,85,247,0.4)"
    DL_BG       = "rgba(56,189,248,0.08)"
    DL_COLOR    = "#38bdf8"
    DL_BOR      = "rgba(56,189,248,0.3)"
    BADGE_BG    = "rgba(192,132,252,0.08)"
    BADGE_BOR   = "rgba(192,132,252,0.18)"
    BADGE_TEXT  = "#c084fc"
    HERO_GRAD   = "linear-gradient(135deg,#e0c3fc 0%,#c084fc 45%,#38bdf8 100%)"
    DIV_GRAD    = "linear-gradient(90deg,transparent,rgba(192,132,252,0.35),rgba(56,189,248,0.25),transparent)"
    SHADOW      = "0 4px 32px rgba(0,0,0,0.35), 0 1px 6px rgba(0,0,0,0.25)"
    SHADOW_H    = "0 8px 48px rgba(0,0,0,0.45)"
    EMPTY_BOR   = "rgba(192,132,252,0.14)"
    TAG_BG      = "rgba(192,132,252,0.1)"
    TAG_BOR     = "rgba(192,132,252,0.3)"
    TAG_TEXT    = "#c084fc"
else:
    BG          = "#faf7ff"
    BG2         = "#f3eeff"
    CARD        = "rgba(255,255,255,0.82)"
    CARD_BORDER = "rgba(139,92,246,0.13)"
    CARD_HOVER  = "rgba(139,92,246,0.25)"
    TEXT        = "#1e1b2e"
    TEXT_MUTED  = "rgba(30,27,46,0.5)"
    TEXT_FAINT  = "rgba(30,27,46,0.22)"
    ACCENT      = "#7c3aed"
    ACCENT2     = "#0ea5e9"
    ACCENT_RGB  = "124,58,237"
    ACCENT2_RGB = "14,165,233"
    LABEL_C     = "rgba(30,27,46,0.6)"
    INPUT_BG    = "rgba(255,255,255,0.9)"
    INPUT_BOR   = "rgba(124,58,237,0.22)"
    STEP_DONE   = "#0ea5e9"
    BTN_GRAD    = "linear-gradient(135deg,#6d28d9,#8b5cf6)"
    BTN_HOV     = "linear-gradient(135deg,#7c3aed,#a78bfa)"
    BTN_SHADOW  = "rgba(109,40,217,0.38)"
    DL_BG       = "rgba(14,165,233,0.07)"
    DL_COLOR    = "#0369a1"
    DL_BOR      = "rgba(14,165,233,0.3)"
    BADGE_BG    = "rgba(124,58,237,0.07)"
    BADGE_BOR   = "rgba(124,58,237,0.16)"
    BADGE_TEXT  = "#6d28d9"
    HERO_GRAD   = "linear-gradient(135deg,#4c1d95 0%,#7c3aed 45%,#0ea5e9 100%)"
    DIV_GRAD    = "linear-gradient(90deg,transparent,rgba(124,58,237,0.28),rgba(14,165,233,0.2),transparent)"
    SHADOW      = "0 4px 24px rgba(109,40,217,0.08), 0 1px 4px rgba(0,0,0,0.05)"
    SHADOW_H    = "0 8px 36px rgba(109,40,217,0.13)"
    EMPTY_BOR   = "rgba(124,58,237,0.14)"
    TAG_BG      = "rgba(124,58,237,0.07)"
    TAG_BOR     = "rgba(124,58,237,0.28)"
    TAG_TEXT    = "#7c3aed"

# ═══════════════════════════════════════════════
# SVG BACKGROUND  (anime-themed: floating sakura + stars)
# ═══════════════════════════════════════════════
if dark:
    SVG_BG = f"""
    <svg xmlns='http://www.w3.org/2000/svg' width='1400' height='900'>
      <defs>
        <radialGradient id='g1' cx='20%' cy='15%' r='55%'>
          <stop offset='0%' stop-color='#2d1457' stop-opacity='0.7'/>
          <stop offset='100%' stop-color='#0d0d14' stop-opacity='0'/>
        </radialGradient>
        <radialGradient id='g2' cx='80%' cy='80%' r='50%'>
          <stop offset='0%' stop-color='#0a2540' stop-opacity='0.6'/>
          <stop offset='100%' stop-color='#0d0d14' stop-opacity='0'/>
        </radialGradient>
      </defs>
      <rect width='1400' height='900' fill='{BG}'/>
      <rect width='1400' height='900' fill='url(#g1)'/>
      <rect width='1400' height='900' fill='url(#g2)'/>
      <!-- Stars -->
      {''.join([f"<circle cx='{(i*137)%1400}' cy='{(i*89)%900}' r='{0.8 if i%3==0 else 1.2}' fill='white' opacity='{0.15+0.2*(i%4)/4}'/>" for i in range(80)])}
      <!-- Floating orbs -->
      <circle cx='200' cy='150' r='180' fill='rgba(120,60,220,0.06)'/>
      <circle cx='1200' cy='700' r='220' fill='rgba(30,150,220,0.05)'/>
      <circle cx='700' cy='450' r='300' fill='rgba(180,100,255,0.03)'/>
      <!-- Sakura petals (simple) -->
      <ellipse cx='120' cy='80' rx='8' ry='5' fill='rgba(255,182,193,0.18)' transform='rotate(-30 120 80)'/>
      <ellipse cx='1300' cy='200' rx='6' ry='4' fill='rgba(255,182,193,0.15)' transform='rotate(20 1300 200)'/>
      <ellipse cx='950' cy='120' rx='7' ry='4' fill='rgba(255,182,193,0.12)' transform='rotate(45 950 120)'/>
      <ellipse cx='400' cy='750' rx='8' ry='5' fill='rgba(255,182,193,0.14)' transform='rotate(-15 400 750)'/>
      <ellipse cx='1100' cy='600' rx='6' ry='4' fill='rgba(255,182,193,0.1)' transform='rotate(60 1100 600)'/>
    </svg>"""
else:
    SVG_BG = f"""
    <svg xmlns='http://www.w3.org/2000/svg' width='1400' height='900'>
      <defs>
        <radialGradient id='g1' cx='15%' cy='10%' r='55%'>
          <stop offset='0%' stop-color='#ede9fe' stop-opacity='0.9'/>
          <stop offset='100%' stop-color='#faf7ff' stop-opacity='0'/>
        </radialGradient>
        <radialGradient id='g2' cx='85%' cy='85%' r='50%'>
          <stop offset='0%' stop-color='#e0f2fe' stop-opacity='0.7'/>
          <stop offset='100%' stop-color='#faf7ff' stop-opacity='0'/>
        </radialGradient>
        <radialGradient id='g3' cx='50%' cy='50%' r='40%'>
          <stop offset='0%' stop-color='#f3e8ff' stop-opacity='0.5'/>
          <stop offset='100%' stop-color='#faf7ff' stop-opacity='0'/>
        </radialGradient>
      </defs>
      <rect width='1400' height='900' fill='{BG}'/>
      <rect width='1400' height='900' fill='url(#g1)'/>
      <rect width='1400' height='900' fill='url(#g2)'/>
      <rect width='1400' height='900' fill='url(#g3)'/>
      <!-- Soft decorative circles -->
      <circle cx='100' cy='100' r='200' fill='rgba(167,139,250,0.07)'/>
      <circle cx='1320' cy='800' r='250' fill='rgba(125,211,252,0.07)'/>
      <circle cx='700' cy='500' r='350' fill='rgba(196,181,253,0.04)'/>
      <!-- Sakura petals -->
      <ellipse cx='80' cy='60' rx='10' ry='6' fill='rgba(244,114,182,0.2)' transform='rotate(-25 80 60)'/>
      <ellipse cx='180' cy='140' rx='8' ry='5' fill='rgba(244,114,182,0.15)' transform='rotate(15 180 140)'/>
      <ellipse cx='1350' cy='100' rx='10' ry='6' fill='rgba(244,114,182,0.18)' transform='rotate(30 1350 100)'/>
      <ellipse cx='1250' cy='220' rx='7' ry='4' fill='rgba(244,114,182,0.13)' transform='rotate(-40 1250 220)'/>
      <ellipse cx='300' cy='800' rx='9' ry='5' fill='rgba(244,114,182,0.16)' transform='rotate(20 300 800)'/>
      <ellipse cx='1100' cy='750' rx='8' ry='5' fill='rgba(244,114,182,0.12)' transform='rotate(-20 1100 750)'/>
      <ellipse cx='650' cy='50' rx='7' ry='4' fill='rgba(244,114,182,0.14)' transform='rotate(50 650 50)'/>
      <!-- Geometric accents -->
      <polygon points='1380,0 1400,0 1400,40' fill='rgba(139,92,246,0.08)'/>
      <polygon points='0,860 0,900 40,900' fill='rgba(14,165,233,0.06)'/>
    </svg>"""

SVG_B64 = base64.b64encode(SVG_BG.encode()).decode()

# ═══════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════
css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@700;900&family=Nunito:wght@300;400;500;600;700&display=swap');

*,*::before,*::after{{box-sizing:border-box;}}

html,body,[data-testid="stAppViewContainer"]{{
    background:transparent!important;
    color:{TEXT}!important;
    font-family:'Nunito',sans-serif;
}}
[data-testid="stAppViewContainer"]{{
    background-image:url("data:image/svg+xml;base64,{SVG_B64}")!important;
    background-size:cover!important;
    background-attachment:fixed!important;
    background-position:center!important;
}}
[data-testid="stHeader"]{{background:transparent!important;}}
[data-testid="stToolbar"]{{display:none;}}
footer{{visibility:hidden;}}
.block-container{{padding:0.75rem 1.5rem 4rem!important;max-width:960px!important;}}

/* NAVBAR */
.navbar{{
    display:flex;align-items:center;justify-content:space-between;
    padding:0.8rem 0 1.5rem;margin-bottom:0.5rem;
}}
.nav-logo{{
    font-family:'Cinzel',serif;font-size:1.3rem;font-weight:900;
    background:{HERO_GRAD};-webkit-background-clip:text;
    -webkit-text-fill-color:transparent;background-clip:text;
    letter-spacing:0.06em;
}}
.nav-links{{display:flex;gap:2rem;font-size:0.82rem;font-weight:600;letter-spacing:0.06em;color:{TEXT_MUTED};}}
.nav-tag{{
    font-size:0.65rem;font-weight:700;letter-spacing:0.2em;text-transform:uppercase;
    color:{TAG_TEXT};border:1.5px solid {TAG_BOR};padding:4px 14px;border-radius:20px;
    background:{TAG_BG};
}}

/* HERO */
.hero{{text-align:center;padding:2.5rem 1rem 1.5rem;}}
.hero h1{{
    font-family:'Cinzel',serif;
    font-size:clamp(2.8rem,6.5vw,5.8rem);
    font-weight:900;line-height:1.05;margin:0.5rem 0 1rem;
    background:{HERO_GRAD};
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
    letter-spacing:0.02em;
}}
.hero-sub{{
    font-size:1rem;color:{TEXT_MUTED};
    max-width:500px;margin:0 auto;line-height:1.8;font-weight:300;
}}
.hero-stats{{
    display:flex;justify-content:center;gap:2.5rem;margin-top:1.5rem;flex-wrap:wrap;
}}
.stat{{text-align:center;}}
.stat-num{{font-family:'Cinzel',serif;font-size:1.4rem;font-weight:900;color:{ACCENT};}}
.stat-lbl{{font-size:0.68rem;letter-spacing:0.12em;text-transform:uppercase;color:{TEXT_MUTED};margin-top:2px;}}

.divider{{height:1.5px;background:{DIV_GRAD};margin:1.8rem 0;}}

/* CARDS */
[data-testid="stVerticalBlockBorderWrapper"]{{
    background:{CARD}!important;
    border:1px solid {CARD_BORDER}!important;
    border-radius:18px!important;
    padding:1.45rem!important;
    backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);
    box-shadow:{SHADOW};
    transition:box-shadow 0.3s,border-color 0.3s;
}}
[data-testid="stVerticalBlockBorderWrapper"]:hover{{box-shadow:{SHADOW_H};border-color:{CARD_HOVER}!important;}}
.card-title{{
    font-size:0.65rem;font-weight:700;letter-spacing:0.22em;text-transform:uppercase;
    color:{ACCENT};margin-bottom:1rem;display:flex;align-items:center;gap:10px;
}}
.card-title::after{{content:'';flex:1;height:1px;background:{CARD_BORDER};}}

/* STYLE SELECTOR TILES */
.style-grid{{display:grid;grid-template-columns:1fr 1fr;gap:9px;margin:0.4rem 0 1.1rem;}}
.style-tile{{
    padding:14px 12px 10px;border-radius:14px;text-align:center;cursor:pointer;
    border:1.5px solid {CARD_BORDER};background:{INPUT_BG};
    transition:all 0.2s ease;
}}
.style-tile:hover{{border-color:{ACCENT};transform:translateY(-2px);box-shadow:0 6px 20px rgba({ACCENT_RGB},0.15);}}
.style-tile-icon{{font-size:1.6rem;margin-bottom:5px;}}
.style-tile-name{{font-size:0.8rem;font-weight:700;color:{TEXT};}}
.style-tile-tag{{font-size:0.65rem;color:{TEXT_MUTED};margin-top:2px;}}

/* UPLOAD */
[data-testid="stFileUploader"]{{
    border:2px dashed {CARD_BORDER}!important;
    border-radius:18px!important;background:{BADGE_BG}!important;
    padding:1.2rem!important;transition:all 0.3s!important;
}}
[data-testid="stFileUploader"]:hover{{
    border-color:{ACCENT}!important;background:rgba({ACCENT_RGB},0.06)!important;
}}
[data-testid="stFileUploader"] label{{display:none!important;}}

/* SLIDERS */
[data-testid="stSlider"] > div > div > div{{background:rgba({ACCENT_RGB},0.18)!important;}}
[data-testid="stSlider"] [aria-valuenow]{{
    background:{ACCENT}!important;border:2.5px solid {BG}!important;
    box-shadow:0 0 0 3px rgba({ACCENT_RGB},0.3),0 2px 10px rgba({ACCENT_RGB},0.35)!important;
}}

/* SELECTBOX */
[data-testid="stSelectbox"] > div > div{{
    background:{INPUT_BG}!important;border:1.5px solid {INPUT_BOR}!important;
    border-radius:12px!important;color:{TEXT}!important;
}}
[data-testid="stSelectbox"] > div > div:hover{{border-color:{ACCENT}!important;}}
[data-testid="stSelectbox"] svg{{fill:{ACCENT}!important;}}

/* LABELS */
label{{color:{LABEL_C}!important;font-size:0.78rem!important;font-weight:600!important;letter-spacing:0.04em!important;}}

/* GENERATE BUTTON */
.stButton > button{{
    width:100%;
    background:{BTN_GRAD}!important;
    color:white!important;border:none!important;
    border-radius:16px!important;padding:0.95rem 2rem!important;
    font-family:'Cinzel',serif!important;font-size:1.05rem!important;font-weight:700!important;
    letter-spacing:0.05em!important;cursor:pointer!important;transition:all 0.3s!important;
    box-shadow:0 6px 28px {BTN_SHADOW}!important;
}}
.stButton > button:hover{{
    background:{BTN_HOV}!important;
    box-shadow:0 10px 36px {BTN_SHADOW}!important;
    transform:translateY(-2px)!important;
}}
.stButton > button:active{{transform:translateY(0)!important;}}

/* IMAGES */
[data-testid="stImage"] img{{
    border-radius:16px!important;
    border:1px solid {CARD_BORDER}!important;
    box-shadow:{SHADOW}!important;
}}

/* DOWNLOAD BUTTON */
[data-testid="stDownloadButton"] > button{{
    width:100%!important;background:{DL_BG}!important;color:{DL_COLOR}!important;
    border:1.5px solid {DL_BOR}!important;border-radius:13px!important;
    padding:0.75rem 1.5rem!important;font-family:'Nunito',sans-serif!important;
    font-size:0.87rem!important;font-weight:700!important;transition:all 0.3s!important;
}}
[data-testid="stDownloadButton"] > button:hover{{
    background:rgba({ACCENT2_RGB},0.14)!important;
    box-shadow:0 4px 18px rgba({ACCENT2_RGB},0.22)!important;
}}

/* BADGES */
.badge-row{{display:flex;gap:8px;flex-wrap:wrap;margin:0.8rem 0;}}
.badge{{
    background:{BADGE_BG};border:1px solid {BADGE_BOR};
    border-radius:8px;padding:5px 12px;font-size:0.72rem;color:{TEXT_MUTED};
}}
.badge b{{color:{BADGE_TEXT};}}

/* STEPS */
.step-row{{
    display:flex;gap:8px;align-items:center;padding:9px 0;
    font-size:0.82rem;color:{TEXT_MUTED};
    border-bottom:1px solid {CARD_BORDER};
}}
.step-row:last-child{{border-bottom:none;}}
.step-dot{{width:8px;height:8px;border-radius:50%;background:rgba({ACCENT_RGB},0.22);flex-shrink:0;}}
.step-dot.done{{background:{STEP_DONE};box-shadow:0 0 10px rgba({ACCENT2_RGB},0.55);}}

/* EMPTY STATE */
.empty-state{{
    height:330px;display:flex;flex-direction:column;
    align-items:center;justify-content:center;
    border:2px dashed {EMPTY_BOR};border-radius:16px;gap:14px;
}}

/* FEATURE CHIPS */
.chip-row{{display:flex;gap:8px;flex-wrap:wrap;margin:0.6rem 0;}}
.chip{{
    background:{BADGE_BG};border:1px solid {BADGE_BOR};
    border-radius:30px;padding:5px 14px;font-size:0.73rem;
    color:{TEXT_MUTED};font-weight:600;
}}

/* COMPARISON LABEL */
.compare-label{{
    font-size:0.7rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;
    color:{TEXT_FAINT};text-align:center;padding:4px 0 8px;
}}

/* MISC */
[data-testid="stMarkdownContainer"] p{{color:{TEXT_MUTED};}}
[data-testid="stMarkdownContainer"] strong{{color:{TEXT};}}
.stAlert{{border-radius:14px!important;}}
[data-testid="stSpinner"]{{color:{ACCENT}!important;}}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# STYLES CONFIG  (no folder refs)
# ═══════════════════════════════════════════════
STYLES = {
    "🌿 Ghibli Dream":    {"prompt": "Studio Ghibli anime style, Hayao Miyazaki, lush painterly nature, warm golden light, soft watercolor, masterpiece", "tag": "Soft · Painterly"},
    "🌸 Dreamscape":      {"prompt": "surreal dreamlike anime art, Satoshi Kon inspired, vivid saturated colors, psychedelic illustration, cinematic", "tag": "Vivid · Surreal"},
    "⛅ Cinematic Sky":   {"prompt": "Makoto Shinkai ultra-detailed anime, dramatic volumetric sky, lens flare, hyper-realistic lighting, breathtaking scenery", "tag": "Epic · Cinematic"},
    "✏️ Classic Anime":  {"prompt": "classic 90s anime cel-shading style, clean bold lines, expressive character, vibrant retro animation, detailed", "tag": "Retro · Bold"},
    "🌙 Night Fantasy":   {"prompt": "dark fantasy anime night scene, moonlight, glowing particles, ethereal atmosphere, detailed illustration, magic", "tag": "Dark · Ethereal"},
    "🎨 Oil Paint":       {"prompt": "anime oil painting style, thick impasto brush strokes, rich deep colors, impressionist, artistic masterpiece", "tag": "Textured · Rich"},
}

# ═══════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model():
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "nitrosocke/Ghibli-Diffusion",
        torch_dtype=torch.float32
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    return pipe.to("cpu")

def prepare_init_image(img, size):
    img = ImageOps.exif_transpose(img).convert("RGB")
    resample = getattr(Image, "Resampling", Image).LANCZOS
    return ImageOps.fit(img, (size, size), method=resample, centering=(0.5, 0.5))

def polish_output(img):
    img = img.convert("RGB")
    img = ImageEnhance.Color(img).enhance(1.08)
    img = ImageEnhance.Contrast(img).enhance(1.06)
    img = ImageEnhance.Sharpness(img).enhance(1.18)
    return img.filter(ImageFilter.UnsharpMask(radius=1.1, percent=115, threshold=3))

# ═══════════════════════════════════════════════
# NAVBAR
# ═══════════════════════════════════════════════
nav_l, nav_r = st.columns([3, 1])
with nav_l:
    st.markdown(f"""
    <div class="navbar">
        <div class="nav-logo">⛩ AnimeGAN Studio</div>
        <div class="nav-tag">✦ Stable Diffusion · AI Art</div>
    </div>""", unsafe_allow_html=True)
with nav_r:
    toggle_label = "☀️ Light Mode" if dark else "🌙 Dark Mode"
    if st.button(toggle_label, key="theme_toggle"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# ═══════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
    <h1>Transform Photos<br>Into Anime Art</h1>
    <div class="hero-sub">
        Upload any photo and watch AI reimagine it in stunning anime styles —<br>
        from Ghibli watercolors to Shinkai's cinematic skies.
    </div>
    <div class="hero-stats">
        <div class="stat"><div class="stat-num">6</div><div class="stat-lbl">Art Styles</div></div>
        <div class="stat"><div class="stat-num">AI</div><div class="stat-lbl">Powered</div></div>
        <div class="stat"><div class="stat-num">768p</div><div class="stat-lbl">Max Output</div></div>
        <div class="stat"><div class="stat-num">PNG</div><div class="stat-lbl">Export</div></div>
    </div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# MAIN LAYOUT
# ═══════════════════════════════════════════════
left_col, right_col = st.columns([1, 1], gap="large")

# ──────────── LEFT PANEL ────────────
with left_col:

    # ── Upload Card ──
    with st.container(border=True):
        st.markdown('<div class="card-title">📸 Your Photo</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("drop", type=["jpg","jpeg","png"], label_visibility="collapsed")

        image = None
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            w, h = image.size
            st.markdown(f"""
            <div class="badge-row">
                <div class="badge">📐 <b>{w}×{h}</b></div>
                <div class="badge">🖼 <b>{uploaded_file.name[:18]}</b></div>
                <div class="badge">💾 <b>{uploaded_file.size//1024} KB</b></div>
            </div>""", unsafe_allow_html=True)
            st.image(image, use_container_width=True)
        else:
            st.markdown(f"""
            <div class="empty-state">
                <div style="font-size:3rem">🖼️</div>
                <div style="font-size:0.9rem;font-weight:600;color:{TEXT_MUTED}">Drop your photo here</div>
                <div style="font-size:0.75rem;color:{TEXT_FAINT}">JPG · PNG · up to any size</div>
                <div class="chip-row" style="justify-content:center">
                    <div class="chip">Portraits</div>
                    <div class="chip">Landscapes</div>
                    <div class="chip">Scenes</div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Style + Settings Card ──
    with st.container(border=True):
        st.markdown('<div class="card-title">🎭 Anime Style</div>', unsafe_allow_html=True)

        selected_style = st.selectbox("Style", list(STYLES.keys()), label_visibility="collapsed")
        cfg = STYLES[selected_style]

        # Style info pill
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba({ACCENT_RGB},0.07),rgba({ACCENT2_RGB},0.05));
            border:1px solid {CARD_BORDER};border-radius:12px;padding:11px 15px;
            margin:0.4rem 0 1.2rem;display:flex;align-items:center;gap:12px">
            <span style="font-size:1.8rem">{selected_style.split()[0]}</span>
            <div>
                <div style="font-weight:700;color:{TEXT};font-size:0.88rem">{selected_style[2:]}</div>
                <div style="font-size:0.7rem;color:{TEXT_MUTED};margin-top:3px">{cfg['tag']}</div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f'<div class="card-title">⚙️ Controls</div>', unsafe_allow_html=True)

        strength = st.slider("✨ Style Strength", 0.30, 1.0, 0.70, 0.05,
            help="Higher = stronger anime transformation. Lower = keeps more of the original photo.")
        guidance = st.slider("🎯 Prompt Guidance", 4.0, 15.0, 7.5, 0.5,
            help="How strictly the AI follows the style description.")
        steps = st.slider("🧪 Quality Steps", 20, 60, 35, 5,
            help="More steps usually gives cleaner detail, but takes longer on CPU.")

        col_a, col_b = st.columns(2)
        with col_a:
            output_size = st.selectbox("📐 Resolution", ["512×512","640×640","768×768"])
        with col_b:
            neg_prompt_on = st.checkbox("🚫 Quality filter", value=True, help="Removes blurry/low-quality outputs")
            polish_on = st.checkbox("✨ Final polish", value=True, help="Adds light sharpness, contrast, and color refinement")

        size_px = int(output_size.split("×")[0])
        quality_prompt = f"{cfg['prompt']}, masterpiece, best quality, ultra detailed, clean line art, refined lighting, sharp focus, high resolution"
        neg_prompt = "blurry, low quality, lowres, pixelated, jpeg artifacts, distorted, deformed, ugly, bad anatomy, bad face, extra fingers, missing fingers, watermark, text, signature, oversaturated, noisy" if neg_prompt_on else ""

        st.markdown("<br>", unsafe_allow_html=True)
        generate_btn = st.button("🌸  Generate Anime Art", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tips Card ──
    with st.container(border=True):
        st.markdown(f"""
        <div class="card-title">💡 Tips for Best Results</div>
        <div style="font-size:0.8rem;line-height:1.9;color:{TEXT_MUTED}">
            <b style="color:{TEXT}">📷 Portrait photos</b> — centered, well-lit faces work best<br>
            <b style="color:{TEXT}">🎚 Strength 0.6–0.75</b> — preserves likeness while applying style<br>
            <b style="color:{TEXT}">🎯 Guidance 7–9</b> — balanced style without over-saturation<br>
            <b style="color:{TEXT}">🧪 35+ steps</b> — cleaner details and smoother anime shading<br>
            <b style="color:{TEXT}">📐 512px</b> — fastest on CPU · 768px for high detail prints<br>
            <b style="color:{TEXT}">✅ Quality filter + polish</b> — keep enabled for cleaner output
        </div>""", unsafe_allow_html=True)

# ──────────── RIGHT PANEL ────────────
with right_col:

    # ── Result Card ──
    with st.container(border=True):
        st.markdown('<div class="card-title">✦ Anime Result</div>', unsafe_allow_html=True)

        result_ph  = st.empty()
        status_ph  = st.empty()

        if not uploaded_file and not st.session_state.result_image:
            result_ph.markdown(f"""
            <div class="empty-state">
                <div style="font-size:4rem">🌸</div>
                <div style="font-size:0.95rem;font-weight:700;color:{TEXT_MUTED}">Your anime artwork appears here</div>
                <div style="font-size:0.75rem;color:{TEXT_FAINT}">Upload a photo → pick a style → Generate</div>
                <div class="chip-row" style="justify-content:center">
                    <div class="chip">AI Powered</div><div class="chip">High Quality</div><div class="chip">Instant Download</div>
                </div>
            </div>""", unsafe_allow_html=True)
        elif st.session_state.result_image and not generate_btn:
            result_ph.image(st.session_state.result_image, use_container_width=True)

        if generate_btn:
            if not uploaded_file:
                st.warning("⚠️ Please upload a photo first!")
            else:
                st.session_state.generation_done = False
                with st.spinner(""):
                    status_ph.markdown(f"""
                    <div style="margin-top:0.8rem">
                        <div class="step-row"><div class="step-dot done"></div>📷 Photo validated ({w}×{h}px)</div>
                        <div class="step-row"><div class="step-dot done"></div>🎭 Style selected: {selected_style[2:]}</div>
                        <div class="step-row"><div class="step-dot"></div>🤖 Loading AI model…</div>
                        <div class="step-row"><div class="step-dot"></div>🌸 Running high-quality diffusion…</div>
                        <div class="step-row"><div class="step-dot"></div>✨ Polishing artwork…</div>
                    </div>""", unsafe_allow_html=True)

                    pipe = load_model()
                    init_image = prepare_init_image(image, size_px)

                    kwargs = dict(
                        prompt=quality_prompt,
                        image=init_image,
                        strength=strength,
                        guidance_scale=guidance,
                        num_inference_steps=steps,
                    )
                    if neg_prompt:
                        kwargs["negative_prompt"] = neg_prompt

                    result = pipe(**kwargs).images[0]
                    if polish_on:
                        result = polish_output(result)
                    st.session_state.result_image = result

                status_ph.markdown(f"""
                <div style="margin-top:0.8rem">
                    <div class="step-row"><div class="step-dot done"></div>📷 Photo validated ({w}×{h}px)</div>
                    <div class="step-row"><div class="step-dot done"></div>🎭 Style: {selected_style[2:]}</div>
                    <div class="step-row"><div class="step-dot done"></div>🤖 Model loaded</div>
                    <div class="step-row"><div class="step-dot done"></div>🌸 High-quality diffusion complete</div>
                    <div class="step-row"><div class="step-dot done"></div>✅ Artwork polished and ready!</div>
                </div>""", unsafe_allow_html=True)

                result_ph.image(result, use_container_width=True)
                st.session_state.generation_done = True

    # ── Download + Compare Card ──
    if st.session_state.result_image is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="card-title">⬇ Export & Compare</div>', unsafe_allow_html=True)

            buf = io.BytesIO()
            st.session_state.result_image.save(buf, format="PNG")
            ts  = int(time.time())

            dl_col, clr_col = st.columns([2,1])
            with dl_col:
                st.download_button(
                    label="⬇  Download Artwork (PNG)",
                    data=buf.getvalue(),
                    file_name=f"animegan_{selected_style.split()[1].lower()}_{ts}.png",
                    mime="image/png",
                )
            with clr_col:
                if st.button("🗑 Clear", use_container_width=True):
                    st.session_state.result_image = None
                    st.session_state.generation_done = False
                    st.rerun()

            # Side-by-side comparison if we have both
            if uploaded_file and image:
                st.markdown(f'<div class="card-title" style="margin-top:1rem">🔍 Side-by-Side</div>', unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f'<div class="compare-label">Original</div>', unsafe_allow_html=True)
                    st.image(image, use_container_width=True)
                with c2:
                    st.markdown(f'<div class="compare-label">{selected_style}</div>', unsafe_allow_html=True)
                    st.image(st.session_state.result_image, use_container_width=True)

# ═══════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════
st.markdown(f"""
<div class="divider" style="margin-top:2rem"></div>
<div style="text-align:center;padding:1rem 0 2rem;color:{TEXT_FAINT};font-size:0.7rem;letter-spacing:0.14em">
    ⛩ ANIMEGAN STUDIO &nbsp;·&nbsp; STABLE DIFFUSION &nbsp;·&nbsp; GHIBLI · SHINKAI · KON
</div>""", unsafe_allow_html=True)
