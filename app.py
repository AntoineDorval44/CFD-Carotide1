"""
app.py — Tableau de bord interactif CFD Carotide (Streamlit + Plotly 3D)

Artère CYLINDRIQUE 3D avec sténose réglable. La surface de l'artère est colorée
par la grandeur physique choisie (vitesse / pression / WSS), recalculée en direct
selon le diamètre de sténose. Comparaison permanente artère saine vs sténosée.

Lancement :
    streamlit run app.py

PFE Antoine Dorval — Arts et Métiers ParisTech, Biomécanique 2025-2026
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="CFD Carotide 3D", layout="wide")


def show_animation(fig, height):
    """
    Affiche une figure animée Plotly dans un cadre HTML isolé (iframe).
    Évite que les animations Plotly ne gèlent le reste de la page Streamlit
    (bug connu « insertBefore » entre Plotly et le moteur React de Streamlit).
    """
    html = fig.to_html(include_plotlyjs="cdn", full_html=False, auto_play=False)
    components.html(html, height=height, scrolling=False)

ARTERY_LENGTH_MM = 160.0
R0_MM = 4.0                 # rayon luminal sain de référence [mm]


# ---------------------------------------------------------------------------
# Géométrie : profil de rayon interne R(x) (cylindre + constriction gaussienne)
# ---------------------------------------------------------------------------
def radius_profile(x_m, R0_m, stenosis_frac, x0_m, length_m):
    sigma = max(length_m / 4.0, 1e-4)
    return R0_m * (1.0 - stenosis_frac * np.exp(-((x_m - x0_m) ** 2) / (2 * sigma ** 2)))


# ---------------------------------------------------------------------------
# Solveur analytique paramétré (recalcul instantané)
# ---------------------------------------------------------------------------
def solve(R0_mm, stenosis_diam_pct, x0_mm, plaque_len_mm,
          U_in, K, n, mu_min, mu_max, rho, nx=240):
    L = ARTERY_LENGTH_MM * 1e-3
    x = np.linspace(0, L, nx)
    stenosis_frac = stenosis_diam_pct / 100.0       # réduction de diamètre = réduction de rayon
    R_local = radius_profile(x, R0_mm * 1e-3, stenosis_frac, x0_mm * 1e-3, plaque_len_mm * 1e-3)

    R_inlet = R_local[0]
    Q = U_in * np.pi * R_inlet ** 2
    U_mean = Q / (np.pi * R_local ** 2)
    U_max = U_mean * (3 * n + 1) / (n + 1)

    gamma_wall = ((3 * n + 1) / (n * R_local)) * U_mean
    mu_app = np.clip(K * gamma_wall ** (n - 1), mu_min, mu_max)
    wss = mu_app * gamma_wall

    tau_wall = mu_app * gamma_wall
    dp_dx = 2 * tau_wall / R_local
    dx = np.gradient(x)
    p = np.cumsum((dp_dx * dx)[::-1])[::-1]
    p -= p[-1]

    return {
        "x_mm": x * 1000, "R_local_mm": R_local * 1000,
        "U_max_line": U_max, "U_mean_line": U_mean, "p_line": p, "wss_line": wss,
        "U_max": float(U_max.max()), "U_mean": float(U_mean.mean()),
        "delta_P": float(p.max() - p.min()),
        "WSS_mean": float(wss.mean()), "WSS_max": float(wss.max()),
        "R_min_mm": float(R_local.min() * 1000),
        "Re": float(rho * U_in * 2 * R_inlet / mu_max),
        "throat_velocity": float(U_max[np.argmin(R_local)]),
    }


# ---------------------------------------------------------------------------
# Champs / couleurs disponibles
# ---------------------------------------------------------------------------
CFD_COLORSCALE = [
    [0.0, "#000080"], [0.14, "#0000FF"], [0.28, "#00FFFF"], [0.42, "#00FF00"],
    [0.57, "#FFFF00"], [0.71, "#FF8000"], [0.85, "#FF0000"], [1.0, "#800000"],
]

FIELD_DEFS = {
    "Vitesse axiale":            dict(key="U_max_line", unit="m/s", scale=CFD_COLORSCALE),
    "Pression":                  dict(key="p_line",     unit="Pa",  scale="RdBu_r"),
    "Wall Shear Stress (WSS)":   dict(key="wss_line",   unit="Pa",  scale="Turbo"),
}


# ---------------------------------------------------------------------------
# Classification clinique du risque (seuils issus de la littérature)
# ---------------------------------------------------------------------------
def stenosis_risk(pct_diam):
    """Message court de risque selon le degré de sténose (critères NASCET)."""
    if pct_diam < 50:
        return st.success, "**Sténose légère** — risque faible. Suivi clinique simple."
    elif pct_diam < 70:
        return st.warning, ("**Sténose modérée** — risque intermédiaire. "
                            "Surveillance et traitement médical.")
    else:
        return st.error, ("**Sténose sévère** — risque d'AVC élevé. "
                          "Indication chirurgicale (seuil 70 %).")


def risk_gauge(pct):
    """Jauge visuelle de sévérité (vert / orange / rouge)."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": " %", "font": {"size": 36}},
        gauge={
            "axis": {"range": [0, 90], "tickwidth": 1},
            "bar": {"color": "rgba(0,0,0,0.75)", "thickness": 0.3},
            "steps": [
                {"range": [0, 50],  "color": "#A5D6A7"},   # vert  : léger
                {"range": [50, 70], "color": "#FFCC80"},   # orange: modéré
                {"range": [70, 90], "color": "#EF9A9A"},   # rouge : sévère
            ],
            "threshold": {"line": {"color": "#B71C1C", "width": 4},
                          "thickness": 0.85, "value": 70},
        },
    ))
    fig.update_layout(height=230, margin=dict(l=25, r=25, t=15, b=5))
    return fig


# ---------------------------------------------------------------------------
# Forme d'onde cardiaque (carotide commune) — débit pulsatile
# ---------------------------------------------------------------------------
def carotid_waveform(tau):
    """
    Forme d'onde de vitesse normalisée (moyenne = 1) sur un cycle cardiaque.
    `tau` : phase normalisée dans [0, 1[ (0 = début systole).

    Synthèse physiologique de la carotide commune :
      - montée systolique rapide, pic vers tau≈0.13
      - onde dicrote secondaire vers tau≈0.32
      - plateau diastolique positif (carotide = lit à basse résistance)
    """
    tau = np.asarray(tau) % 1.0
    systole   = 1.00 * np.exp(-((tau - 0.13) / 0.055) ** 2)   # pic systolique
    dicrote   = 0.28 * np.exp(-((tau - 0.32) / 0.060) ** 2)   # rebond dicrote
    diastole  = 0.22                                          # flux diastolique résiduel
    w = systole + dicrote + diastole
    # Normaliser pour que la MOYENNE sur le cycle vaille 1
    grid = np.linspace(0, 1, 1000, endpoint=False)
    mean = (np.exp(-((grid - 0.13) / 0.055) ** 2)
            + 0.28 * np.exp(-((grid - 0.32) / 0.060) ** 2) + 0.22).mean()
    return w / mean


# ---------------------------------------------------------------------------
# Rendu 3D — artère cylindrique fermée, surface colorée par la grandeur choisie
# ---------------------------------------------------------------------------
def artery_3d_figure(res, field_key, cmin, cmax, colorscale, unit, title, n_theta=60):
    """Cylindre 3D fermé de rayon R(x), surface colorée par res[field_key]."""
    x = res["x_mm"]
    R = res["R_local_mm"]
    C_line = res[field_key]

    theta = np.linspace(0, 2 * np.pi, n_theta)
    Xg = np.tile(x, (n_theta, 1))
    Th = np.tile(theta.reshape(-1, 1), (1, len(x)))
    Rg = np.tile(R, (n_theta, 1))
    Yg = Rg * np.cos(Th)
    Zg = Rg * np.sin(Th)
    Cg = np.tile(C_line, (n_theta, 1))

    fig = go.Figure(go.Surface(
        x=Xg, y=Yg, z=Zg, surfacecolor=Cg,
        colorscale=colorscale, cmin=cmin, cmax=cmax,
        colorbar=dict(title=unit, len=0.7),
        lighting=dict(ambient=0.65, diffuse=0.7, specular=0.15, roughness=0.5),
        hovertemplate="x=%{x:.0f} mm<br>" + unit + "=%{surfacecolor:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=14)),
        height=430, margin=dict(l=0, r=0, t=35, b=0),
        scene=dict(
            xaxis_title="x [mm]", yaxis_title="", zaxis_title="",
            yaxis=dict(showticklabels=False), zaxis=dict(showticklabels=False),
            aspectmode="manual", aspectratio=dict(x=4, y=1, z=1),
            camera=dict(eye=dict(x=2.0, y=1.5, z=1.1)),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


def internal_velocity_cut(res, vmax, n, title, nr=110):
    """Coupe longitudinale 2D : écoulement interne (profil parabolique selon r)."""
    x = res["x_mm"]; R = res["R_local_mm"]; U = res["U_max_line"]
    r = np.linspace(-R.max(), R.max(), nr)
    exponent = (n + 1) / n
    U2d = np.full((nr, len(x)), np.nan)
    for j in range(len(x)):
        m = np.abs(r) <= R[j]
        U2d[m, j] = U[j] * (1.0 - (np.abs(r[m]) / R[j]) ** exponent)
    fig = go.Figure(go.Heatmap(z=U2d, x=x, y=r, colorscale=CFD_COLORSCALE,
                               zmin=0, zmax=vmax, colorbar=dict(title="V [m/s]")))
    fig.add_trace(go.Scatter(x=x, y=R, mode="lines", line=dict(color="black", width=2), showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=-R, mode="lines", line=dict(color="black", width=2), showlegend=False))
    fig.update_layout(title=dict(text=title, x=0.5, font=dict(size=13)),
                      height=300, margin=dict(l=10, r=10, t=35, b=10),
                      xaxis_title="x [mm]", yaxis_title="r [mm]", yaxis=dict(range=[-7, 7]))
    return fig


# ---------------------------------------------------------------------------
# Animation pulsatile : battements cardiaques qui s'enchaînent
# ---------------------------------------------------------------------------
def pulsatile_animation(stenosis, x0, plaque_len, U_in, K, n, mu_min, mu_max, rho, hr,
                        N=18, n_beats=3):
    """
    Figure animée (bouton ▶) : onde cardiaque + profils vitesse/pression/WSS
    qui pulsent au rythme cardiaque, sur plusieurs battements enchaînés.
    """
    period = 60.0 / hr

    # Pré-calcul d'UN cycle (N instants) — réutilisé pour chaque battement
    cycle = []
    for i in range(N):
        Ui = U_in * float(carotid_waveform(i / N))
        rh = solve(R0_MM, 0,        x0, plaque_len, Ui, K, n, mu_min, mu_max, rho)
        rs = solve(R0_MM, stenosis, x0, plaque_len, Ui, K, n, mu_min, mu_max, rho)
        cycle.append((Ui, rh, rs))

    # Onde de fond (plusieurs battements)
    t_line = np.linspace(0, n_beats * period, n_beats * N + 1)
    wave_line = U_in * carotid_waveform(t_line / period)
    x_mm = cycle[0][1]["x_mm"]

    # Bornes Y fixes = pic systolique (sinon les courbes seraient coupées en animation)
    U_pk = max(c[2]["U_max"] for c in cycle)
    P_pk = max(c[2]["delta_P"] for c in cycle)
    W_pk = max(c[2]["WSS_max"] for c in cycle)

    BLUE, RED = "#2196F3", "#F44336"
    fig = make_subplots(
        rows=2, cols=3,
        specs=[[{"colspan": 3}, None, None], [{}, {}, {}]],
        row_heights=[0.32, 0.68],
        subplot_titles=("Onde de vitesse cardiaque (entrée)",
                        "Vitesse [m/s]", "Pression [Pa]", "WSS [Pa]"),
        vertical_spacing=0.16,
    )

    Ui0, rh0, rs0 = cycle[0]
    # 0: onde de fond (statique)
    fig.add_trace(go.Scatter(x=t_line, y=wave_line, mode="lines",
                             line=dict(color="#C62828", width=2), showlegend=False), row=1, col=1)
    # 1: marqueur mobile
    fig.add_trace(go.Scatter(x=[0], y=[Ui0], mode="markers",
                             marker=dict(color="black", size=12), showlegend=False), row=1, col=1)
    # 2-3: vitesse, 4-5: pression, 6-7: WSS
    fig.add_trace(go.Scatter(x=x_mm, y=rh0["U_max_line"], line=dict(color=BLUE), name="Saine"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_mm, y=rs0["U_max_line"], line=dict(color=RED), name="Sténosée"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_mm, y=rh0["p_line"], line=dict(color=BLUE), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=x_mm, y=rs0["p_line"], line=dict(color=RED), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=x_mm, y=rh0["wss_line"], line=dict(color=BLUE), showlegend=False), row=2, col=3)
    fig.add_trace(go.Scatter(x=x_mm, y=rs0["wss_line"], line=dict(color=RED), showlegend=False), row=2, col=3)

    # Frames : n_beats battements enchaînés
    frames = []
    for b in range(n_beats):
        for i in range(N):
            Ui, rh, rs = cycle[i]
            t = (b + i / N) * period
            frames.append(go.Frame(name=f"{b}-{i}", traces=[1, 2, 3, 4, 5, 6, 7], data=[
                go.Scatter(x=[t], y=[Ui], mode="markers", marker=dict(color="black", size=12)),
                go.Scatter(x=x_mm, y=rh["U_max_line"], line=dict(color=BLUE)),
                go.Scatter(x=x_mm, y=rs["U_max_line"], line=dict(color=RED)),
                go.Scatter(x=x_mm, y=rh["p_line"], line=dict(color=BLUE)),
                go.Scatter(x=x_mm, y=rs["p_line"], line=dict(color=RED)),
                go.Scatter(x=x_mm, y=rh["wss_line"], line=dict(color=BLUE)),
                go.Scatter(x=x_mm, y=rs["wss_line"], line=dict(color=RED)),
            ]))
    fig.frames = frames

    fig.update_xaxes(title_text="temps [s]", row=1, col=1)
    for c in (1, 2, 3):
        fig.update_xaxes(title_text="x [mm]", row=2, col=c)
    fig.update_yaxes(range=[0, U_pk * 1.1], row=2, col=1)
    fig.update_yaxes(range=[0, P_pk * 1.1 + 1e-6], row=2, col=2)
    fig.update_yaxes(range=[0, W_pk * 1.1], row=2, col=3)

    fig.update_layout(
        height=560, margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.0, y=1.22, xanchor="left",
            buttons=[
                dict(label="Lancer le battement", method="animate",
                     args=[None, {"frame": {"duration": int(period * 1000 / N), "redraw": True},
                                  "fromcurrent": True, "transition": {"duration": 0},
                                  "mode": "immediate"}]),
                dict(label="Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate"}]),
            ],
        )],
    )
    return fig


# ---------------------------------------------------------------------------
# Animation des particules de sang (écoulement dans la coupe longitudinale)
# ---------------------------------------------------------------------------
def particle_flow_animation(stenosis, x0, plaque_len, U_in, K, n, mu_min, mu_max, rho,
                            pulsatile, hr, n_part=420, n_frames=80, dt=0.018):
    """
    Animation de l'écoulement : des particules suivent les lignes de courant
    (streamtube), convergent et accélèrent au goulot. En mode pulsatile, leur
    vitesse globale suit le battement cardiaque, et une onde de pulsation défile
    en temps réel au-dessus, synchronisée avec les particules.
    """
    L = ARTERY_LENGTH_MM * 1e-3
    res = solve(R0_MM, stenosis, x0, plaque_len, U_in, K, n, mu_min, mu_max, rho)
    xg = res["x_mm"] * 1e-3
    Rg = res["R_local_mm"] * 1e-3
    Umax = res["U_max_line"]
    exponent = (n + 1) / n
    cmax = float(Umax.max())
    period = 60.0 / hr

    def R_of(x):    return np.interp(x, xg, Rg)
    def Umax_of(x): return np.interp(x, xg, Umax)

    rng = np.random.default_rng(0)
    s  = rng.uniform(-0.94, 0.94, n_part)   # ligne de courant (fraction du rayon)
    xp = rng.uniform(0, L, n_part)

    x_line = res["x_mm"]
    R_line_mm = res["R_local_mm"]

    # Pré-calcul des positions de particules + valeur d'onde, frame par frame
    part_x, part_y, part_c, wave_t, wave_v = [], [], [], [], []
    for f in range(n_frames):
        t = f * dt
        w = float(carotid_waveform(t / period)) if pulsatile else 1.0
        speed = Umax_of(xp) * (1.0 - np.abs(s) ** exponent) * w
        xp = xp + speed * dt
        out = xp > L
        if out.any():
            xp[out] = xp[out] - L
            s[out] = rng.uniform(-0.94, 0.94, int(out.sum()))
        part_x.append(xp.copy() * 1000)
        part_y.append(s * R_of(xp) * 1000)
        part_c.append(Umax_of(xp) * (1.0 - np.abs(s) ** exponent) * w)
        wave_t.append(t)
        wave_v.append(U_in * w)

    def particle_trace(i):
        return go.Scatter(x=part_x[i], y=part_y[i], mode="markers",
                          marker=dict(size=5, color=part_c[i], colorscale=CFD_COLORSCALE,
                                      cmin=0, cmax=cmax, colorbar=dict(title="V [m/s]")))

    play_pause = [dict(
        type="buttons", showactive=False, x=0.0, y=1.18, xanchor="left",
        buttons=[
            dict(label="Lancer l'écoulement", method="animate",
                 args=[None, {"frame": {"duration": 50, "redraw": True},
                              "fromcurrent": True, "transition": {"duration": 0},
                              "mode": "immediate"}]),
            dict(label="Pause", method="animate",
                 args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]),
        ],
    )]

    if pulsatile:
        # 2 lignes : onde temps réel (haut) + particules (bas)
        t_line = np.linspace(0, n_frames * dt, n_frames)
        wave_line = U_in * carotid_waveform(t_line / period)
        fig = make_subplots(rows=2, cols=1, row_heights=[0.3, 0.7], vertical_spacing=0.16,
                            subplot_titles=("Pulsation cardiaque (temps réel)",
                                            "Écoulement du sang"))
        # traces : 0 onde(static), 1 marqueur(dyn), 2 paroi haut, 3 paroi bas, 4 particules(dyn)
        fig.add_trace(go.Scatter(x=t_line, y=wave_line, mode="lines",
                                 line=dict(color="#C62828", width=2), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=[wave_t[0]], y=[wave_v[0]], mode="markers",
                                 marker=dict(color="black", size=11), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_line, y=R_line_mm, mode="lines",
                                 line=dict(color="black", width=2), showlegend=False, hoverinfo="skip"), row=2, col=1)
        fig.add_trace(go.Scatter(x=x_line, y=-R_line_mm, mode="lines",
                                 line=dict(color="black", width=2), showlegend=False, hoverinfo="skip"), row=2, col=1)
        fig.add_trace(particle_trace(0), row=2, col=1)
        fig.frames = [go.Frame(name=str(f), traces=[1, 4], data=[
            go.Scatter(x=[wave_t[f]], y=[wave_v[f]], mode="markers",
                       marker=dict(color="black", size=11)),
            particle_trace(f),
        ]) for f in range(n_frames)]
        fig.update_xaxes(title_text="temps [s]", row=1, col=1)
        fig.update_xaxes(title_text="x [mm]", range=[0, ARTERY_LENGTH_MM], row=2, col=1)
        fig.update_yaxes(title_text="U [m/s]", row=1, col=1)
        fig.update_yaxes(title_text="r [mm]", range=[-7, 7], row=2, col=1)
        fig.update_layout(height=480, margin=dict(l=10, r=10, t=50, b=10),
                          updatemenus=play_pause)
    else:
        fig = go.Figure(
            data=[
                go.Scatter(x=x_line, y=R_line_mm, mode="lines",
                           line=dict(color="black", width=2), showlegend=False, hoverinfo="skip"),
                go.Scatter(x=x_line, y=-R_line_mm, mode="lines",
                           line=dict(color="black", width=2), showlegend=False, hoverinfo="skip"),
                particle_trace(0),
            ],
            frames=[go.Frame(name=str(f), traces=[2], data=[particle_trace(f)])
                    for f in range(n_frames)],
        )
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10),
                          xaxis_title="x [mm]", yaxis_title="r [mm]",
                          yaxis=dict(range=[-7, 7]), xaxis=dict(range=[0, ARTERY_LENGTH_MM]),
                          updatemenus=play_pause)
    return fig


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
st.title("Simulation CFD 3D — Carotide avec sténose réglable")
st.caption("PFE Antoine Dorval · Arts et Métiers ParisTech · Biomécanique 2025-2026")

# --- Sélecteur de simulation (n'affecte pas la simulation du tube droit) ---
sim_mode = st.sidebar.radio(
    "Type de simulation",
    ["Artère droite", "Bifurcation carotidienne (ICA + ECA)"],
)
st.sidebar.divider()
if sim_mode.startswith("Bifurcation"):
    from bifurcation import render_bifurcation
    render_bifurcation()
    st.stop()

with st.sidebar:
    st.header("⚙️ Paramètres")

    st.subheader("Sténose")
    stenosis = st.slider("Réduction de diamètre [%]", 0, 90, 60, 1,
                         help="Sévérité de la plaque au goulot — LE paramètre principal")
    with st.expander("Forme de la plaque"):
        x0 = st.slider("Position du goulot [mm]", 20, 140, 80, 5)
        plaque_len = st.slider("Longueur de la plaque [mm]", 5, 60, 25, 5)

    st.subheader("Conditions aux limites")
    pulsatile = st.checkbox("Pulsation cardiaque (modèle réel)", value=False,
                            help="Décoché = débit constant idéal. Coché = le sang arrive "
                                 "par impulsions, comme un vrai cœur.")
    if pulsatile:
        U_in = st.slider("Vitesse MOYENNE sur le cycle [m/s]", 0.01, 0.60, 0.15, 0.01)
        hr = st.slider("Fréquence cardiaque [bpm]", 50, 100, 70, 1,
                       help="~70 bpm = adulte au repos")
    else:
        U_in = st.slider("Vitesse d'entrée [m/s]", 0.01, 0.60, 0.15, 0.01)
        hr = 70

    st.subheader("Rhéologie du sang (Power Law)")
    preset = st.selectbox("Modèle",
                          ["Power Law", "Newtonien", "Power Law fort", "Personnalisé"])
    K_def, n_def = {"Power Law": (0.01467, 0.7755),
                    "Newtonien": (0.0035, 1.0),
                    "Power Law fort": (0.020, 0.65),
                    "Personnalisé": (0.01467, 0.7755)}[preset]
    K = st.slider("K [Pa·sⁿ]", 0.001, 0.05, K_def, 0.001, format="%.4f")
    n = st.slider("n [-]", 0.4, 1.0, n_def, 0.005)

    with st.expander("Paramètres avancés"):
        rho = st.slider("ρ [kg/m³]", 1000.0, 1100.0, 1060.0, 5.0)
        mu_min = st.slider("μ min [Pa·s]", 0.0005, 0.005, 0.001, 0.0005, format="%.4f")
        mu_max = st.slider("μ max [Pa·s]", 0.05, 0.30, 0.160, 0.01)

# --- Vitesse de référence : pic systolique si pulsatile (vues statiques), sinon constante ---
if pulsatile:
    U_display = U_in * float(carotid_waveform(0.13))   # pic systolique
else:
    U_display = U_in

# --- Calculs : saine (0 %) et sténosée (curseur), au pic systolique si pulsatile ---
res_h = solve(R0_MM, 0,        x0, plaque_len, U_display, K, n, mu_min, mu_max, rho)
res_s = solve(R0_MM, stenosis, x0, plaque_len, U_display, K, n, mu_min, mu_max, rho)

# --- Jauge de risque + message court ---
rk1, rk2 = st.columns([1, 1.4])
with rk1:
    show_animation(risk_gauge(stenosis), 250)
with rk2:
    risk_fn, risk_msg = stenosis_risk(stenosis)
    st.write("")
    risk_fn(risk_msg)
    st.caption("Échelle de gravité clinique (réduction de diamètre). "
               "Détails et sources dans le volet « Risques cliniques » en bas de page.")

# --- Métriques ---
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Vitesse au goulot", f"{res_s['throat_velocity']:.2f} m/s",
          f"{res_s['throat_velocity']/res_h['U_max']:.1f}× vs saine",
          help="Vitesse maximale du sang au point le plus serré. Elle croît comme "
               "1/(1−réduction)² : doubler la sténose la multiplie beaucoup plus que par 2.")
c2.metric("Chute de pression", f"{res_s['delta_P']:.1f} Pa", f"{res_s['delta_P']/133.322:.2f} mmHg",
          help="Différence de pression entre l'entrée et la sortie de l'artère. Plus la "
               "sténose est serrée, plus le cœur doit pousser fort (chute de pression élevée).")
c3.metric("WSS max", f"{res_s['WSS_max']:.1f} Pa",
          f"{res_s['WSS_max']/res_h['WSS_max']:.0f}× vs saine",
          help="Wall Shear Stress : contrainte de frottement du sang sur la paroi. "
               "Maximale au goulot. Au-delà de ~7 Pa : risque de lésion de la paroi.")
c4.metric("Diamètre au goulot", f"{2*res_s['R_min_mm']:.1f} mm", f"sain {2*R0_MM:.0f} mm",
          help="Diamètre résiduel de la lumière au point le plus serré.")
c5.metric("Reynolds", f"{res_s['Re']:.0f}",
          help="Nombre de Reynolds : rapport forces d'inertie / forces visqueuses. "
               "Sous ~2000 l'écoulement est laminaire ; au-dessus il devient turbulent. "
               "Un goulot serré peut le faire grimper et créer des turbulences en aval.")

st.divider()

# --- Sélecteur de grandeur affichée sur la surface 3D ---
st.subheader("Artère 3D — surface colorée par la grandeur physique")
field_name = st.radio("Grandeur à visualiser sur la paroi de l'artère :",
                      list(FIELD_DEFS.keys()), horizontal=True)
fd = FIELD_DEFS[field_name]
key = fd["key"]

# Échelle de couleur commune (saine + sténosée) pour une comparaison juste
cmin = min(res_h[key].min(), res_s[key].min())
cmax = max(res_h[key].max(), res_s[key].max())
if cmin == cmax:
    cmax = cmin + 1e-6

g1, g2 = st.columns(2)
with g1:
    show_animation(artery_3d_figure(res_h, key, cmin, cmax, fd["scale"], fd["unit"],
                                    "Artère saine (0 %)"), 460)
with g2:
    show_animation(artery_3d_figure(res_s, key, cmin, cmax, fd["scale"], fd["unit"],
                                    f"Artère sténosée ({stenosis} %)"), 460)
_pulse_note = " (instant = pic systolique)" if pulsatile else ""
_npts = 60 * len(res_s["x_mm"])
st.caption(f"Surface = {field_name} [{fd['unit']}]{_pulse_note}. Même échelle de couleur "
           "pour les deux artères, comparaison directe. Clic-glisser pour pivoter, molette "
           "pour zoomer. Bouge le curseur « Réduction de diamètre » pour voir l'évolution. "
           f"Maillage surface : 60 × {len(res_s['x_mm'])} = {_npts} points.")

st.divider()

# --- Animation de l'écoulement (particules de sang) ---
st.subheader("Écoulement du sang dans l'artère sténosée")
show_animation(
    particle_flow_animation(stenosis, x0, plaque_len, U_in, K, n, mu_min, mu_max, rho,
                            pulsatile, hr),
    height=520 if pulsatile else 340)
_flow_note = ("Les particules surgissent au rythme cardiaque (systole / diastole). "
              if pulsatile else "Débit constant (modèle idéal). ")
st.caption("Clique sur « Lancer l'écoulement ». " + _flow_note +
           "Les particules suivent les lignes de courant : elles se resserrent et "
           "accélèrent (couleur chaude) en passant le goulot, exactement comme le sang réel.")

st.divider()

# --- Profils axiaux ---
if pulsatile:
    st.subheader("Profils axiaux pulsés")
    show_animation(
        pulsatile_animation(stenosis, x0, plaque_len, U_in, K, n, mu_min, mu_max, rho, hr),
        height=600)
    st.caption(f"Cœur à {hr} bpm. Clique sur « Lancer le battement » : le sang arrive par "
               "impulsions et les courbes pulsent en continu. Échelles calées sur le pic systolique.")
else:
    st.subheader("Profils axiaux — ligne centrale")
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=("Vitesse [m/s]", "Pression [Pa]", "WSS [Pa]"))
    for label, r, color in [("Saine", res_h, "#2196F3"), (f"Sténosée {stenosis}%", res_s, "#F44336")]:
        fig.add_trace(go.Scatter(x=r["x_mm"], y=r["U_max_line"], name=label,
                                 line=dict(color=color), legendgroup=label), row=1, col=1)
        fig.add_trace(go.Scatter(x=r["x_mm"], y=r["p_line"], name=label, line=dict(color=color),
                                 legendgroup=label, showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=r["x_mm"], y=r["wss_line"], name=label, line=dict(color=color),
                                 legendgroup=label, showlegend=False), row=1, col=3)
    fig.update_xaxes(title_text="x [mm]")
    # Échelle linéaire. On annote directement la valeur de l'artère saine (bleu) et du
    # goulot sténosé (rouge) pour que les valeurs faibles restent lisibles.
    ann = [
        (1, res_h["U_max"],  f"saine {res_h['U_max']:.2f} m/s", "#1565C0"),
        (1, res_s["U_max"],  f"goulot {res_s['U_max']:.2f} m/s", "#B71C1C"),
        (2, res_h["delta_P"], f"saine {res_h['delta_P']:.0f} Pa", "#1565C0"),
        (3, res_h["WSS_mean"], f"saine {res_h['WSS_mean']:.2f} Pa", "#1565C0"),
        (3, res_s["WSS_max"],  f"goulot {res_s['WSS_max']:.1f} Pa", "#B71C1C"),
    ]
    for col, yval, txt, color in ann:
        fig.add_annotation(x=8, y=yval, xref=f"x{col}", yref=f"y{col}", text=txt,
                           showarrow=False, xanchor="left", yshift=8,
                           font=dict(color=color, size=11))
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
    show_animation(fig, 380)
    st.caption("Échelle linéaire. La valeur de l'artère saine (bleu) est annotée directement sur "
               "chaque graphique : même quand la courbe paraît basse, sa valeur réelle est lisible. "
               "Bouge le diamètre de sténose pour voir la courbe rouge monter.")

# --- Tableau comparatif ---
st.subheader("Comparatif")
df = pd.DataFrame({
    "Paramètre": ["Vitesse max [m/s]", "ΔP [Pa]", "WSS moyen [Pa]", "WSS max [Pa]",
                  "Diamètre goulot [mm]"],
    "Saine":    [res_h["U_max"], res_h["delta_P"], res_h["WSS_mean"], res_h["WSS_max"], 2*res_h["R_min_mm"]],
    "Sténosée": [res_s["U_max"], res_s["delta_P"], res_s["WSS_mean"], res_s["WSS_max"], 2*res_s["R_min_mm"]],
})
df["Ratio"] = (df["Sténosée"] / df["Saine"]).map(lambda v: f"{v:.1f}×")
df["Saine"] = df["Saine"].map(lambda v: f"{v:.3f}")
df["Sténosée"] = df["Sténosée"].map(lambda v: f"{v:.3f}")
st.dataframe(df, use_container_width=True, hide_index=True)

# --- Coupe interne (bonus) ---
with st.expander("Voir l'écoulement interne (coupe longitudinale, profil de vitesse)"):
    vmax = max(res_h["U_max"], res_s["U_max"])
    e1, e2 = st.columns(2)
    with e1:
        show_animation(internal_velocity_cut(res_h, vmax, n, "Saine"), 320)
    with e2:
        show_animation(internal_velocity_cut(res_s, vmax, n, f"Sténosée ({stenosis}%)"), 320)
    st.caption("Ici on coupe l'artère en long pour voir le sang à l'intérieur : "
               "profil parabolique (max au centre, nul à la paroi = no-slip).")

st.divider()

# --- Vraie CFD ---
with st.expander("Vraie CFD (Lattice-Boltzmann) sur la sténose"):
    st.markdown("""
    Calcul hors-ligne qui résout Navier-Stokes sur un maillage (≠ modèle analytique du site).
    Il confirme un ratio de vitesse d'environ 3,0× au goulot et révèle la zone de
    recirculation en aval — invisible pour le modèle analytique.

    | Paramètre | Valeur |
    |---|---|
    | Méthode | Lattice-Boltzmann D2Q9, collision BGK |
    | Maillage | 260 × 80 = 20 800 cellules (11 873 fluides) |
    | Diamètre canal | 48 cellules |
    | Sténose | 60 % (profil cosinus, plaque 60 cellules) |
    | Vitesse d'entrée | 0,045 (unités réseau) |
    | Reynolds | 150 |
    | Viscosité ν | 0,0144 |
    | Relaxation τ | 0,543 |
    | Itérations | 14 000 (convergées) |
    | Parois / entrée / sortie | rebond / vitesse imposée / sortie libre |
    """)
    import os as _os
    _cfd = "results/figures/real_cfd_stenosis.png"
    if _os.path.exists(_cfd):
        st.image(_cfd, caption="CFD Lattice-Boltzmann sur une sténose 2D — "
                 "le reflux (bleu) et les tourbillons révèlent la recirculation en aval.")
    else:
        st.info("Lance `python run_real_cfd.py` pour générer la figure.")

# --- Explications : pression, risques cliniques, modèle (tout à la fin) ---
with st.expander("Pression (ΔP) — explication"):
    st.markdown("""
    Le site affiche la **chute de pression ΔP** (pression perdue par frottement),
    pas la pression sanguine absolue.

    | | Valeur | Quoi |
    |---|---|---|
    | Pression artérielle | 80–120 mmHg ≈ 10 600–16 000 Pa | pression absolue du sang |
    | ΔP affichée (artère saine) | ~60 Pa ≈ 0,45 mmHg | pression perdue sur le segment |

    Dans une grosse artère saine, ΔP est très faible (< 1 mmHg) : la résistance est dans les
    artérioles, pas dans la carotide. ΔP devient important (plusieurs mmHg) sur une sténose
    serrée — c'est le marqueur clinique. Vérification (Hagen-Poiseuille) :
    `ΔP = 8·μ·L·U / R² ≈ 60 Pa`.

    Comparaison avec la vraie CFD :

    | Grandeur | Modèle analytique | Vraie CFD |
    |---|---|---|
    | Vitesse au goulot, WSS, ΔP, débit | bien reproduits | oui |
    | Recirculation en aval | non capturée | capturée |
    | Turbulence, champ 3D | non | oui |
    """)

with st.expander("Risques cliniques d'une sténose sévère"):
    st.markdown("""
    Quand le diamètre se réduit fortement, plusieurs mécanismes s'enchaînent :

    1. **Rupture de la plaque d'athérome.** Au goulot, la vitesse et le WSS explosent. Un WSS
       très élevé sollicite la chape fibreuse qui recouvre la plaque. Si elle cède, le cœur
       lipidique est exposé au sang.

    2. **Thrombo-embolie et AVC.** La plaque rompue déclenche la coagulation. Des fragments
       se détachent, remontent vers le cerveau et bouchent une artère cérébrale (AIT ou AVC).

    3. **Recirculation en aval.** Juste après le rétrécissement, l'écoulement décolle et crée
       des tourbillons à WSS faible, qui favorisent la croissance de la plaque.

    4. **Chute de pression.** Une sténose serrée réduit l'irrigation en aval.

    Seuils de gravité (réduction de diamètre au goulot) :

    | Seuil | Valeur |
    |---|---|
    | Légère / modérée / sévère | < 50 % / 50–69 % / 70–99 % |
    | Risque AVC (sténose sévère symptomatique) | ~26 % à 2 ans sans chirurgie, ~9 % avec |
    | WSS normal / faible / élevé | 1–7 Pa / < 0,4 Pa / > 7 Pa |
    """)

with st.expander("Modèle de calcul"):
    st.markdown("""
    **Géométrie** : artère cylindrique de rayon `R(x)` avec constriction gaussienne ;
    le curseur règle la réduction de diamètre (0 % = saine).

    **Solveur analytique Power Law** (recalcul instantané) :
    - Vitesse : `u(r) = U_max·(1 − (r/R)^((n+1)/n))`, débit `Q = U_in·π·R²` conservé
    - WSS : `τ_w = μ_app·γ̇_w`, `γ̇_w = (3n+1)/(n·R)·U_mean`
    - Pression : intégration de `dP/dx = 2·τ_w/R`
    - Mode pulsatile : onde carotidienne `U_in(t) = U_moy · w(t)`, résolu en quasi-stationnaire.
    """)

with st.expander("Références"):
    st.markdown("""
    - Azahari et al. (2018) — modèle Power Law du sang.
    - Dhange et al. (2022) — chute de pression analytique.
    - Natarajan et al. (2020) — analyse spectrale de l'écoulement carotidien.
    - Ku, Giddens, Zarins & Glagov (1985) — écoulement pulsé et athérosclérose, *Arteriosclerosis* 5:293.
    - Gijsen, van de Vosse & Janssen (1999) — propriétés non-newtoniennes du sang, *J. Biomech.* 32:705.
    - Bharadvaj, Mabon & Giddens (1982) — géométrie de la bifurcation carotidienne, *J. Biomech.* 15:349.
    - Malek, Alper & Izumo (1999) — seuils de WSS, *JAMA* 282:2035.
    - NASCET (1991), *NEJM* 325:445 ; ECST (1998), *Lancet* — grades de sténose.
    """)
