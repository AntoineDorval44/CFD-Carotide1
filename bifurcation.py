"""
bifurcation.py — Simulation CFD d'une bifurcation carotidienne (CCA → ICA + ECA)
                 avec sténose sur l'ICA. Page Streamlit autonome.

Architecture identique au tube droit (app.py) :
  - Moteur analytique Power Law (quasi-1D le long de chaque branche)
  - Rendu Plotly, mêmes échelles de couleur
  - Contrôles via la barre latérale Streamlit

Géométrie : modèle de Bharadvaj et al. (1982), variante asymétrique
  CCA (tronc)   D = 8.00 mm,  L = 40 mm  (vertical, entrée en bas)
  ICA (interne) D = 6.17 mm,  L = 55 mm  (quasi-droite, ~15° à gauche, sténosée)
  ECA (externe) D = 4.50 mm,  L = 30 mm  (~35° à droite, saine)

Sténose ICA : plaque étendue démarrant à la bifurcation, profil cosinus (Young & Tsai 1973)
  r(s) = R_ICA · [1 − (δ/2)·(1 − cos(2π s / L_s))]   pour 0 < s < L_s

Réf. : Bharadvaj 1982, Ku 1985, Gijsen 1999, Zarins 1983, Young & Tsai 1973
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components
from skimage import measure
from scipy.spatial import cKDTree


def _show_animation(fig, height):
    """Affiche une figure animée dans un cadre HTML isolé (évite le gel de la page)."""
    html = fig.to_html(include_plotlyjs="cdn", full_html=False, auto_play=False)
    components.html(html, height=height, scrolling=False)


def carotid_waveform(tau):
    """Onde de vitesse carotidienne normalisée (moyenne = 1) sur un cycle."""
    tau = np.asarray(tau) % 1.0
    w = (np.exp(-((tau - 0.13) / 0.055) ** 2)
         + 0.28 * np.exp(-((tau - 0.32) / 0.060) ** 2) + 0.22)
    g = np.linspace(0, 1, 1000, endpoint=False)
    mean = (np.exp(-((g - 0.13) / 0.055) ** 2)
            + 0.28 * np.exp(-((g - 0.32) / 0.060) ** 2) + 0.22).mean()
    return w / mean

# --- Dimensions de référence (Bharadvaj 1982) ---
D_CCA, D_ICA, D_ECA = 8.0e-3, 6.17e-3, 4.5e-3          # diamètres [m]
R_CCA, R_ICA, R_ECA = D_CCA / 2, D_ICA / 2, D_ECA / 2
L_CCA, L_ICA, L_ECA = 40e-3, 55e-3, 30e-3              # longueurs [m]
# Pas de sinus/bulbe : ICA de diamètre constant (artère simple qui se divise).
R_SINUS = R_ICA                                        # -> aucun gonflement
W_SINUS = 12e-3

CFD_COLORSCALE = [
    [0.0, "#000080"], [0.14, "#0000FF"], [0.28, "#00FFFF"], [0.42, "#00FF00"],
    [0.57, "#FFFF00"], [0.71, "#FF8000"], [0.85, "#FF0000"], [1.0, "#800000"],
]


# ---------------------------------------------------------------------------
# Géométrie : profil de rayon de l'ICA avec sténose cosinus
# ---------------------------------------------------------------------------
def ica_radius(s, stenosis_frac, Ls):
    """
    Rayon de l'ICA à la distance s de la bifurcation [m].
    = rayon de base (sinus carotidien : large au bulbe, s'affine vers R_ICA)
      × facteur de sténose (plaque cosinus sur [0, Ls]).
    """
    s = np.asarray(s, dtype=float)
    r_base = R_ICA + (R_SINUS - R_ICA) * np.exp(-(s / W_SINUS) ** 2)   # bulbe → R_ICA
    r = r_base.copy()
    inside = (s >= 0) & (s <= Ls)
    phase = s[inside] / Ls
    r[inside] = r_base[inside] * (1.0 - (stenosis_frac / 2.0) * (1.0 - np.cos(2 * np.pi * phase)))
    return r


# ---------------------------------------------------------------------------
# Solveur analytique de la bifurcation
# ---------------------------------------------------------------------------
def solve_bifurcation(U_cca, stenosis_pct, Ls_mm, bif_angle_deg,
                      K, n, mu_min, mu_max, rho, ns=160):
    """
    Résout l'écoulement quasi-1D dans les trois branches.
    Répartition du débit ICA/ECA par résistance hydraulique (∝ ∫ ds / r^4).
    Retourne un dict complet (géométrie, champs par branche, métriques).
    """
    stenosis = stenosis_pct / 100.0
    Ls = Ls_mm * 1e-3

    # Angles : asymétrie réaliste (ICA quasi-droite, ECA plus déviée)
    ica_angle = np.radians(bif_angle_deg * 0.30)    # ex. 50° -> 15°
    eca_angle = np.radians(bif_angle_deg * 0.70)    # ex. 50° -> 35°

    # Abscisses curvilignes
    s_cca = np.linspace(0, L_CCA, ns)
    s_ica = np.linspace(0, L_ICA, ns)
    s_eca = np.linspace(0, L_ECA, ns)

    r_cca = np.full_like(s_cca, R_CCA)
    r_ica = ica_radius(s_ica, stenosis, Ls)
    r_eca = np.full_like(s_eca, R_ECA)

    # Débit CCA
    Q_cca = U_cca * np.pi * R_CCA ** 2

    # Résistances hydrauliques (géométrie) : I = ∫ ds / r^4
    I_ica = np.trapz(1.0 / r_ica ** 4, s_ica)
    I_eca = np.trapz(1.0 / r_eca ** 4, s_eca)
    # Répartition du débit (pressions de sortie égales) : Q ∝ 1/I
    g_ica, g_eca = 1.0 / I_ica, 1.0 / I_eca
    Q_ica = Q_cca * g_ica / (g_ica + g_eca)
    Q_eca = Q_cca - Q_ica

    def branch_fields(s, r, Q):
        U_mean = Q / (np.pi * r ** 2)
        U_max = U_mean * (3 * n + 1) / (n + 1)
        gamma_w = ((3 * n + 1) / (n * r)) * U_mean
        mu_app = np.clip(K * gamma_w ** (n - 1), mu_min, mu_max)
        wss = mu_app * gamma_w
        dp_dx = 2.0 * wss / r
        dP = np.trapz(dp_dx, s)            # chute de pression sur la branche
        return dict(U_mean=U_mean, U_max=U_max, wss=wss, dP=float(dP),
                    mu=mu_app, dp_dx=dp_dx)

    f_cca = branch_fields(s_cca, r_cca, Q_cca)
    f_ica = branch_fields(s_ica, r_ica, Q_ica)
    f_eca = branch_fields(s_eca, r_eca, Q_eca)

    # --- Profils de pression (pour le coloriage 3D) ---
    def cum_from_outlet(dp_dx, s):
        """p(s) = ∫_s^fin dp_dx ds'  (0 à la sortie, max à l'entrée de la branche)."""
        out = np.zeros_like(s)
        seg = 0.5 * (dp_dx[1:] + dp_dx[:-1]) * np.diff(s)
        out[:-1] = np.cumsum(seg[::-1])[::-1]
        return out

    p_ica = cum_from_outlet(f_ica["dp_dx"], s_ica)
    p_eca = cum_from_outlet(f_eca["dp_dx"], s_eca)
    P_apex = 0.5 * (p_ica[0] + p_eca[0])
    p_cca = P_apex + cum_from_outlet(f_cca["dp_dx"], s_cca)
    f_cca["p"], f_ica["p"], f_eca["p"] = p_cca, p_ica, p_eca

    # Vitesse max dans la sténose ICA
    sten_mask = (s_ica >= 0) & (s_ica <= Ls)
    v_throat = float(f_ica["U_max"][sten_mask].max())
    i_wssmax = int(np.argmax(f_ica["wss"]))

    metrics = {
        "U_cca_max":   float(f_cca["U_max"].max()),
        "v_throat":    v_throat,
        "ratio_v":     v_throat / float(f_cca["U_max"].max()),
        "dP_cca_ica":  f_cca["dP"] + f_ica["dP"],
        "dP_cca_eca":  f_cca["dP"] + f_eca["dP"],
        "wss_ica":     float(f_ica["wss"].mean()),
        "wss_eca":     float(f_eca["wss"].mean()),
        "wss_max":     float(f_ica["wss"].max()),
        "wss_max_loc": float(s_ica[i_wssmax] * 1000),
        "Re":          float(rho * U_cca * D_CCA / f_cca["mu"][0]),
        "Q_ica_frac":  Q_ica / Q_cca,
        "Q_eca_frac":  Q_eca / Q_cca,
        "stenosis_pct": stenosis_pct,
        "Ls_mm":       Ls_mm,
    }

    return dict(
        s_cca=s_cca, s_ica=s_ica, s_eca=s_eca,
        r_cca=r_cca, r_ica=r_ica, r_eca=r_eca,
        f_cca=f_cca, f_ica=f_ica, f_eca=f_eca,
        ica_angle=ica_angle, eca_angle=eca_angle, Ls=Ls,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Géométrie 2D des branches (centerlines + repères)
# ---------------------------------------------------------------------------
def _branch_axes(res):
    """Renvoie pour chaque branche : (start, direction, perp) en mm."""
    apex = np.array([0.0, L_CCA * 1000])           # point de bifurcation (mm)
    th_i, th_e = res["ica_angle"], res["eca_angle"]
    return {
        "cca": (np.array([0.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])),
        "ica": (apex, np.array([-np.sin(th_i), np.cos(th_i)]),
                np.array([np.cos(th_i), np.sin(th_i)])),
        "eca": (apex, np.array([np.sin(th_e), np.cos(th_e)]),
                np.array([np.cos(th_e), -np.sin(th_e)])),
    }


def _xy_to_branch_s(xy, res):
    """Assigne à des points (x,y mm) leur branche (0/1/2) et abscisse s (m)."""
    apex = np.array([0.0, L_CCA * 1000])
    di = np.array([-np.sin(res["ica_angle"]), np.cos(res["ica_angle"])])
    de = np.array([np.sin(res["eca_angle"]), np.cos(res["eca_angle"])])
    rel = xy - apex
    s_i = (rel @ di) / 1000.0
    s_e = (rel @ de) / 1000.0
    perp_i = np.abs(rel @ np.array([di[1], -di[0]]))
    perp_e = np.abs(rel @ np.array([de[1], -de[0]]))
    above = xy[:, 1] > L_CCA * 1000
    branch = np.zeros(len(xy), dtype=int)
    s = xy[:, 1] / 1000.0
    ui = above & (perp_i <= perp_e); ue = above & (perp_e < perp_i)
    branch[ui] = 1; s[ui] = s_i[ui]
    branch[ue] = 2; s[ue] = s_e[ue]
    return branch, np.clip(s, 0, None)


# ---------------------------------------------------------------------------
# Figure : champ de vitesse (Y continu, champ rempli)
# ---------------------------------------------------------------------------
def velocity_figure(res, geom, n, show_streamlines):
    exp = (n + 1) / n
    s2, b2, q2, inside = geom["s2"], geom["branch2"], geom["qfrac2"], geom["inside2"]
    U = np.full(s2.shape, np.nan)
    for b, name in [(0, "cca"), (1, "ica"), (2, "eca")]:
        mb = (b2 == b) & inside
        if mb.any():
            Umax = np.interp(s2[mb], res[f"s_{name}"], res[f"f_{name}"]["U_max"])
            U[mb] = Umax * (1.0 - q2[mb] ** exp)

    fig = go.Figure(go.Heatmap(
        x=geom["xs"], y=geom["ys"], z=U.T, colorscale=CFD_COLORSCALE,
        zmin=0, zmax=float(np.nanmax(U)), colorbar=dict(title="V [m/s]"),
        hoverongaps=False, zsmooth="best"))
    for poly in geom["walls"]:
        fig.add_trace(go.Scatter(x=poly[:, 0], y=poly[:, 1], mode="lines",
                                 line=dict(color="black", width=2),
                                 hoverinfo="skip", showlegend=False))
    if show_streamlines:
        ax = _branch_axes(res)
        sc, dc, pc = ax["cca"]
        # Lignes de courant CONTINUES : CCA -> ICA et CCA -> ECA (le flux se divise)
        for name in ("ica", "eca"):
            s_b, r_b = (res["s_ica"], res["r_ica"]) if name == "ica" else (res["s_eca"], res["r_eca"])
            sb, db, pb = ax[name]
            for qq in (-0.45, 0.0, 0.45):
                # tronçon CCA
                cx = sc[0] + res["s_cca"] * 1000 * dc[0] + qq * res["r_cca"] * 1000 * pc[0]
                cy = sc[1] + res["s_cca"] * 1000 * dc[1] + qq * res["r_cca"] * 1000 * pc[1]
                # tronçon branche
                bx = sb[0] + s_b * 1000 * db[0] + qq * r_b * 1000 * pb[0]
                by = sb[1] + s_b * 1000 * db[1] + qq * r_b * 1000 * pb[1]
                fig.add_trace(go.Scatter(
                    x=np.concatenate([cx, bx]), y=np.concatenate([cy, by]),
                    mode="lines", line=dict(color="rgba(255,255,255,0.45)", width=1),
                    hoverinfo="skip", showlegend=False))
    _annotate(fig, res, _branch_axes(res))
    _layout(fig, "Champ de vitesse — bifurcation")
    return fig


# ---------------------------------------------------------------------------
# Figure : carte du WSS sur la paroi (contour continu coloré)
# ---------------------------------------------------------------------------
def wss_figure(res, geom):
    wmax = max(res["f_cca"]["wss"].max(), res["f_ica"]["wss"].max(),
               res["f_eca"]["wss"].max())
    fig = go.Figure()
    first = True
    for poly in geom["walls"]:
        branch, s = _xy_to_branch_s(poly, res)
        wss = np.zeros(len(poly))
        for b, name in [(0, "cca"), (1, "ica"), (2, "eca")]:
            mb = branch == b
            if mb.any():
                wss[mb] = np.interp(s[mb], res[f"s_{name}"], res[f"f_{name}"]["wss"])
        fig.add_trace(go.Scatter(
            x=poly[:, 0], y=poly[:, 1], mode="markers",
            marker=dict(size=6, color=wss, colorscale="Turbo", cmin=0, cmax=wmax,
                        colorbar=dict(title="WSS [Pa]") if first else None, showscale=first),
            hovertemplate="WSS=%{marker.color:.2f} Pa<extra></extra>", showlegend=False))
        first = False
    _annotate(fig, res, _branch_axes(res), wss_mode=True)
    _layout(fig, "Wall Shear Stress sur la paroi")
    return fig


# ---------------------------------------------------------------------------
# Annotations (embolie, recirculation, étiquettes branches)
# ---------------------------------------------------------------------------
def _annotate(fig, res, ax, wss_mode=False):
    m = res["metrics"]
    # Étiquettes de branches
    apex = ax["ica"][0]
    fig.add_annotation(x=10, y=8, text="CCA (entrée)", showarrow=False,
                       font=dict(size=11, color="#444"))
    ica_end = apex + res["s_ica"][-1] * 1000 * ax["ica"][1]
    eca_end = apex + res["s_eca"][-1] * 1000 * ax["eca"][1]
    fig.add_annotation(x=ica_end[0], y=ica_end[1], text="ICA", showarrow=False,
                       font=dict(size=11, color="#444"), yshift=12)
    fig.add_annotation(x=eca_end[0], y=eca_end[1], text="ECA", showarrow=False,
                       font=dict(size=11, color="#444"), yshift=12)

    # Zone de recirculation (en aval de la plaque, si sténose marquée)
    if m["stenosis_pct"] >= 40:
        Ls = res["Ls"]
        s_rec = (Ls + 8e-3)
        if s_rec < L_ICA:
            start, d, p = ax["ica"]
            rc = start + s_rec * 1000 * d
            fig.add_annotation(x=rc[0], y=rc[1], text="zone de recirculation",
                               showarrow=True, arrowhead=2, ax=35, ay=0,
                               font=dict(size=10, color="#6A1B9A"),
                               arrowcolor="#6A1B9A")

    # Risque d'embolie cérébrale si sténose sévère
    if m["stenosis_pct"] >= 70:
        start, d, p = ax["ica"]
        tip = start + (L_ICA * 1000 + 6) * d
        fig.add_annotation(x=tip[0], y=tip[1], text="Risque d'embolie cérébrale",
                           showarrow=True, arrowhead=3, arrowwidth=2,
                           ax=0, ay=-40, font=dict(size=11, color="#B71C1C"),
                           arrowcolor="#B71C1C")


def _layout(fig, title):
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=13)),
        height=560, margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(title="x [mm]", scaleanchor="y", scaleratio=1,
                   range=[-45, 45], zeroline=False),
        yaxis=dict(title="y [mm]", range=[-5, 110], zeroline=False),
        plot_bgcolor="white",
    )


def ica_profiles_figure(res, res_h):
    """
    Profils vitesse / pression / WSS le long du trajet CCA -> ICA.
    Bleu = artère saine (0 %), Rouge = artère bouchée (sténose), valeurs annotées.
    """
    bif_mm = L_CCA * 1000

    def path(r, key):
        d = np.concatenate([r["s_cca"] * 1000, (L_CCA + r["s_ica"]) * 1000])
        y = np.concatenate([r["f_cca"][key], r["f_ica"][key]])
        return d, y

    BLUE, RED = "#2196F3", "#F44336"
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=("Vitesse [m/s]", "Pression [Pa]", "WSS [Pa]"))
    keys = [(1, "U_max", "m/s", "%.2f"), (2, "p", "Pa", "%.0f"), (3, "wss", "Pa", "%.2f")]
    for col, key, unit, fmt in keys:
        dh, yh = path(res_h, key)
        ds, ys = path(res, key)
        fig.add_trace(go.Scatter(x=dh, y=yh, line=dict(color=BLUE),
                                 name="Saine", legendgroup="s",
                                 showlegend=(col == 1)), row=1, col=col)
        fig.add_trace(go.Scatter(x=ds, y=ys, line=dict(color=RED),
                                 name="Bouchée", legendgroup="b",
                                 showlegend=(col == 1)), row=1, col=col)
        fig.add_vline(x=bif_mm, line_dash="dot", line_color="gray", row=1, col=col)
        # Annotations des valeurs (saine = bleu, bouchée au goulot = rouge)
        fig.add_annotation(x=8, y=yh.max(), xref=f"x{col}", yref=f"y{col}",
                           text="saine " + (fmt % yh.max()) + " " + unit,
                           showarrow=False, xanchor="left", yshift=8,
                           font=dict(color=BLUE, size=10))
        fig.add_annotation(x=8, y=ys.max(), xref=f"x{col}", yref=f"y{col}",
                           text="goulot " + (fmt % ys.max()) + " " + unit,
                           showarrow=False, xanchor="left", yshift=8,
                           font=dict(color=RED, size=10))
    fig.update_xaxes(title_text="distance depuis l'entrée CCA [mm]")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=1.18, x=0.5, xanchor="center"))
    return fig


# ---------------------------------------------------------------------------
# Vue 3D : bifurcation tubulaire, surface colorée par la grandeur choisie
# ---------------------------------------------------------------------------
FIELDS_3D = {
    "Vitesse axiale":          dict(key="U_max", unit="m/s", scale=CFD_COLORSCALE),
    "Pression":                dict(key="p",     unit="Pa",  scale="RdBu_r"),
    "Wall Shear Stress (WSS)": dict(key="wss",   unit="Pa",  scale="Turbo"),
}


def _smin(a, b, k):
    """Union lissée (smooth minimum quadratique) — crée un congé propre à la fourche."""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1 - h)


def _branch_sdf(P, pts, radii):
    """Distance signée (capsule polyligne) — vectorisée sur tous les segments."""
    A = pts[:-1]                      # (K,3)
    AB = pts[1:] - A                  # (K,3)
    L2 = np.maximum((AB * AB).sum(1), 1e-12)        # (K,)
    PA = P[:, None, :] - A[None, :, :]              # (M,K,3)
    t = np.clip((PA * AB[None]).sum(2) / L2[None], 0.0, 1.0)   # (M,K)
    proj = A[None] + t[..., None] * AB[None]        # (M,K,3)
    d = np.linalg.norm(P[:, None, :] - proj, axis=2)           # (M,K)
    r = radii[:-1][None] + t * (radii[1:] - radii[:-1])[None]  # (M,K)
    return (d - r).min(1)


@st.cache_data(show_spinner=False)
def build_geometry(stenosis_pct, Ls_mm, bif_angle, spacing=1.2, blend=3.5):
    """
    Géométrie UNIQUE et continue de la bifurcation (partagée 3D + 2D).

    Volume fluide = union LISSÉE (smin) de trois capsules (CCA, ICA, ECA) →
    fourche continue avec congé propre. Surface 3D par marching cubes, extrémités
    OUVERTES (entrée CCA alignée sur le bord de grille ; sorties ICA/ECA découpées
    au plan de sortie). Contour 2D par marching squares sur la coupe z=0.

    Mise en cache : recalcul seulement si la géométrie change.
    """
    stenosis = stenosis_pct / 100.0
    Ls = Ls_mm * 1e-3
    ica_a = np.radians(bif_angle * 0.30)
    eca_a = np.radians(bif_angle * 0.70)
    apex = np.array([0.0, L_CCA * 1000.0])
    di = np.array([-np.sin(ica_a), np.cos(ica_a)])
    de = np.array([np.sin(eca_a), np.cos(eca_a)])
    seg, ext = 2.0e-3, 4e-3

    # Centerlines (nœuds en mm, z=0) + rayons (mm) ; branches ICA/ECA prolongées de `ext`
    s_cca = np.arange(0, L_CCA + 1e-9, seg)
    pc = np.column_stack([np.zeros_like(s_cca), s_cca * 1000, np.zeros_like(s_cca)])
    rc = np.full_like(s_cca, R_CCA * 1000)

    s_ie = np.arange(0, L_ICA + ext, seg)
    ci = apex + np.outer(s_ie * 1000, di)
    pi = np.column_stack([ci[:, 0], ci[:, 1], np.zeros_like(s_ie)])
    ri = ica_radius(s_ie, stenosis, Ls) * 1000

    s_ee = np.arange(0, L_ECA + ext, seg)
    ce = apex + np.outer(s_ee * 1000, de)
    pe = np.column_stack([ce[:, 0], ce[:, 1], np.zeros_like(s_ee)])
    re = np.full_like(s_ee, R_ECA * 1000)

    allpts = np.vstack([pc, pi, pe])
    allr = np.concatenate([rc, ri, re])
    margin = allr.max() + blend + 2.0
    x0, x1 = allpts[:, 0].min() - margin, allpts[:, 0].max() + margin
    y0, y1 = 0.0, allpts[:, 1].max() + margin       # y0=0 -> entrée CCA ouverte
    z0, z1 = -margin, margin
    xs = np.arange(x0, x1, spacing)
    ys = np.arange(y0, y1, spacing)
    zs = np.arange(z0, z1, spacing)

    # --- Champ 3D et surface ---
    Xg, Yg, Zg = np.meshgrid(xs, ys, zs, indexing="ij")
    P = np.column_stack([Xg.ravel(), Yg.ravel(), Zg.ravel()])
    f = _smin(_smin(_branch_sdf(P, pc, rc), _branch_sdf(P, pi, ri), blend),
              _branch_sdf(P, pe, re), blend).reshape(Xg.shape)
    verts, faces, _, _ = measure.marching_cubes(f, level=0.0,
                                                spacing=(spacing, spacing, spacing))
    verts = verts + np.array([x0, y0, z0])

    # Attribution branche / abscisse par projection
    s_proj = np.zeros(len(verts))
    b_proj = np.zeros(len(verts), dtype=int)
    s_cc = verts[:, 1] / 1000.0
    s_ii = ((verts[:, :2] - apex) @ di) / 1000.0
    s_ee2 = ((verts[:, :2] - apex) @ de) / 1000.0
    # distance perpendiculaire à chaque axe pour choisir la branche
    perp_i = np.abs((verts[:, :2] - apex) @ np.array([di[1], -di[0]]))
    perp_e = np.abs((verts[:, :2] - apex) @ np.array([de[1], -de[0]]))
    is_branch = verts[:, 1] > L_CCA * 1000     # au-dessus de l'apex
    use_ica = is_branch & (perp_i <= perp_e)
    use_eca = is_branch & (perp_e < perp_i)
    b_proj[use_ica] = 1; s_proj[use_ica] = s_ii[use_ica]
    b_proj[use_eca] = 2; s_proj[use_eca] = s_ee2[use_eca]
    b_proj[~is_branch] = 0; s_proj[~is_branch] = s_cc[~is_branch]

    # Découpe des sorties ICA/ECA (extrémités ouvertes et planes)
    beyond = (((b_proj == 1) & (s_proj > L_ICA)) | ((b_proj == 2) & (s_proj > L_ECA)))
    keep = ~beyond[faces].any(axis=1)
    faces = faces[keep]

    vbranch = b_proj
    vs = np.clip(s_proj, 0, None)

    # --- Contour 2D (coupe z=0) ---
    X2, Y2 = np.meshgrid(xs, ys, indexing="ij")
    P2 = np.column_stack([X2.ravel(), Y2.ravel(), np.zeros(X2.size)])
    f2 = _smin(_smin(_branch_sdf(P2, pc, rc), _branch_sdf(P2, pi, ri), blend),
               _branch_sdf(P2, pe, re), blend).reshape(X2.shape)
    outline = []
    for c in measure.find_contours(f2, 0.0):
        ox = x0 + c[:, 0] * spacing
        oy = y0 + c[:, 1] * spacing
        outline.append(np.column_stack([ox, oy]))

    # --- Parois OUVERTES : on retire les segments de bouchon aux 3 ouvertures ---
    npi = np.array([di[1], -di[0]]); npe = np.array([de[1], -de[0]])
    walls = []
    tol = 1.3e-3
    for poly in outline:
        rel = poly - apex
        s_i = (rel @ di) / 1000.0
        s_e = (rel @ de) / 1000.0
        pi_ = np.abs(rel @ npi); pe_ = np.abs(rel @ npe)
        above = poly[:, 1] > L_CCA * 1000
        use_i = above & (pi_ <= pe_)
        sB = np.where(use_i, s_i, np.where(above, s_e, poly[:, 1] / 1000.0))
        cap = ((~above) & (poly[:, 1] < tol * 1000)) \
            | (above & use_i & (sB > L_ICA - tol)) \
            | (above & (~use_i) & (sB > L_ECA - tol))
        idx = np.where(~cap)[0]
        if len(idx) == 0:
            continue
        for run in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
            if len(run) > 2:
                walls.append(poly[run])

    # --- Champ 2D : assignation par cellule (branche, s, q, intérieur) ---
    sdf2 = f2
    # nearest-branch assignment in 2D
    s2c = Y2 / 1000.0
    rel = np.stack([X2 - apex[0], Y2 - apex[1]], axis=-1)
    s2i = (rel @ di) / 1000.0
    s2e = (rel @ de) / 1000.0
    perp2i = np.abs(rel @ np.array([di[1], -di[0]]))
    perp2e = np.abs(rel @ np.array([de[1], -de[0]]))
    branch2 = np.zeros(X2.shape, dtype=int)
    s2 = np.zeros(X2.shape)
    above = Y2 > L_CCA * 1000
    ui = above & (perp2i <= perp2e); ue = above & (perp2e < perp2i)
    branch2[ui] = 1; s2[ui] = s2i[ui]
    branch2[ue] = 2; s2[ue] = s2e[ue]
    branch2[~above] = 0; s2[~above] = s2c[~above]
    inside2 = sdf2 < 0
    # q (fraction radiale) = distance perpendiculaire / rayon local
    perp2 = np.where(branch2 == 0, np.abs(X2),
            np.where(branch2 == 1, perp2i, perp2e))
    # rayon local par branche
    rloc = np.where(branch2 == 0, R_CCA * 1000,
            np.where(branch2 == 1, np.interp(np.clip(s2, 0, L_ICA), s_ie, ri),
                     R_ECA * 1000))
    qfrac = np.clip(perp2 / np.maximum(rloc, 1e-6), 0, 1)

    return dict(verts=verts, faces=faces, vbranch=vbranch, vs=vs,
                outline=outline, walls=walls, xs=xs, ys=ys,
                branch2=branch2, s2=np.clip(s2, 0, None), qfrac2=qfrac, inside2=inside2)


def mesh_intensity(res, vbranch, vs, key):
    """Valeur de la grandeur `key` à chaque sommet (lookup le long de la branche)."""
    out = np.zeros(len(vs))
    for b, name in [(0, "cca"), (1, "ica"), (2, "eca")]:
        mb = vbranch == b
        if mb.any():
            out[mb] = np.interp(vs[mb], res[f"s_{name}"], res[f"f_{name}"][key])
    return out


def bifurcation_mesh_figure(verts, faces, intensity, cmin, cmax, colorscale, unit, title):
    """Surface continue en Y (Mesh3d), colorée par la grandeur physique."""
    fig = go.Figure(go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        intensity=intensity, colorscale=colorscale, cmin=cmin, cmax=cmax,
        colorbar=dict(title=unit, len=0.7), flatshading=False,
        lighting=dict(ambient=0.6, diffuse=0.7, specular=0.2),
        hovertemplate=f"{unit}=%{{intensity:.2f}}<extra></extra>"))
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=13)),
        height=480, margin=dict(l=0, r=0, t=35, b=0),
        scene=dict(xaxis_title="x [mm]", yaxis_title="y [mm]", zaxis_title="",
                   zaxis=dict(showticklabels=False), aspectmode="data",
                   camera=dict(eye=dict(x=1.6, y=-1.4, z=0.8))))
    return fig


# ---------------------------------------------------------------------------
# Animation des particules de sang dans la bifurcation
# ---------------------------------------------------------------------------
def particle_flow_bifurcation(res, geom, U_in, n, pulsatile, hr, n_part=420, n_frames=80, dt=0.02):
    """Particules circulant de la CCA vers ICA/ECA (split par débit), accélérées au goulot."""
    ax = _branch_axes(res)
    m = res["metrics"]
    exp = (n + 1) / n      # exposant du profil de vitesse Power Law

    # Données par branche : grilles s (m), r, U_max, et géométrie
    B = {}
    for bid, name, s, r, f, Lb in [
        (0, "cca", res["s_cca"], res["r_cca"], res["f_cca"], L_CCA),
        (1, "ica", res["s_ica"], res["r_ica"], res["f_ica"], L_ICA),
        (2, "eca", res["s_eca"], res["r_eca"], res["f_eca"], L_ECA)]:
        start, d, p = ax[name]
        B[bid] = dict(s=s, r=r, Umax=f["U_max"], L=Lb, start=start, d=d, p=p)

    cmax = max(res["f_cca"]["U_max"].max(), res["f_ica"]["U_max"].max(),
               res["f_eca"]["U_max"].max())
    period = 60.0 / hr
    q_ica = m["Q_ica_frac"]

    rng = np.random.default_rng(0)
    bid = rng.choice([0, 1, 2], n_part, p=[0.55, 0.30, 0.15])
    sp = np.array([rng.uniform(0, B[b]["L"]) for b in bid])
    qq = rng.uniform(-0.92, 0.92, n_part)

    px_f, py_f, col_f, wt, wv = [], [], [], [], []
    for fme in range(n_frames):
        t = fme * dt
        w = float(carotid_waveform(t / period)) if pulsatile else 1.0
        # vitesse locale
        speed = np.zeros(n_part)
        for b in (0, 1, 2):
            mb = bid == b
            if mb.any():
                Um = np.interp(sp[mb], B[b]["s"], B[b]["Umax"])
                speed[mb] = Um * (1.0 - np.abs(qq[mb]) ** exp) * w
        sp = sp + speed * dt
        # transitions
        for b in (0, 1, 2):
            done = (bid == b) & (sp > B[b]["L"])
            if done.any():
                if b == 0:   # CCA -> ICA ou ECA selon le débit
                    to_ica = rng.random(int(done.sum())) < q_ica
                    idx = np.where(done)[0]
                    bid[idx] = np.where(to_ica, 1, 2)
                    sp[idx] = 0.0
                else:        # ICA/ECA -> retour CCA
                    bid[done] = 0
                    sp[done] = 0.0
                qq[done] = rng.uniform(-0.92, 0.92, int(done.sum()))
        # positions (mm)
        X = np.zeros(n_part); Y = np.zeros(n_part); C = np.zeros(n_part)
        for b in (0, 1, 2):
            mb = bid == b
            if mb.any():
                s_mm = sp[mb] * 1000
                r_mm = np.interp(sp[mb], B[b]["s"], B[b]["r"]) * 1000
                start, d, p = B[b]["start"], B[b]["d"], B[b]["p"]
                X[mb] = start[0] + s_mm * d[0] + qq[mb] * r_mm * p[0]
                Y[mb] = start[1] + s_mm * d[1] + qq[mb] * r_mm * p[1]
                Um = np.interp(sp[mb], B[b]["s"], B[b]["Umax"])
                C[mb] = Um * (1.0 - np.abs(qq[mb]) ** exp) * w
        px_f.append(X.copy()); py_f.append(Y.copy()); col_f.append(C.copy())
        wt.append(t); wv.append(U_in * w)

    def part_trace(i):
        return go.Scatter(x=px_f[i], y=py_f[i], mode="markers",
                          marker=dict(size=4, color=col_f[i], colorscale=CFD_COLORSCALE,
                                      cmin=0, cmax=cmax, colorbar=dict(title="V [m/s]")))

    # Parois ouvertes (contour du Y sans bouchons, statique)
    wall_traces = [go.Scatter(x=poly[:, 0], y=poly[:, 1], mode="lines",
                              line=dict(color="black", width=2),
                              hoverinfo="skip", showlegend=False)
                   for poly in geom["walls"]]

    play_pause = [dict(type="buttons", showactive=False, x=0.0, y=1.12, xanchor="left",
        buttons=[
            dict(label="Lancer l'écoulement", method="animate",
                 args=[None, {"frame": {"duration": 50, "redraw": True},
                              "fromcurrent": True, "transition": {"duration": 0},
                              "mode": "immediate"}]),
            dict(label="Pause", method="animate",
                 args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])])]

    if pulsatile:
        t_line = np.linspace(0, n_frames * dt, n_frames)
        wave_line = U_in * carotid_waveform(t_line / period)
        fig = make_subplots(rows=1, cols=2, column_widths=[0.32, 0.68],
                            subplot_titles=("Pulsation (temps réel)", "Écoulement du sang"))
        fig.add_trace(go.Scatter(x=t_line, y=wave_line, mode="lines",
                                 line=dict(color="#C62828", width=2), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=[wt[0]], y=[wv[0]], mode="markers",
                                 marker=dict(color="black", size=10), showlegend=False), row=1, col=1)
        n_wall = len(wall_traces)
        for wt_tr in wall_traces:
            fig.add_trace(wt_tr, row=1, col=2)
        fig.add_trace(part_trace(0), row=1, col=2)
        part_idx = 2 + n_wall      # index du trace particules
        fig.frames = [go.Frame(name=str(i), traces=[1, part_idx], data=[
            go.Scatter(x=[wt[i]], y=[wv[i]], mode="markers", marker=dict(color="black", size=10)),
            part_trace(i)]) for i in range(n_frames)]
        fig.update_xaxes(title_text="temps [s]", row=1, col=1)
        fig.update_yaxes(title_text="U [m/s]", row=1, col=1)
        fig.update_xaxes(title_text="x [mm]", range=[-45, 45], row=1, col=2,
                         scaleanchor="y2", scaleratio=1)
        fig.update_yaxes(title_text="y [mm]", range=[-5, 110], row=1, col=2)
        fig.update_layout(height=560, margin=dict(l=10, r=10, t=50, b=10), updatemenus=play_pause)
    else:
        fig = go.Figure(data=wall_traces + [part_trace(0)])
        part_idx = len(wall_traces)
        fig.frames = [go.Frame(name=str(i), traces=[part_idx], data=[part_trace(i)])
                      for i in range(n_frames)]
        fig.update_layout(height=560, margin=dict(l=10, r=10, t=40, b=10),
                          xaxis=dict(title="x [mm]", range=[-45, 45], scaleanchor="y", scaleratio=1),
                          yaxis=dict(title="y [mm]", range=[-5, 110]),
                          plot_bgcolor="white", updatemenus=play_pause)
    return fig


# ---------------------------------------------------------------------------
# Page Streamlit
# ---------------------------------------------------------------------------
def render_bifurcation():
    st.subheader("Bifurcation carotidienne (CCA → ICA + ECA) avec sténose de l'ICA")
    st.caption("La plaque d'athérome se développe sur "
               "l'artère carotide interne (ICA), branche qui irrigue le cerveau.")

    with st.sidebar:
        st.subheader("Bifurcation — paramètres")
        stenosis = st.slider("Degré de sténose ICA [%]", 0, 90, 50, 1)
        Ls_mm = st.slider("Longueur de la plaque [mm]", 10, 30, 20, 1)
        bif_angle = st.slider("Angle de bifurcation [°]", 30, 70, 50, 1,
                              help="Réparti de façon asymétrique : ICA ~30 %, ECA ~70 %")

        st.subheader("Conditions aux limites")
        pulsatile = st.checkbox("Pulsation cardiaque (modèle réel)", value=False)
        if pulsatile:
            U_cca = st.slider("Vitesse MOYENNE CCA [m/s]", 0.01, 0.60, 0.15, 0.01)
            hr = st.slider("Fréquence cardiaque [bpm]", 50, 100, 70, 1)
        else:
            U_cca = st.slider("Vitesse d'entrée CCA [m/s]", 0.01, 0.60, 0.15, 0.01)
            hr = 70

        st.subheader("Rhéologie du sang")
        model = st.selectbox("Modèle", ["Power Law", "Newtonien"])
        if model == "Newtonien":
            K, n = 0.0035, 1.0
        else:
            K, n = 0.01467, 0.7755
        mu_min, mu_max, rho = 0.001, 0.160, 1060.0

        st.subheader("Affichage")
        show_streamlines = st.checkbox("Lignes de courant", value=True)
        show_wss = st.checkbox("Carte du WSS sur la paroi", value=True)

    # Vitesse de référence : pic systolique si pulsatile (vues statiques), sinon constante
    U_disp = U_cca * float(carotid_waveform(0.13)) if pulsatile else U_cca

    res = solve_bifurcation(U_disp, stenosis, Ls_mm, bif_angle, K, n, mu_min, mu_max, rho)
    res_h = solve_bifurcation(U_disp, 0, Ls_mm, bif_angle, K, n, mu_min, mu_max, rho)
    m = res["metrics"]

    # --- Métriques ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vitesse max sténose", f"{m['v_throat']:.2f} m/s",
              f"{m['ratio_v']:.1f}× vs CCA")
    c2.metric("ΔP CCA→ICA", f"{m['dP_cca_ica']:.1f} Pa",
              f"{m['dP_cca_ica']/133.322:.2f} mmHg")
    c3.metric("WSS max (ICA)", f"{m['wss_max']:.1f} Pa", f"à {m['wss_max_loc']:.0f} mm")
    c4.metric("Reynolds CCA", f"{m['Re']:.0f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Débit vers ICA", f"{m['Q_ica_frac']*100:.0f} %", "du débit CCA")
    c6.metric("Débit vers ECA", f"{m['Q_eca_frac']*100:.0f} %", "du débit CCA")
    c7.metric("WSS moyen ICA", f"{m['wss_ica']:.2f} Pa")
    c8.metric("WSS moyen ECA", f"{m['wss_eca']:.2f} Pa")

    if stenosis >= 70:
        st.error("Sténose sévère de l'ICA — risque d'embolie cérébrale (fragment de plaque "
                 "pouvant migrer vers le cerveau). Indication chirurgicale (seuil 70 %).")
    elif stenosis >= 50:
        st.warning("Sténose modérée de l'ICA — surveillance et traitement médical.")

    st.divider()

    # --- Vue 3D : surface CONTINUE en Y colorée par la grandeur choisie ---
    st.subheader("Bifurcation 3D — surface continue colorée par la grandeur physique")
    field_name = st.radio("Grandeur sur la paroi :", list(FIELDS_3D.keys()), horizontal=True,
                          key="bif_field")
    fd = FIELDS_3D[field_name]
    fk = fd["key"]
    cmin = min(res_h["f_cca"][fk].min(), res["f_ica"][fk].min(), res["f_eca"][fk].min(),
               res_h["f_ica"][fk].min())
    cmax = max(res["f_cca"][fk].max(), res["f_ica"][fk].max(), res["f_eca"][fk].max(),
               res_h["f_ica"][fk].max())
    if cmin == cmax:
        cmax = cmin + 1e-6

    # Géométries continues (mises en cache ; recalculées seulement si la géométrie change)
    geom_h = build_geometry(0, Ls_mm, bif_angle)
    geom_s = build_geometry(stenosis, Ls_mm, bif_angle)
    ih = mesh_intensity(res_h, geom_h["vbranch"], geom_h["vs"], fk)
    isn = mesh_intensity(res, geom_s["vbranch"], geom_s["vs"], fk)

    g3a, g3b = st.columns(2)
    with g3a:
        _show_animation(bifurcation_mesh_figure(geom_h["verts"], geom_h["faces"], ih,
                                               cmin, cmax, fd["scale"], fd["unit"],
                                               "Bifurcation saine (0 %)"), 500)
    with g3b:
        _show_animation(bifurcation_mesh_figure(geom_s["verts"], geom_s["faces"], isn,
                                               cmin, cmax, fd["scale"], fd["unit"],
                                               f"ICA sténosée ({stenosis} %)"), 500)
    _note = " (instant = pic systolique)" if pulsatile else ""
    st.caption(f"Surface continue en Y = {field_name} [{fd['unit']}]{_note}. Clic-glisser pour "
               "pivoter. CCA en bas, ICA (sténosée) à gauche, ECA à droite. Même échelle. "
               f"Maillage : {len(geom_s['verts'])} sommets, {len(geom_s['faces'])} faces "
               f"(résolution {1.2} mm).")

    st.divider()

    # --- Animation des particules de sang ---
    st.subheader("Écoulement du sang dans la bifurcation")
    _show_animation(
        particle_flow_bifurcation(res, geom_s, U_cca, n, pulsatile, hr),
        height=580)
    _flow = ("Les particules surgissent au rythme cardiaque. " if pulsatile
             else "Débit constant (modèle idéal). ")
    st.caption("Clique sur « Lancer l'écoulement ». " + _flow +
               "Les particules partent de la CCA et se répartissent vers l'ICA et l'ECA selon "
               "le débit ; elles accélèrent (couleur chaude) en passant le goulot de la sténose.")

    st.divider()

    # --- Champ 2D + carte WSS ---
    if show_wss:
        g1, g2 = st.columns(2)
        with g1:
            _show_animation(velocity_figure(res, geom_s, n, show_streamlines), 580)
        with g2:
            _show_animation(wss_figure(res, geom_s), 580)
    else:
        _show_animation(velocity_figure(res, geom_s, n, show_streamlines), 580)

    st.caption("Vue 2D : la CCA (entrée) est en bas ; l'ICA (sténosée) monte quasi-droite à "
               f"gauche, l'ECA part à droite. La plaque s'étend sur {Ls_mm} mm depuis la "
               "bifurcation. Quand l'ICA se bouche, une partie du sang est déviée vers l'ECA.")

    st.divider()

    # --- Profils le long de l'ICA (artère bouchée) ---
    st.subheader("Profils le long de l'ICA (artère bouchée)")
    _show_animation(ica_profiles_figure(res, res_h), 380)
    st.caption("Trajet CCA → ICA. Bleu = artère saine, rouge = artère bouchée (valeurs "
               "annotées). La ligne pointillée marque la bifurcation. Au goulot : vitesse et "
               "WSS montent, pression chute.")

    # --- Répartition du débit ---
    st.subheader("Répartition du débit ICA / ECA")
    split = go.Figure(go.Bar(
        x=["ICA (cerveau)", "ECA (face/cou)"],
        y=[m["Q_ica_frac"] * 100, m["Q_eca_frac"] * 100],
        marker_color=["#F44336", "#2196F3"],
        text=[f"{m['Q_ica_frac']*100:.0f} %", f"{m['Q_eca_frac']*100:.0f} %"],
        textposition="outside"))
    split.add_hline(y=65, line_dash="dot", line_color="gray",
                    annotation_text="ICA sain ~65 %")
    split.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10),
                        yaxis_title="% du débit CCA", yaxis_range=[0, 100])
    _show_animation(split, 320)
    st.caption("En l'absence de sténose, ~65 % du débit va vers l'ICA (irrigation cérébrale). "
               "Plus la sténose ICA augmente, plus le débit est dévié vers l'ECA.")

    st.divider()

    # --- Vraie CFD sur la bifurcation (remontée au-dessus des explications) ---
    with st.expander("Vraie CFD (Lattice-Boltzmann) sur la bifurcation"):
        st.markdown("""
        Calcul hors-ligne qui résout Navier-Stokes sur un maillage (≠ modèle analytique du
        site). Il capture la recirculation dans les branches.

        | Paramètre | Valeur |
        |---|---|
        | Méthode | Lattice-Boltzmann D2Q9, collision BGK |
        | Maillage | 330 × 210 = 69 300 cellules (~11 700 fluides) |
        | Échelle | 3,75 cellules / mm (D_CCA = 30 cellules) |
        | Sténose ICA | 60 % (cosinus, plaque 20 mm) |
        | Vitesse d'entrée | 0,03 (unités réseau) |
        | Reynolds | 50 |
        | Viscosité ν | 0,018 |
        | Relaxation τ | 0,554 |
        | Itérations | 16 000 |
        | Parois / entrée / sorties | rebond / CCA en bas / ICA+ECA en haut (sortie libre) |
        """)
        import os as _os
        _cfdb = "results/figures/real_cfd_bifurcation.png"
        if _os.path.exists(_cfdb):
            st.image(_cfdb, caption="CFD Lattice-Boltzmann sur la bifurcation 2D — "
                     "division CCA vers ICA/ECA et recirculation en aval de la sténose.")
        else:
            st.info("Lance `python run_real_cfd_bifurcation.py` pour générer la figure.")

    # --- Explications, modèle, références : tout à la fin ---
    with st.expander("Comparaison : modèle simple (tube droit) vs réel (bifurcation)"):
        d = stenosis / 100.0
        v_simple = U_cca / max((1 - d) ** 2, 1e-6) * (3 * n + 1) / (n + 1)
        v_real = m["v_throat"]
        comp = pd.DataFrame({
            "Aspect": ["Géométrie", "Sorties", "Vitesse max au goulot",
                       "Débit dans la branche sténosée", "Recirculation",
                       "Écoulement asymétrique"],
            "Modèle simple (tube droit)": [
                "1 tube droit", "1", f"{v_simple:.2f} m/s",
                "100 % du débit", "non capturée", "non"],
            "Modèle réel (bifurcation)": [
                "tronc + 2 branches", "2 (cerveau + face)", f"{v_real:.2f} m/s",
                f"{m['Q_ica_frac']*100:.0f} % (reste dévié vers l'ECA)",
                "présente en aval", "oui"],
        })
        st.dataframe(comp, use_container_width=True, hide_index=True)
        st.markdown(f"""
        Ce qui diffère :
        - Vitesse au goulot : le tube droit la surestime ({v_simple:.2f} vs {v_real:.2f} m/s).
        - Répartition du débit : la bifurcation dévie une partie du sang vers l'ECA.
        - Recirculation : tourbillons en aval, capturés par la vraie CFD, absents du modèle 1D.
        - Turbulence et effets 3D : présents en réalité, hors de portée du modèle 1D.

        Pression : la valeur ΔP affichée est la chute de pression (frottement), pas la pression
        absolue (~80–120 mmHg). Faible dans une artère saine, élevée sur une sténose serrée.
        """)

    with st.expander("Modèle de calcul"):
        st.markdown("""
        Géométrie : CCA Ø 8 mm, ICA Ø 6,17 mm, ECA Ø 4,5 mm, bifurcation asymétrique.
        Sténose ICA : profil cosinus démarrant à la bifurcation.

        Moteur : analytique Power Law quasi-1D le long de chaque branche. La répartition du
        débit ICA/ECA est calculée par la résistance hydraulique de chaque branche (∝ ∫ ds / r⁴).

        Ordres de grandeur attendus :

        | Quantité | Attendu |
        |---|---|
        | Vitesse sténose 50 % | ~2× CCA |
        | Vitesse sténose 70 % | ~3,3× CCA |
        | WSS sténose vs sain | 4–10× |
        | Q_ICA / Q_CCA (sain) | 0,60–0,70 |
        | Recirculation en aval | présente |

        Limite : le modèle 1D ne résout pas la recirculation — c'est la vraie CFD (ci-dessus)
        qui la capture.
        """)

    with st.expander("Références"):
        st.markdown("""
        - Bharadvaj, Mabon & Giddens (1982) — géométrie de la bifurcation carotidienne, *J. Biomech.* 15:349.
        - Ku, Giddens, Zarins & Glagov (1985) — écoulement pulsé et athérosclérose, *Arteriosclerosis* 5:293.
        - Gijsen, van de Vosse & Janssen (1999) — sang non-newtonien, *J. Biomech.* 32:705.
        - Zarins et al. (1983) — répartition du débit carotidien.
        - Young & Tsai (1973) — profil de sténose cosinus, *J. Biomech.* 6:395.
        """)
