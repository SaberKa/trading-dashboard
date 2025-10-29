import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import numpy as np

# =============================
# Configuration de la page
# =============================
st.set_page_config(page_title="Crypto Portfolio Dashboard", layout="wide", initial_sidebar_state="expanded")

# =============================
# CSS personnalisé pour un design moderne
# =============================
st.markdown("""
<style>
    /* Amélioration générale */
    .main {
        padding: 0rem 1rem;
    }

    /* Cartes métriques personnalisées */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }

    .metric-card-positive {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }

    .metric-card-negative {
        background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
    }

    .metric-card-neutral {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }

    /* Titres */
    h1 {
        color: #1f2937;
        font-weight: 700;
        margin-bottom: 2rem;
    }

    h2 {
        color: #374151;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    h3 {
        color: #4b5563;
        font-weight: 500;
    }

    /* Dataframes */
    .dataframe {
        font-size: 0.9rem;
    }

    /* Sidebar */
    .css-1d391kg {
        background-color: #f9fafb;
    }

    /* Boutons et sélections */
    .stSelectbox {
        margin-bottom: 1rem;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
    }

    .badge-success {
        background-color: #10b981;
        color: white;
    }

    .badge-danger {
        background-color: #ef4444;
        color: white;
    }

    .badge-info {
        background-color: #3b82f6;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# =============================
# Import fonction prix (API publique)
# =============================
import requests

def get_current_price(symbol: str):
    """Récupère le prix actuel via l'API publique Binance (sans authentification)"""
    try:
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return float(data['price'])
        return None
    except Exception:
        return None


# =============================
# Cache Prix avec fallback USDT/USDC
# =============================
class PriceCache:
    def __init__(self):
        self.cache = {}

    def get_with_fallback(self, symbol: str):
        if symbol in self.cache:
            return self.cache[symbol]

        # Essai direct
        try:
            price = get_current_price(symbol)
            if price is not None:
                self.cache[symbol] = price
                return price
        except Exception:
            pass

        # Fallback USDT/USDC
        alt = None
        if symbol.endswith("USDC"):
            alt = symbol[:-4] + "USDT"
        elif symbol.endswith("USDT"):
            alt = symbol[:-4] + "USDC"

        if alt:
            try:
                price = get_current_price(alt)
                if price is not None:
                    self.cache[symbol] = price
                    return price
            except Exception:
                pass

        self.cache[symbol] = None
        return None


price_cache = PriceCache()


# =============================
# Fonctions de chargement et préparation
# =============================
def recalculate_quantities_with_reinvestment(trades_df: pd.DataFrame,
                                             initial_capital_per_symbol: float = 1000.0) -> pd.DataFrame:
    """
    Recalcule les quantités en supposant :
    - Chaque premier BUY d'un symbole = initial_capital_per_symbol
    - Chaque SELL suivant vend TOUT
    - Chaque BUY suivant réinvestit TOUT le capital (capital initial +/- gains/pertes)
    """
    df = trades_df.copy()
    df = df.sort_values("datetime").reset_index(drop=True)

    # Tracking du capital par symbole
    symbol_capital = {}  # symbol -> capital actuel disponible
    symbol_positions = {}  # symbol -> quantité actuellement détenue

    new_quantities = []

    for idx, row in df.iterrows():
        symbol = row["symbol"]
        action = row["action"]
        price = float(row["price"])

        if symbol not in symbol_capital:
            symbol_capital[symbol] = initial_capital_per_symbol
            symbol_positions[symbol] = 0.0

        if action == "buy":
            # Utiliser tout le capital disponible pour acheter
            capital_to_invest = symbol_capital[symbol]
            quantity = capital_to_invest / price
            symbol_positions[symbol] = quantity
            new_quantities.append(quantity)

        elif action == "sell":
            # Vendre TOUTE la position
            quantity = symbol_positions[symbol]
            sale_value = quantity * price

            # Mettre à jour le capital (on récupère la valeur de vente)
            symbol_capital[symbol] = sale_value
            symbol_positions[symbol] = 0.0
            new_quantities.append(quantity)

    df["quantity"] = new_quantities
    return df


# =============================
# Fonctions de chargement et préparation
# =============================
@st.cache_data(show_spinner=False)
def load_trades(path: str, recalculate_qty: bool = True, initial_capital: float = 1000.0) -> pd.DataFrame:
    """Charge et nettoie le fichier des trades"""
    df = pd.read_csv(path, parse_dates=["datetime"], dayfirst=True)
    df = df.sort_values("datetime").reset_index(drop=True)

    # Nettoyage et typage
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0.0)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    df["confiance"] = pd.to_numeric(df.get("confiance"), errors="coerce") if "confiance" in df.columns else None
    df["action"] = df["action"].str.lower().str.strip()

    # Recalculer les quantités avec réinvestissement
    if recalculate_qty:
        df = recalculate_quantities_with_reinvestment(df, initial_capital)

    return df


# =============================
# Calculs PnL avec FIFO
# =============================
def calculate_pnl_fifo(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule le PnL réalisé avec la méthode FIFO (First In First Out).
    Chaque vente est appariée avec le(s) achat(s) le(s) plus ancien(s).
    """
    positions = {}  # symbol -> liste de lots {qty, price, dt}
    realized_rows = []

    for _, row in trades_df.iterrows():
        symbol = row["symbol"]
        action = row["action"]
        qty = float(row["quantity"])
        price = float(row["price"])
        dt = row["datetime"]
        confiance = row.get("confiance")

        if symbol not in positions:
            positions[symbol] = []

        if action == "buy":
            positions[symbol].append({"qty": qty, "price": price, "dt": dt})

        elif action == "sell":
            qty_to_sell = qty

            while qty_to_sell > 1e-8 and positions[symbol]:  # Seuil de tolérance pour erreurs d'arrondi
                lot = positions[symbol][0]
                available = lot["qty"]
                matched_qty = min(qty_to_sell, available)

                buy_price = lot["price"]
                cost_usd = buy_price * matched_qty
                pnl_usd = (price - buy_price) * matched_qty
                pnl_pct = (pnl_usd / cost_usd * 100.0) if cost_usd > 0 else 0.0

                realized_rows.append({
                    "datetime": dt,
                    "symbol": symbol,
                    "action": "sell",
                    "quantity": matched_qty,
                    "price_buy": buy_price,
                    "price_sell": price,
                    "cost_$": cost_usd,
                    "pnl_$": pnl_usd,
                    "pnl_%": pnl_pct,
                    "confiance": confiance,
                    "buy_date": lot["dt"]
                })

                qty_to_sell -= matched_qty

                if matched_qty >= available - 1e-8:  # Seuil de tolérance
                    positions[symbol].pop(0)
                else:
                    positions[symbol][0]["qty"] -= matched_qty

    if not realized_rows:
        return pd.DataFrame()

    realized = pd.DataFrame(realized_rows).sort_values("datetime").reset_index(drop=True)
    realized["global_cum_pnl_$"] = realized["pnl_$"].cumsum()

    return realized


# =============================
# Positions ouvertes
# =============================
def get_open_positions(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Récupère les positions encore ouvertes (lots FIFO non vendus)"""
    positions = {}

    for _, row in trades_df.iterrows():
        symbol = row["symbol"]
        action = row["action"]
        qty = float(row["quantity"])
        price = float(row["price"])
        dt = row["datetime"]

        if symbol not in positions:
            positions[symbol] = []

        if action == "buy":
            positions[symbol].append({"qty": qty, "price": price, "dt": dt})
        elif action == "sell":
            qty_to_sell = qty
            while qty_to_sell > 1e-8 and positions[symbol]:  # Seuil de tolérance
                lot = positions[symbol][0]
                available = lot["qty"]
                matched_qty = min(qty_to_sell, available)

                if matched_qty >= available - 1e-8:  # Seuil de tolérance
                    positions[symbol].pop(0)
                else:
                    positions[symbol][0]["qty"] -= matched_qty

                qty_to_sell -= matched_qty

    open_rows = []
    for symbol, lots in positions.items():
        for lot in lots:
            if lot["qty"] > 1e-8:  # Seuil de tolérance
                open_rows.append({
                    "symbol": symbol,
                    "quantity": float(lot["qty"]),
                    "price_buy": float(lot["price"]),
                    "datetime": lot["dt"],
                })

    if not open_rows:
        return pd.DataFrame()

    return pd.DataFrame(open_rows).sort_values(["symbol", "datetime"]).reset_index(drop=True)


# =============================
# Enrichissement avec prix actuels
# =============================
def enrich_with_live_prices(open_df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les prix actuels et calcule le PnL latent"""
    if open_df.empty:
        return open_df

    rows = []
    for _, r in open_df.iterrows():
        symbol = r["symbol"]
        qty = float(r["quantity"])
        price_buy = float(r["price_buy"])

        price_now = price_cache.get_with_fallback(symbol)

        if price_now is None or price_buy == 0:
            cost_usd = price_buy * qty
            rows.append({
                "current_price": None,
                "value_$": None,
                "cost_$": cost_usd,
                "pnl_live_$": None,
                "pnl_live_%": None,
            })
        else:
            cost_usd = price_buy * qty
            value_usd = price_now * qty
            pnl_usd = value_usd - cost_usd
            pnl_pct = (pnl_usd / cost_usd * 100.0) if cost_usd > 0 else 0.0

            rows.append({
                "current_price": float(price_now),
                "value_$": float(value_usd),
                "cost_$": float(cost_usd),
                "pnl_live_$": float(pnl_usd),
                "pnl_live_%": float(pnl_pct),
            })

    extra = pd.DataFrame(rows)
    return pd.concat([open_df.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)


# =============================
# Statistiques de trading
# =============================
def calculate_trading_stats(realized_pnl: pd.DataFrame) -> dict:
    """Calcule des statistiques de performance du trading"""
    if realized_pnl.empty:
        return {}

    total_trades = len(realized_pnl)
    winning_trades = len(realized_pnl[realized_pnl["pnl_$"] > 0])
    losing_trades = len(realized_pnl[realized_pnl["pnl_$"] < 0])

    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

    avg_win = realized_pnl[realized_pnl["pnl_$"] > 0]["pnl_$"].mean() if winning_trades > 0 else 0
    avg_loss = realized_pnl[realized_pnl["pnl_$"] < 0]["pnl_$"].mean() if losing_trades > 0 else 0

    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

    max_win = realized_pnl["pnl_$"].max()
    max_loss = realized_pnl["pnl_$"].min()

    # Calcul du drawdown maximum
    cumulative = realized_pnl["global_cum_pnl_$"]
    running_max = cumulative.expanding().max()
    drawdown = cumulative - running_max
    max_drawdown = drawdown.min()

    # Durée moyenne des trades (en jours)
    if "buy_date" in realized_pnl.columns:
        realized_pnl_copy = realized_pnl.copy()
        # S'assurer que les colonnes sont bien en datetime avec format mixte
        realized_pnl_copy["datetime"] = pd.to_datetime(realized_pnl_copy["datetime"], format='mixed', dayfirst=True)
        realized_pnl_copy["buy_date"] = pd.to_datetime(realized_pnl_copy["buy_date"], format='mixed', dayfirst=True)
        realized_pnl_copy["holding_period"] = (realized_pnl_copy["datetime"] - realized_pnl_copy[
            "buy_date"]).dt.total_seconds() / 86400
        avg_holding = realized_pnl_copy["holding_period"].mean()
    else:
        avg_holding = None

    return {
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "max_win": max_win,
        "max_loss": max_loss,
        "max_drawdown": max_drawdown,
        "avg_holding_days": avg_holding
    }


# =============================
# Buy & Hold Comparison
# =============================
def calculate_buy_and_hold(trades_df: pd.DataFrame, open_positions_enriched: pd.DataFrame) -> dict:
    """
    Calcule la performance Buy & Hold basée sur les positions actuellement ouvertes.
    C'est la comparaison la plus pertinente : "Si j'avais gardé mes positions ouvertes depuis l'achat"
    """
    if open_positions_enriched.empty:
        return {"total_cost": 0, "total_value": 0, "total_pnl": 0, "total_pct": 0, "per_symbol": {}}

    # Filtrer les positions avec prix actuel disponible
    valid_positions = open_positions_enriched[open_positions_enriched["current_price"].notna()].copy()

    if valid_positions.empty:
        return {"total_cost": 0, "total_value": 0, "total_pnl": 0, "total_pct": 0, "per_symbol": {}}

    total_cost = float(valid_positions["cost_$"].sum())
    total_value = float(valid_positions["value_$"].sum())
    total_pnl = total_value - total_cost
    total_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0

    # Par symbole
    per_symbol = {}
    for symbol in valid_positions["symbol"].unique():
        symbol_data = valid_positions[valid_positions["symbol"] == symbol]
        cost = float(symbol_data["cost_$"].sum())
        value = float(symbol_data["value_$"].sum())
        pnl = value - cost
        pct = (pnl / cost * 100) if cost > 0 else 0

        per_symbol[symbol] = {
            "cost": cost,
            "value": value,
            "pnl": pnl,
            "pct": pct,
            "current_price": float(symbol_data["current_price"].iloc[0])
        }

    return {
        "total_cost": total_cost,
        "total_value": total_value,
        "total_pnl": total_pnl,
        "total_pct": total_pct,
        "per_symbol": per_symbol
    }


# =============================
# Fonctions d'affichage
# =============================
def display_metric_card(label: str, value: str, delta: str = None, card_type: str = "neutral"):
    """Affiche une carte métrique stylisée"""
    card_class = f"metric-card metric-card-{card_type}"
    delta_html = f"<div style='font-size: 0.9rem; margin-top: 0.5rem;'>{delta}</div>" if delta else ""

    st.markdown(f"""
    <div class='{card_class}'>
        <div style='font-size: 0.85rem; opacity: 0.9; margin-bottom: 0.5rem;'>{label}</div>
        <div style='font-size: 2rem; font-weight: 700;'>{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def format_currency(value: float, decimals: int = 2) -> str:
    """Formate une valeur en devise avec symbole"""
    if value >= 0:
        return f"${value:,.{decimals}f}"
    else:
        return f"-${abs(value):,.{decimals}f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Formate un pourcentage avec signe"""
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{decimals}f}%"


# =============================
# Interface principale
# =============================
st.title("📊 Crypto Portfolio Dashboard")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    csv_path = st.text_input("📁 Fichier des trades", value="trades_log.csv")

    st.markdown("---")
    st.markdown("### 💰 Capital")
    recalc_qty = st.checkbox("Recalculer les quantités", value=True,
                             help="Recalcule les quantités en supposant un capital initial par symbole et réinvestissement total")

    if recalc_qty:
        initial_capital = st.number_input("Capital initial par symbole ($)",
                                          min_value=100.0,
                                          max_value=100000.0,
                                          value=1000.0,
                                          step=100.0)
    else:
        initial_capital = 1000.0

    st.markdown("---")
    st.markdown("### 📈 Filtres")
    date_filter = st.checkbox("Filtrer par période", value=False)

    if date_filter:
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("De", value=pd.to_datetime("2025-01-01"))
        with col2:
            end_date = st.date_input("À", value=datetime.now())

# Chargement des données
try:
    with st.spinner("Chargement des données..."):
        df = load_trades(csv_path, recalculate_qty=recalc_qty, initial_capital=initial_capital)

        # Application du filtre de date si activé
        if date_filter:
            df = df[(df["datetime"].dt.date >= start_date) & (df["datetime"].dt.date <= end_date)]

        if df.empty:
            st.error("Aucune donnée à afficher pour cette période.")
            st.stop()

except Exception as e:
    st.error(f"❌ Erreur de chargement : {e}")
    st.stop()

# Calculs principaux
with st.spinner("Calcul des métriques..."):
    realized_pnl = calculate_pnl_fifo(df)
    open_positions = get_open_positions(df)
    open_positions_enriched = enrich_with_live_prices(open_positions)
    trading_stats = calculate_trading_stats(realized_pnl)
    buy_hold_stats = calculate_buy_and_hold(df, open_positions_enriched)

# Debug des quantités recalculées
if recalc_qty:
    with st.expander("🔍 Voir les quantités recalculées"):
        st.caption(
            "Les quantités ont été recalculées en supposant un réinvestissement total du capital après chaque trade")
        debug_trades = df[["datetime", "symbol", "action", "quantity", "price"]].copy()
        debug_trades["value_$"] = debug_trades["quantity"] * debug_trades["price"]
        st.dataframe(debug_trades, use_container_width=True, hide_index=True)

# =============================
# ONGLETS PRINCIPAUX
# =============================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Vue Globale", "💹 Positions Ouvertes", "📈 Par Crypto", "📉 Statistiques"])

# =============================
# TAB 1: VUE GLOBALE
# =============================
with tab1:
    st.subheader("Performance Globale du Portefeuille")

    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if not realized_pnl.empty:
            realized_total = float(realized_pnl["pnl_$"].sum())
            card_type = "positive" if realized_total > 0 else "negative" if realized_total < 0 else "neutral"
            display_metric_card(
                "PnL Réalisé Total",
                format_currency(realized_total),
                card_type=card_type
            )
        else:
            display_metric_card("PnL Réalisé Total", "$0.00", card_type="neutral")

    with col2:
        if not open_positions_enriched.empty and open_positions_enriched["pnl_live_$"].notna().any():
            latent_total = float(open_positions_enriched["pnl_live_$"].fillna(0).sum())
            card_type = "positive" if latent_total > 0 else "negative" if latent_total < 0 else "neutral"
            display_metric_card(
                "PnL Latent Total",
                format_currency(latent_total),
                card_type=card_type
            )
        else:
            display_metric_card("PnL Latent Total", "$0.00", card_type="neutral")

    with col3:
        realized = float(realized_pnl["pnl_$"].sum()) if not realized_pnl.empty else 0
        latent = float(
            open_positions_enriched["pnl_live_$"].fillna(0).sum()) if not open_positions_enriched.empty else 0
        total_pnl = realized + latent
        card_type = "positive" if total_pnl > 0 else "negative" if total_pnl < 0 else "neutral"
        display_metric_card(
            "PnL Total Portfolio",
            format_currency(total_pnl),
            card_type=card_type
        )

    with col4:
        if not open_positions_enriched.empty and open_positions_enriched["value_$"].notna().any():
            portfolio_value = float(open_positions_enriched["value_$"].fillna(0).sum())
            display_metric_card(
                "Valeur Portfolio",
                format_currency(portfolio_value),
                card_type="info"
            )
        else:
            display_metric_card("Valeur Portfolio", "$0.00", card_type="neutral")

    st.markdown("---")

    # Graphique d'évolution du PnL
    if not realized_pnl.empty:
        st.subheader("📈 Évolution du PnL")

        fig = make_subplots(specs=[[{"secondary_y": False}]])

        # PnL cumulé réalisé
        fig.add_trace(
            go.Scatter(
                x=realized_pnl["datetime"],
                y=realized_pnl["global_cum_pnl_$"],
                mode="lines",
                name="PnL Réalisé Cumulé",
                line=dict(color="#3b82f6", width=3),
                fill='tozeroy',
                fillcolor='rgba(59, 130, 246, 0.1)'
            )
        )

        # Point actuel (réalisé + latent)
        if not open_positions_enriched.empty and open_positions_enriched["pnl_live_$"].notna().any():
            last_realized = float(realized_pnl["global_cum_pnl_$"].iloc[-1])
            last_date = realized_pnl["datetime"].max()
            latent = float(open_positions_enriched["pnl_live_$"].fillna(0).sum())
            current_total = last_realized + latent

            # Ligne pointillée vers "maintenant"
            fig.add_trace(
                go.Scatter(
                    x=[last_date, datetime.now()],
                    y=[last_realized, current_total],
                    mode="lines",
                    name="Projection avec latent",
                    line=dict(color="#10b981", width=2, dash="dash"),
                )
            )

            # Point actuel
            fig.add_trace(
                go.Scatter(
                    x=[datetime.now()],
                    y=[current_total],
                    mode="markers+text",
                    name="PnL Total Actuel",
                    marker=dict(size=15, color="#10b981", symbol="diamond"),
                    text=["Maintenant"],
                    textposition="top center",
                )
            )

        # Ligne de zéro
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)

        fig.update_layout(
            title="",
            xaxis_title="Date",
            yaxis_title="PnL Cumulé ($)",
            hovermode='x unified',
            height=500,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

    # Comparaison avec Buy & Hold
    st.markdown("---")
    st.subheader("🔄 Comparaison avec Buy & Hold")
    st.caption(
        "Comparaison entre votre stratégie de trading et une stratégie Buy & Hold passive sur les positions ouvertes")

    col1, col2, col3 = st.columns(3)

    with col1:
        bh_pnl = buy_hold_stats["total_pnl"]
        bh_type = "positive" if bh_pnl > 0 else "negative" if bh_pnl < 0 else "neutral"
        display_metric_card(
            "PnL Buy & Hold",
            format_currency(bh_pnl),
            format_percentage(buy_hold_stats["total_pct"]),
            card_type=bh_type
        )

    with col2:
        realized = float(realized_pnl["pnl_$"].sum()) if not realized_pnl.empty else 0
        latent = float(
            open_positions_enriched["pnl_live_$"].fillna(0).sum()) if not open_positions_enriched.empty else 0
        trading_pnl = realized + latent

        diff = trading_pnl - bh_pnl
        diff_type = "positive" if diff > 0 else "negative" if diff < 0 else "neutral"
        display_metric_card(
            "Différence vs B&H",
            format_currency(diff),
            "Meilleur que B&H" if diff > 0 else "Moins bon que B&H" if diff < 0 else "Égal à B&H",
            card_type=diff_type
        )

    with col3:
        if buy_hold_stats["total_cost"] > 0:
            bh_roi = buy_hold_stats["total_pct"]

            # ROI du trading
            total_cost_realized = float(realized_pnl["cost_$"].sum()) if not realized_pnl.empty else 0
            total_cost_open = float(open_positions_enriched["cost_$"].sum()) if not open_positions_enriched.empty else 0
            total_cost = total_cost_realized + total_cost_open

            trading_roi = (trading_pnl / total_cost * 100) if total_cost > 0 else 0

            roi_diff = trading_roi - bh_roi
            roi_type = "positive" if roi_diff > 0 else "negative" if roi_diff < 0 else "neutral"

            display_metric_card(
                "Différence ROI",
                format_percentage(roi_diff),
                f"Trading: {format_percentage(trading_roi)} | B&H: {format_percentage(bh_roi)}",
                card_type=roi_type
            )

# =============================
# TAB 2: POSITIONS OUVERTES
# =============================
with tab2:
    st.subheader("💼 Positions Actuellement Ouvertes")

    if not open_positions_enriched.empty:
        # Résumé par symbole
        summary_rows = []
        for symbol in open_positions_enriched["symbol"].unique():
            symbol_data = open_positions_enriched[open_positions_enriched["symbol"] == symbol]

            total_qty = float(symbol_data["quantity"].sum())
            avg_buy = float((symbol_data["price_buy"] * symbol_data["quantity"]).sum() / total_qty)
            current_price = symbol_data["current_price"].iloc[0] if symbol_data["current_price"].notna().any() else None
            cost = float(symbol_data["cost_$"].sum())
            value = float(symbol_data["value_$"].sum()) if symbol_data["value_$"].notna().any() else None
            pnl = float(symbol_data["pnl_live_$"].sum()) if symbol_data["pnl_live_$"].notna().any() else None
            pnl_pct = (pnl / cost * 100) if (pnl is not None and cost > 0) else None

            summary_rows.append({
                "Symbole": symbol,
                "Quantité": total_qty,
                "Prix Moyen Achat": avg_buy,
                "Prix Actuel": current_price,
                "Coût Total ($)": cost,
                "Valeur Actuelle ($)": value,
                "PnL Latent ($)": pnl,
                "PnL Latent (%)": pnl_pct
            })

        summary_df = pd.DataFrame(summary_rows)

        # Formatage pour l'affichage
        display_df = summary_df.copy()
        for col in ["Prix Moyen Achat", "Prix Actuel", "Coût Total ($)", "Valeur Actuelle ($)", "PnL Latent ($)"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")

        if "PnL Latent (%)" in display_df.columns:
            display_df["PnL Latent (%)"] = display_df["PnL Latent (%)"].apply(
                lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A"
            )

        display_df["Quantité"] = display_df["Quantité"].apply(lambda x: f"{x:.6f}".rstrip('0').rstrip('.'))

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Graphique de répartition
        st.markdown("---")
        st.subheader("📊 Répartition du Portfolio")

        col1, col2 = st.columns(2)

        with col1:
            # Pie chart par valeur
            valid_summary = summary_df[summary_df["Valeur Actuelle ($)"].notna()]
            if not valid_summary.empty:
                fig_pie = go.Figure(data=[go.Pie(
                    labels=valid_summary["Symbole"],
                    values=valid_summary["Valeur Actuelle ($)"],
                    hole=0.4,
                    marker=dict(colors=['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'])
                )])
                fig_pie.update_layout(
                    title="Répartition par Valeur ($)",
                    height=400,
                    template="plotly_white"
                )
                st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Bar chart PnL latent
            valid_pnl = summary_df[summary_df["PnL Latent ($)"].notna()]
            if not valid_pnl.empty:
                colors = ['#10b981' if x > 0 else '#ef4444' for x in valid_pnl["PnL Latent ($)"]]
                fig_bar = go.Figure(data=[go.Bar(
                    x=valid_pnl["Symbole"],
                    y=valid_pnl["PnL Latent ($)"],
                    marker_color=colors
                )])
                fig_bar.update_layout(
                    title="PnL Latent par Crypto ($)",
                    xaxis_title="",
                    yaxis_title="PnL ($)",
                    height=400,
                    template="plotly_white"
                )
                fig_bar.add_hline(y=0, line_dash="dot", line_color="gray")
                st.plotly_chart(fig_bar, use_container_width=True)

        # Détails des lots
        st.markdown("---")
        st.subheader("📋 Détail des Lots Ouverts")

        detail_df = open_positions_enriched.copy()
        detail_df = detail_df.sort_values(["symbol", "datetime"])

        # Formatage
        display_detail = detail_df[
            ["symbol", "quantity", "price_buy", "datetime", "current_price", "value_$", "cost_$", "pnl_live_$",
             "pnl_live_%"]].copy()
        display_detail.columns = ["Symbole", "Quantité", "Prix Achat", "Date Achat", "Prix Actuel", "Valeur ($)",
                                  "Coût ($)", "PnL ($)", "PnL (%)"]

        for col in ["Prix Achat", "Prix Actuel", "Valeur ($)", "Coût ($)", "PnL ($)"]:
            display_detail[col] = display_detail[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")

        display_detail["PnL (%)"] = display_detail["PnL (%)"].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A")
        display_detail["Quantité"] = display_detail["Quantité"].apply(lambda x: f"{x:.6f}".rstrip('0').rstrip('.'))

        st.dataframe(display_detail, use_container_width=True, hide_index=True)

    else:
        st.info("Aucune position ouverte actuellement.")

# =============================
# TAB 3: VUE PAR CRYPTO
# =============================
with tab3:
    st.subheader("🔍 Analyse par Crypto")

    if not df.empty:
        symbols = sorted(df["symbol"].unique())
        selected_symbol = st.selectbox("Sélectionner une crypto", symbols)

        # Données du symbole sélectionné
        symbol_realized = realized_pnl[
            realized_pnl["symbol"] == selected_symbol].copy() if not realized_pnl.empty else pd.DataFrame()
        symbol_open = open_positions_enriched[open_positions_enriched[
                                                  "symbol"] == selected_symbol].copy() if not open_positions_enriched.empty else pd.DataFrame()

        # Métriques du symbole
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if not symbol_realized.empty:
                pnl = float(symbol_realized["pnl_$"].sum())
                card_type = "positive" if pnl > 0 else "negative" if pnl < 0 else "neutral"
                display_metric_card(
                    f"PnL Réalisé {selected_symbol}",
                    format_currency(pnl),
                    card_type=card_type
                )
            else:
                display_metric_card(f"PnL Réalisé {selected_symbol}", "$0.00", card_type="neutral")

        with col2:
            if not symbol_open.empty and symbol_open["pnl_live_$"].notna().any():
                latent = float(symbol_open["pnl_live_$"].sum())
                card_type = "positive" if latent > 0 else "negative" if latent < 0 else "neutral"
                display_metric_card(
                    f"PnL Latent {selected_symbol}",
                    format_currency(latent),
                    card_type=card_type
                )
            else:
                display_metric_card(f"PnL Latent {selected_symbol}", "$0.00", card_type="neutral")

        with col3:
            if not symbol_realized.empty:
                nb_trades = len(symbol_realized)
                wins = len(symbol_realized[symbol_realized["pnl_$"] > 0])
                win_rate = (wins / nb_trades * 100) if nb_trades > 0 else 0

                display_metric_card(
                    "Win Rate",
                    f"{win_rate:.1f}%",
                    f"{wins} / {nb_trades} trades",
                    card_type="info"
                )
            else:
                display_metric_card("Win Rate", "N/A", card_type="neutral")

        with col4:
            bh_symbol = buy_hold_stats["per_symbol"].get(selected_symbol)
            if bh_symbol:
                bh_pnl = bh_symbol["pnl"]
                bh_type = "positive" if bh_pnl > 0 else "negative" if bh_pnl < 0 else "neutral"
                display_metric_card(
                    f"B&H {selected_symbol}",
                    format_currency(bh_pnl),
                    format_percentage(bh_symbol["pct"]),
                    card_type=bh_type
                )
            else:
                display_metric_card(f"B&H {selected_symbol}", "N/A", card_type="neutral")

        st.markdown("---")

        # Graphiques
        if not symbol_realized.empty:
            # Identifier les sorties par stop loss
            symbol_realized["is_stop"] = symbol_realized["confiance"].apply(
                lambda x: True if pd.notna(x) and float(x) == 0.0 else False
            )

            # DEBUG: Afficher les détails des calculs
            with st.expander("🔍 Debug - Détails des calculs"):
                st.write("**Trades réalisés (brut) :**")
                debug_df = symbol_realized[
                    ["datetime", "quantity", "price_buy", "price_sell", "cost_$", "pnl_$"]].copy()
                st.dataframe(debug_df, use_container_width=True)
                st.write(f"**Somme PnL:** ${symbol_realized['pnl_$'].sum():.2f}")
                st.write(f"**Nombre de trades:** {len(symbol_realized)}")

            # Graphique des trades
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.6, 0.4],
                subplot_titles=("PnL Cumulé", "PnL par Trade"),
                vertical_spacing=0.1
            )

            # PnL cumulé
            fig.add_trace(
                go.Scatter(
                    x=symbol_realized["datetime"],
                    y=symbol_realized["pnl_$"].cumsum(),
                    mode="lines+markers",
                    name="PnL Cumulé",
                    line=dict(color="#3b82f6", width=2),
                    marker=dict(size=8)
                ),
                row=1, col=1
            )

            # PnL par trade (barres colorées selon stop/no stop)
            colors = symbol_realized.apply(
                lambda row: '#ef4444' if row["is_stop"] else ('#10b981' if row["pnl_$"] > 0 else '#f59e0b'),
                axis=1
            )

            fig.add_trace(
                go.Bar(
                    x=symbol_realized["datetime"],
                    y=symbol_realized["pnl_$"],
                    name="PnL Trade",
                    marker_color=colors,
                    showlegend=False
                ),
                row=2, col=1
            )

            fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=1)
            fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)

            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="PnL Cumulé ($)", row=1, col=1)
            fig.update_yaxes(title_text="PnL Trade ($)", row=2, col=1)

            fig.update_layout(
                height=700,
                template="plotly_white",
                hovermode='x unified',
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)

            # Légende des couleurs
            st.markdown("""
            <div style='display: flex; gap: 20px; margin-top: -10px; margin-bottom: 20px;'>
                <div><span style='color: #10b981;'>●</span> Trade gagnant</div>
                <div><span style='color: #f59e0b;'>●</span> Trade perdant</div>
                <div><span style='color: #ef4444;'>●</span> Sortie par stop loss</div>
            </div>
            """, unsafe_allow_html=True)

            # Tableau des trades
            st.subheader("📋 Historique des Trades")

            display_trades = symbol_realized[
                ["datetime", "quantity", "price_buy", "price_sell", "cost_$", "pnl_$", "pnl_%", "is_stop"]].copy()
            display_trades.columns = ["Date", "Quantité", "Prix Achat", "Prix Vente", "Coût ($)", "PnL ($)", "PnL (%)",
                                      "Stop Loss"]

            display_trades["Quantité"] = display_trades["Quantité"].apply(lambda x: f"{x:.6f}".rstrip('0').rstrip('.'))
            for col in ["Prix Achat", "Prix Vente", "Coût ($)", "PnL ($)"]:
                display_trades[col] = display_trades[col].apply(lambda x: f"${x:,.2f}")
            display_trades["PnL (%)"] = display_trades["PnL (%)"].apply(lambda x: f"{x:+.2f}%")
            display_trades["Stop Loss"] = display_trades["Stop Loss"].apply(lambda x: "✓" if x else "")

            st.dataframe(display_trades, use_container_width=True, hide_index=True)

        else:
            st.info(f"Aucun trade réalisé sur {selected_symbol}")

        # Position ouverte
        if not symbol_open.empty:
            st.markdown("---")
            st.subheader("📌 Position Ouverte")

            display_open = symbol_open[
                ["quantity", "price_buy", "datetime", "current_price", "value_$", "cost_$", "pnl_live_$",
                 "pnl_live_%"]].copy()
            display_open.columns = ["Quantité", "Prix Achat", "Date Achat", "Prix Actuel", "Valeur ($)", "Coût ($)",
                                    "PnL ($)", "PnL (%)"]

            display_open["Quantité"] = display_open["Quantité"].apply(lambda x: f"{x:.6f}".rstrip('0').rstrip('.'))
            for col in ["Prix Achat", "Prix Actuel", "Valeur ($)", "Coût ($)", "PnL ($)"]:
                display_open[col] = display_open[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
            display_open["PnL (%)"] = display_open["PnL (%)"].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A")

            st.dataframe(display_open, use_container_width=True, hide_index=True)

    else:
        st.info("Aucune donnée disponible")

# =============================
# TAB 4: STATISTIQUES
# =============================
with tab4:
    st.subheader("📉 Statistiques de Trading")

    if trading_stats:
        # Métriques de performance
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            display_metric_card(
                "Nombre de Trades",
                str(trading_stats["total_trades"]),
                f"Gagnants: {trading_stats['winning_trades']} | Perdants: {trading_stats['losing_trades']}",
                card_type="info"
            )

        with col2:
            wr = trading_stats["win_rate"]
            wr_type = "positive" if wr >= 50 else "negative"
            display_metric_card(
                "Win Rate",
                f"{wr:.1f}%",
                card_type=wr_type
            )

        with col3:
            pf = trading_stats["profit_factor"]
            pf_display = f"{pf:.2f}" if pf != float('inf') else "∞"
            pf_type = "positive" if pf > 1 else "negative"
            display_metric_card(
                "Profit Factor",
                pf_display,
                "Ratio gain/perte",
                card_type=pf_type
            )

        with col4:
            md = trading_stats["max_drawdown"]
            display_metric_card(
                "Drawdown Max",
                format_currency(md),
                card_type="negative" if md < 0 else "neutral"
            )

        st.markdown("---")

        # Graphiques statistiques
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("💰 Gains vs Pertes")
            avg_win = trading_stats["avg_win"]
            avg_loss = abs(trading_stats["avg_loss"])

            fig_gainloss = go.Figure(data=[
                go.Bar(
                    x=["Gain Moyen", "Perte Moyenne"],
                    y=[avg_win, avg_loss],
                    marker_color=['#10b981', '#ef4444']
                )
            ])
            fig_gainloss.update_layout(
                title="",
                yaxis_title="Montant ($)",
                template="plotly_white",
                height=350
            )
            st.plotly_chart(fig_gainloss, use_container_width=True)

            st.metric("Gain Moyen", format_currency(avg_win))
            st.metric("Perte Moyenne", format_currency(trading_stats["avg_loss"]))

        with col2:
            st.subheader("🎯 Répartition des Résultats")

            fig_pie_results = go.Figure(data=[go.Pie(
                labels=["Gagnants", "Perdants"],
                values=[trading_stats["winning_trades"], trading_stats["losing_trades"]],
                marker=dict(colors=['#10b981', '#ef4444']),
                hole=0.4
            )])
            fig_pie_results.update_layout(
                title="",
                height=350,
                template="plotly_white"
            )
            st.plotly_chart(fig_pie_results, use_container_width=True)

        st.markdown("---")

        # Records
        col1, col2, col3 = st.columns(3)

        with col1:
            display_metric_card(
                "Meilleur Trade",
                format_currency(trading_stats["max_win"]),
                card_type="positive"
            )

        with col2:
            display_metric_card(
                "Pire Trade",
                format_currency(trading_stats["max_loss"]),
                card_type="negative"
            )

        with col3:
            if trading_stats["avg_holding_days"] is not None:
                display_metric_card(
                    "Durée Moyenne",
                    f"{trading_stats['avg_holding_days']:.1f} jours",
                    card_type="info"
                )
            else:
                display_metric_card("Durée Moyenne", "N/A", card_type="neutral")

        # Distribution des PnL
        if not realized_pnl.empty:
            st.markdown("---")
            st.subheader("📊 Distribution des PnL")

            fig_hist = go.Figure(data=[go.Histogram(
                x=realized_pnl["pnl_$"],
                nbinsx=30,
                marker_color='#3b82f6',
                opacity=0.7
            )])
            fig_hist.update_layout(
                title="",
                xaxis_title="PnL ($)",
                yaxis_title="Fréquence",
                template="plotly_white",
                height=400
            )
            fig_hist.add_vline(x=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_hist, use_container_width=True)

    else:
        st.info("Aucune statistique disponible - aucun trade réalisé.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; font-size: 0.85rem; padding: 1rem;'>
    Dashboard Crypto Portfolio v2.0 | Données mises à jour en temps réel
</div>
""", unsafe_allow_html=True)