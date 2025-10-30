import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import numpy as np
import requests

# =============================
# Configuration de la page
# =============================
st.set_page_config(page_title="Crypto Portfolio Dashboard", layout="wide", initial_sidebar_state="expanded")

# Éviter les erreurs de cache Plotly
import plotly.io as pio

pio.templates.default = "plotly_white"

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

    .metric-card-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
# Import fonction prix (CoinGecko API)
# =============================

# Cache global pour les prix
PRICE_CACHE = {}


def fetch_all_prices_coingecko():
    """Récupère tous les prix en une seule requête"""
    global PRICE_CACHE

    # Liste de tous les coins à récupérer
    coin_ids = [
        "bitcoin", "ethereum", "binancecoin", "cardano", "solana",
        "ripple", "dogecoin", "chainlink", "tron", "stellar"
    ]

    try:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={','.join(coin_ids)}&vs_currencies=usd"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()

            # Mapping inverse : coin_id -> symboles Binance
            mapping = {
                "bitcoin": ["BTCUSDT", "BTCUSDC"],
                "ethereum": ["ETHUSDT", "ETHUSDC"],
                "binancecoin": ["BNBUSDT", "BNBUSDC"],
                "cardano": ["ADAUSDT", "ADAUSDC"],
                "solana": ["SOLUSDT", "SOLUSDC"],
                "ripple": ["XRPUSDT", "XRPUSDC"],
                "dogecoin": ["DOGEUSDT", "DOGEUSDC"],
                "chainlink": ["LINKUSDT", "LINKUSDC"],
                "tron": ["TRXUSDT", "TRXUSDC"],
                "stellar": ["XLMUSDT", "XLMUSDC"],
            }

            # Remplir le cache
            for coin_id, symbols in mapping.items():
                if coin_id in data and 'usd' in data[coin_id]:
                    price = float(data[coin_id]['usd'])
                    for symbol in symbols:
                        PRICE_CACHE[symbol] = price

            return True
    except Exception as e:
        return False

    return False


def get_current_price(symbol: str):
    """Récupère le prix depuis le cache"""
    clean_symbol = symbol.strip().upper()
    return PRICE_CACHE.get(clean_symbol)


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
    df = pd.read_csv(path)

    # Parser les dates avec format mixte et dayfirst=True
    df["datetime"] = pd.to_datetime(df["datetime"], format='mixed', dayfirst=True, errors='coerce')

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
        try:
            realized_pnl_copy = realized_pnl.copy()
            # S'assurer que les colonnes sont bien en datetime
            realized_pnl_copy["datetime"] = pd.to_datetime(realized_pnl_copy["datetime"], errors='coerce')
            realized_pnl_copy["buy_date"] = pd.to_datetime(realized_pnl_copy["buy_date"], errors='coerce')

            # Calculer la durée en jours
            realized_pnl_copy["holding_period"] = (
                                                          realized_pnl_copy["datetime"] - realized_pnl_copy["buy_date"]
                                                  ).dt.total_seconds() / 86400

            avg_holding = realized_pnl_copy["holding_period"].mean()
        except Exception:
            avg_holding = None
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
# Buy & Hold Evolution temporelle (CORRIGÉ)
# =============================
def calculate_bh_evolution(trades_df: pd.DataFrame, realized_pnl: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule l'évolution du PnL Buy & Hold dans le temps.
    Pour chaque symbole, prend le PREMIER achat et simule de le garder jusqu'à maintenant.
    """
    if trades_df.empty:
        return pd.DataFrame()

    # Récupérer tous les achats
    buys = trades_df[trades_df["action"] == "buy"].copy()
    if buys.empty:
        return pd.DataFrame()

    # S'assurer que datetime est bien un datetime
    buys["datetime"] = pd.to_datetime(buys["datetime"], errors='coerce')

    # Pour chaque symbole, identifier le premier achat
    first_buys = {}
    for symbol in buys["symbol"].unique():
        symbol_buys = buys[buys["symbol"] == symbol].sort_values("datetime")
        first_buy = symbol_buys.iloc[0]
        first_buys[symbol] = {
            "quantity": first_buy["quantity"],
            "price": first_buy["price"],
            "date": first_buy["datetime"]
        }

    # Créer une timeline avec toutes les dates de trades réalisés
    if realized_pnl.empty:
        dates = [buys["datetime"].max()]
    else:
        realized_pnl_copy = realized_pnl.copy()
        realized_pnl_copy["datetime"] = pd.to_datetime(realized_pnl_copy["datetime"], errors='coerce')
        dates = sorted(realized_pnl_copy["datetime"].unique())
        # Ajouter la date actuelle
        dates.append(pd.Timestamp(datetime.now()))

    bh_evolution = []

    for date in dates:
        total_cost = 0
        total_value = 0

        # Pour chaque symbole dont le premier achat est avant cette date
        for symbol, first_buy_info in first_buys.items():
            if first_buy_info["date"] <= date:
                qty = first_buy_info["quantity"]
                price_buy = first_buy_info["price"]

                # Prix actuel (au moment de la date)
                price_now = price_cache.get_with_fallback(symbol)

                if price_now is not None:
                    cost = qty * price_buy
                    value = qty * price_now

                    total_cost += cost
                    total_value += value

        if total_cost > 0:
            pnl = total_value - total_cost
            bh_evolution.append({
                "datetime": date,
                "bh_cost": total_cost,
                "bh_value": total_value,
                "bh_pnl": pnl
            })

    if not bh_evolution:
        return pd.DataFrame()

    return pd.DataFrame(bh_evolution)


# =============================
# Buy & Hold Comparison (CORRIGÉ)
# =============================
def calculate_buy_and_hold(trades_df: pd.DataFrame) -> dict:
    """
    Calcule la performance Buy & Hold basée sur le PREMIER achat de chaque symbole.
    Simulation: "Si j'avais acheté lors du premier trade de chaque crypto et JAMAIS vendu"
    """
    if trades_df.empty:
        return {"total_cost": 0, "total_value": 0, "total_pnl": 0, "total_pct": 0, "per_symbol": {}}

    # Récupérer TOUS les achats
    all_buys = trades_df[trades_df["action"] == "buy"].copy()

    if all_buys.empty:
        return {"total_cost": 0, "total_value": 0, "total_pnl": 0, "total_pct": 0, "per_symbol": {}}

    total_cost = 0
    total_value = 0
    per_symbol = {}

    # Pour chaque symbole, prendre uniquement le PREMIER achat
    for symbol in all_buys["symbol"].unique():
        symbol_buys = all_buys[all_buys["symbol"] == symbol].sort_values("datetime")

        # PREMIER achat uniquement
        first_buy = symbol_buys.iloc[0]
        first_qty = first_buy["quantity"]
        first_price = first_buy["price"]
        first_cost = first_qty * first_price

        # Prix actuel
        current_price = price_cache.get_with_fallback(symbol)

        if current_price is not None and first_cost > 0:
            symbol_value = first_qty * current_price
            symbol_pnl = symbol_value - first_cost
            symbol_pct = (symbol_pnl / first_cost * 100)

            total_cost += first_cost
            total_value += symbol_value

            per_symbol[symbol] = {
                "cost": float(first_cost),
                "value": float(symbol_value),
                "pnl": float(symbol_pnl),
                "pct": float(symbol_pct),
                "quantity": float(first_qty),
                "current_price": float(current_price),
                "first_buy_date": first_buy["datetime"]
            }

    total_pnl = total_value - total_cost
    total_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0

    return {
        "total_cost": float(total_cost),
        "total_value": float(total_value),
        "total_pnl": float(total_pnl),
        "total_pct": float(total_pct),
        "per_symbol": per_symbol
    }


# =============================
# NOUVEAU : Tableau de Bord Risque
# =============================
def calculate_risk_metrics(open_positions_enriched: pd.DataFrame, initial_capital: float) -> dict:
    """Calcule les métriques de risque du portfolio"""
    if open_positions_enriched.empty:
        return {
            "total_exposure": 0,
            "capital_invested": 0,
            "capital_available": initial_capital,
            "largest_position_pct": 0,
            "largest_position_symbol": None,
            "num_positions": 0,
            "concentration_index": 0,
            "allocation": {}
        }

    # Agréger par symbole
    summary = open_positions_enriched.groupby("symbol").agg({
        "value_$": "sum",
        "cost_$": "sum"
    }).reset_index()

    total_value = summary["value_$"].sum()
    total_cost = summary["cost_$"].sum()

    # Plus grosse position
    largest = summary.loc[summary["value_$"].idxmax()]
    largest_pct = (largest["value_$"] / total_value * 100) if total_value > 0 else 0

    # Allocation par crypto
    allocation = {}
    for _, row in summary.iterrows():
        pct = (row["value_$"] / total_value * 100) if total_value > 0 else 0
        allocation[row["symbol"]] = {
            "value": float(row["value_$"]),
            "percentage": float(pct)
        }

    # Index de concentration Herfindahl (0-10000, plus c'est élevé plus c'est concentré)
    percentages = [alloc["percentage"] for alloc in allocation.values()]
    herfindahl = sum(p ** 2 for p in percentages)

    # Capital disponible (simplifié)
    num_symbols = len(summary)
    capital_available = initial_capital * num_symbols - total_cost

    return {
        "total_exposure": float(total_value),
        "capital_invested": float(total_cost),
        "capital_available": float(capital_available),
        "largest_position_pct": float(largest_pct),
        "largest_position_symbol": largest["symbol"],
        "num_positions": len(summary),
        "concentration_index": float(herfindahl),
        "allocation": allocation
    }


# =============================
# NOUVEAU : Waterfall Chart Data
# =============================
def prepare_waterfall_data(realized_pnl: pd.DataFrame, open_positions_enriched: pd.DataFrame) -> dict:
    """Prépare les données pour le waterfall chart"""
    data = {"symbols": [], "realized": [], "latent": [], "total": []}

    # PnL réalisé par symbole
    if not realized_pnl.empty:
        realized_by_symbol = realized_pnl.groupby("symbol")["pnl_$"].sum()
    else:
        realized_by_symbol = pd.Series(dtype=float)

    # PnL latent par symbole
    if not open_positions_enriched.empty and open_positions_enriched["pnl_live_$"].notna().any():
        latent_by_symbol = open_positions_enriched.groupby("symbol")["pnl_live_$"].sum()
    else:
        latent_by_symbol = pd.Series(dtype=float)

    # Combiner tous les symboles
    all_symbols = set(realized_by_symbol.index) | set(latent_by_symbol.index)

    for symbol in sorted(all_symbols):
        realized = realized_by_symbol.get(symbol, 0)
        latent = latent_by_symbol.get(symbol, 0)
        total = realized + latent

        data["symbols"].append(symbol)
        data["realized"].append(float(realized))
        data["latent"].append(float(latent))
        data["total"].append(float(total))

    return data


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

    # Charger tous les prix en une seule fois
    fetch_all_prices_coingecko()

    open_positions_enriched = enrich_with_live_prices(open_positions)
    trading_stats = calculate_trading_stats(realized_pnl)
    buy_hold_stats = calculate_buy_and_hold(df)
    bh_evolution = calculate_bh_evolution(df, realized_pnl)

    # NOUVEAU : Métriques de risque
    risk_metrics = calculate_risk_metrics(open_positions_enriched, initial_capital)

    # NOUVEAU : Données waterfall
    waterfall_data = prepare_waterfall_data(realized_pnl, open_positions_enriched)

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
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Vue Globale", "💹 Positions Ouvertes", "🔒 Positions Fermées", "📈 Par Crypto", "📉 Statistiques"])

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

    # =============================
    # NOUVEAU : WATERFALL CHART
    # =============================
    st.subheader("💧 Contribution par Crypto au PnL Total")

    if waterfall_data["symbols"]:
        fig_waterfall = go.Figure()

        # Créer les barres empilées
        fig_waterfall.add_trace(go.Bar(
            name="PnL Réalisé",
            x=waterfall_data["symbols"],
            y=waterfall_data["realized"],
            marker_color=['#10b981' if x > 0 else '#ef4444' for x in waterfall_data["realized"]]
        ))

        fig_waterfall.add_trace(go.Bar(
            name="PnL Latent",
            x=waterfall_data["symbols"],
            y=waterfall_data["latent"],
            marker_color=['#3b82f6' if x > 0 else '#f59e0b' for x in waterfall_data["latent"]]
        ))

        fig_waterfall.update_layout(
            barmode='relative',
            xaxis_title="",
            yaxis_title="PnL ($)",
            hovermode='x unified',
            height=400,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        fig_waterfall.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)

        st.plotly_chart(fig_waterfall, use_container_width=True)

        # Tableau récapitulatif
        waterfall_df = pd.DataFrame({
            "Crypto": waterfall_data["symbols"],
            "PnL Réalisé ($)": [format_currency(x) for x in waterfall_data["realized"]],
            "PnL Latent ($)": [format_currency(x) for x in waterfall_data["latent"]],
            "PnL Total ($)": [format_currency(x) for x in waterfall_data["total"]]
        })

        with st.expander("📋 Voir le détail des contributions"):
            st.dataframe(waterfall_df, use_container_width=True, hide_index=True)
    else:
        st.info("Aucune donnée disponible pour le waterfall chart")

    st.markdown("---")

    # Graphique d'évolution du PnL - AVEC BUY & HOLD
    if not realized_pnl.empty:
        st.subheader("📈 Comparaison Trading vs Buy & Hold")

        fig = make_subplots(specs=[[{"secondary_y": False}]])

        # PnL cumulé réalisé (Trading actif)
        fig.add_trace(
            go.Scatter(
                x=realized_pnl["datetime"],
                y=realized_pnl["global_cum_pnl_$"],
                mode="lines",
                name="Trading Actif",
                line=dict(color="#3b82f6", width=3),
                fill='tozeroy',
                fillcolor='rgba(59, 130, 246, 0.1)'
            )
        )

        # Point actuel trading (réalisé + latent)
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
                    name="Trading (projection)",
                    line=dict(color="#3b82f6", width=2, dash="dash"),
                    showlegend=False
                )
            )

            # Point actuel trading
            fig.add_trace(
                go.Scatter(
                    x=[datetime.now()],
                    y=[current_total],
                    mode="markers+text",
                    name="Trading Maintenant",
                    marker=dict(size=15, color="#3b82f6", symbol="diamond"),
                    text=["Trading"],
                    textposition="top center",
                )
            )

        # Évolution Buy & Hold
        if not bh_evolution.empty:
            fig.add_trace(
                go.Scatter(
                    x=bh_evolution["datetime"],
                    y=bh_evolution["bh_pnl"],
                    mode="lines",
                    name="Buy & Hold",
                    line=dict(color="#f59e0b", width=3, dash="dot"),
                )
            )

            # Point actuel B&H
            current_bh = bh_evolution.iloc[-1]["bh_pnl"]
            fig.add_trace(
                go.Scatter(
                    x=[bh_evolution.iloc[-1]["datetime"]],
                    y=[current_bh],
                    mode="markers+text",
                    name="B&H Maintenant",
                    marker=dict(size=15, color="#f59e0b", symbol="diamond"),
                    text=["B&H"],
                    textposition="bottom center",
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

        # Texte explicatif
        st.caption("""
        📌 **Légende:**
        - 🔵 **Trading Actif**: Votre stratégie avec achats/ventes et réinvestissement
        - 🟡 **Buy & Hold**: Si vous aviez acheté lors du PREMIER achat de chaque crypto et JAMAIS vendu
        """)

    # Comparaison avec Buy & Hold (CORRIGÉ AVEC ROI SUR CAPITAL INITIAL)
    st.markdown("---")
    st.subheader("🔄 Comparaison avec Buy & Hold")
    st.caption(
        "Comparaison entre votre stratégie de trading et une stratégie Buy & Hold (acheter au premier trade de chaque crypto et garder)")

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
        # CORRIGÉ : ROI sur capital initial
        # Nombre de symboles tradés
        num_symbols_traded = len(df["symbol"].unique())
        total_initial_capital = initial_capital * num_symbols_traded

        bh_roi = buy_hold_stats["total_pct"]
        trading_roi = (trading_pnl / total_initial_capital * 100) if total_initial_capital > 0 else 0

        roi_diff = trading_roi - bh_roi
        roi_type = "positive" if roi_diff > 0 else "negative" if roi_diff < 0 else "neutral"

        display_metric_card(
            "Différence ROI",
            format_percentage(roi_diff),
            f"Trading: {format_percentage(trading_roi)} | B&H: {format_percentage(bh_roi)}",
            card_type=roi_type
        )

    # =============================
    # NOUVEAU : TABLEAU DE BORD RISQUE
    # =============================
    st.markdown("---")
    st.subheader("⚠️ Tableau de Bord Risque Actuel")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        display_metric_card(
            "Exposition Totale",
            format_currency(risk_metrics["total_exposure"]),
            f"Capital investi: {format_currency(risk_metrics['capital_invested'])}",
            card_type="info"
        )

    with col2:
        display_metric_card(
            "Nombre de Positions",
            str(risk_metrics["num_positions"]),
            f"Cryptos différentes",
            card_type="info"
        )

    with col3:
        if risk_metrics["largest_position_symbol"]:
            display_metric_card(
                "Plus Grosse Position",
                f"{risk_metrics['largest_position_pct']:.1f}%",
                f"{risk_metrics['largest_position_symbol']}",
                card_type="info"
            )
        else:
            display_metric_card(
                "Plus Grosse Position",
                "N/A",
                card_type="neutral"
            )

    with col4:
        # Interprétation de l'index de concentration
        hhi = risk_metrics["concentration_index"]
        if hhi < 1500:
            concentration_label = "Bien diversifié"
            conc_type = "positive"
        elif hhi < 2500:
            concentration_label = "Diversification moyenne"
            conc_type = "neutral"
        else:
            concentration_label = "Très concentré"
            conc_type = "negative"

        display_metric_card(
            "Index de Concentration",
            f"{hhi:.0f}",
            concentration_label,
            card_type=conc_type
        )

    # Graphique d'allocation
    if risk_metrics["allocation"]:
        st.markdown("### 📊 Allocation Actuelle du Portfolio")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Pie chart
            symbols = list(risk_metrics["allocation"].keys())
            values = [risk_metrics["allocation"][s]["value"] for s in symbols]
            percentages = [risk_metrics["allocation"][s]["percentage"] for s in symbols]

            fig_allocation = go.Figure(data=[go.Pie(
                labels=symbols,
                values=values,
                textposition='inside',
                textinfo='percent',  # Affiche seulement le %
                hovertemplate='<b>%{label}</b><br>Valeur: $%{value:,.0f}<br>Part: %{percent}<extra></extra>',
                hole=0.4,
                marker=dict(
                    colors=['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#14b8a6', '#f97316'])
            )])

            fig_allocation.update_layout(
                title="",
                height=400,
                template="plotly_white",
                showlegend=True
            )

            st.plotly_chart(fig_allocation, use_container_width=True)

        with col2:
            st.markdown("#### Répartition")
            allocation_df = pd.DataFrame([
                {
                    "Crypto": symbol,
                    "Valeur": format_currency(data["value"]),
                    "% Portfolio": f"{data['percentage']:.1f}%"
                }
                for symbol, data in risk_metrics["allocation"].items()
            ]).sort_values("% Portfolio", ascending=False, key=lambda x: x.str.rstrip('%').astype(float))

            st.dataframe(allocation_df, use_container_width=True, hide_index=True)

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
# TAB 3: POSITIONS FERMÉES
# =============================
with tab3:
    st.subheader("🔒 Positions Fermées (Trades Réalisés)")

    if not realized_pnl.empty:
        # Statistiques globales des positions fermées
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_trades = len(realized_pnl)
            display_metric_card(
                "Nombre de Trades",
                str(total_trades),
                card_type="info"
            )

        with col2:
            total_pnl_closed = float(realized_pnl["pnl_$"].sum())
            pnl_type = "positive" if total_pnl_closed > 0 else "negative" if total_pnl_closed < 0 else "neutral"
            display_metric_card(
                "PnL Total Fermé",
                format_currency(total_pnl_closed),
                card_type=pnl_type
            )

        with col3:
            winning = len(realized_pnl[realized_pnl["pnl_$"] > 0])
            win_rate = (winning / total_trades * 100) if total_trades > 0 else 0
            wr_type = "positive" if win_rate >= 50 else "negative"
            display_metric_card(
                "Win Rate",
                f"{win_rate:.1f}%",
                f"{winning} gagnants",
                card_type=wr_type
            )

        with col4:
            avg_pnl = float(realized_pnl["pnl_$"].mean())
            avg_type = "positive" if avg_pnl > 0 else "negative" if avg_pnl < 0 else "neutral"
            display_metric_card(
                "PnL Moyen/Trade",
                format_currency(avg_pnl),
                card_type=avg_type
            )

        st.markdown("---")

        # Graphique: Évolution cumulative des positions fermées
        st.subheader("📈 Évolution des Positions Fermées")

        fig_closed = go.Figure()

        fig_closed.add_trace(go.Scatter(
            x=realized_pnl["datetime"],
            y=realized_pnl["global_cum_pnl_$"],
            mode="lines+markers",
            name="PnL Cumulé",
            line=dict(color="#10b981", width=2),
            marker=dict(size=6),
            fill='tozeroy',
            fillcolor='rgba(16, 185, 129, 0.1)'
        ))

        fig_closed.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)

        fig_closed.update_layout(
            xaxis_title="Date",
            yaxis_title="PnL Cumulé ($)",
            hovermode='x unified',
            height=400,
            template="plotly_white"
        )

        st.plotly_chart(fig_closed, use_container_width=True)

        st.markdown("---")

        # Tableau détaillé des positions fermées
        st.subheader("📋 Historique Complet des Trades")

        # Préparer les données pour l'affichage
        display_realized = realized_pnl.copy()

        # Identifier les stop loss
        display_realized["is_stop"] = display_realized["confiance"].apply(
            lambda x: True if pd.notna(x) and float(x) == 0.0 else False
        )

        # Sélectionner et renommer les colonnes
        display_cols = display_realized[
            ["datetime", "symbol", "quantity", "price_buy", "price_sell", "cost_$", "pnl_$", "pnl_%", "is_stop",
             "buy_date"]
        ].copy()

        display_cols.columns = ["Date Vente", "Symbole", "Quantité", "Prix Achat", "Prix Vente",
                                "Coût ($)", "PnL ($)", "PnL (%)", "Stop Loss", "Date Achat"]

        # Formatage
        display_cols["Quantité"] = display_cols["Quantité"].apply(lambda x: f"{x:.6f}".rstrip('0').rstrip('.'))
        for col in ["Prix Achat", "Prix Vente", "Coût ($)", "PnL ($)"]:
            display_cols[col] = display_cols[col].apply(lambda x: f"${x:,.2f}")
        display_cols["PnL (%)"] = display_cols["PnL (%)"].apply(lambda x: f"{x:+.2f}%")
        display_cols["Stop Loss"] = display_cols["Stop Loss"].apply(lambda x: "✓" if x else "")

        # Réorganiser les colonnes
        final_cols = ["Date Vente", "Symbole", "Quantité", "Date Achat", "Prix Achat",
                      "Prix Vente", "Coût ($)", "PnL ($)", "PnL (%)", "Stop Loss"]
        display_cols = display_cols[final_cols]

        # Filtres interactifs
        col_filter1, col_filter2 = st.columns(2)

        with col_filter1:
            symbols_filter = st.multiselect(
                "Filtrer par symbole",
                options=sorted(realized_pnl["symbol"].unique()),
                default=[]
            )

        with col_filter2:
            result_filter = st.selectbox(
                "Filtrer par résultat",
                options=["Tous", "Gagnants uniquement", "Perdants uniquement", "Stop Loss uniquement"]
            )

        # Appliquer les filtres
        filtered_realized = realized_pnl.copy()

        if symbols_filter:
            filtered_realized = filtered_realized[filtered_realized["symbol"].isin(symbols_filter)]

        if result_filter == "Gagnants uniquement":
            filtered_realized = filtered_realized[filtered_realized["pnl_$"] > 0]
        elif result_filter == "Perdants uniquement":
            filtered_realized = filtered_realized[filtered_realized["pnl_$"] < 0]
        elif result_filter == "Stop Loss uniquement":
            filtered_realized["is_stop"] = filtered_realized["confiance"].apply(
                lambda x: True if pd.notna(x) and float(x) == 0.0 else False
            )
            filtered_realized = filtered_realized[filtered_realized["is_stop"] == True]

        # Refaire le formatage pour les données filtrées
        if not filtered_realized.empty:
            filtered_display = filtered_realized.copy()
            filtered_display["is_stop"] = filtered_display["confiance"].apply(
                lambda x: True if pd.notna(x) and float(x) == 0.0 else False
            )

            filtered_display_cols = filtered_display[
                ["datetime", "symbol", "quantity", "price_buy", "price_sell", "cost_$", "pnl_$", "pnl_%", "is_stop",
                 "buy_date"]
            ].copy()

            filtered_display_cols.columns = ["Date Vente", "Symbole", "Quantité", "Prix Achat", "Prix Vente",
                                             "Coût ($)", "PnL ($)", "PnL (%)", "Stop Loss", "Date Achat"]

            filtered_display_cols["Quantité"] = filtered_display_cols["Quantité"].apply(
                lambda x: f"{x:.6f}".rstrip('0').rstrip('.'))
            for col in ["Prix Achat", "Prix Vente", "Coût ($)", "PnL ($)"]:
                filtered_display_cols[col] = filtered_display_cols[col].apply(lambda x: f"${x:,.2f}")
            filtered_display_cols["PnL (%)"] = filtered_display_cols["PnL (%)"].apply(lambda x: f"{x:+.2f}%")
            filtered_display_cols["Stop Loss"] = filtered_display_cols["Stop Loss"].apply(lambda x: "✓" if x else "")

            filtered_display_cols = filtered_display_cols[final_cols]

            st.dataframe(filtered_display_cols, use_container_width=True, hide_index=True)

            st.caption(f"**{len(filtered_display_cols)}** trades affichés")
        else:
            st.info("Aucun trade ne correspond aux filtres sélectionnés.")

        # Répartition par crypto
        st.markdown("---")
        st.subheader("📊 Répartition des Trades par Crypto")

        col1, col2 = st.columns(2)

        with col1:
            # Nombre de trades par crypto
            trades_per_symbol = realized_pnl.groupby("symbol").size().reset_index(name="Nombre")
            trades_per_symbol = trades_per_symbol.sort_values("Nombre", ascending=False)

            fig_trades = go.Figure(data=[go.Bar(
                x=trades_per_symbol["symbol"],
                y=trades_per_symbol["Nombre"],
                marker_color='#3b82f6'
            )])
            fig_trades.update_layout(
                title="Nombre de Trades par Crypto",
                xaxis_title="",
                yaxis_title="Nombre",
                height=350,
                template="plotly_white"
            )
            st.plotly_chart(fig_trades, use_container_width=True)

        with col2:
            # PnL total par crypto
            pnl_per_symbol = realized_pnl.groupby("symbol")["pnl_$"].sum().reset_index()
            pnl_per_symbol = pnl_per_symbol.sort_values("pnl_$", ascending=False)

            colors = ['#10b981' if x > 0 else '#ef4444' for x in pnl_per_symbol["pnl_$"]]

            fig_pnl = go.Figure(data=[go.Bar(
                x=pnl_per_symbol["symbol"],
                y=pnl_per_symbol["pnl_$"],
                marker_color=colors
            )])
            fig_pnl.update_layout(
                title="PnL Total par Crypto",
                xaxis_title="",
                yaxis_title="PnL ($)",
                height=350,
                template="plotly_white"
            )
            fig_pnl.add_hline(y=0, line_dash="dot", line_color="gray")
            st.plotly_chart(fig_pnl, use_container_width=True)

    else:
        st.info("Aucune position fermée pour le moment.")

# =============================
# TAB 4: VUE PAR CRYPTO (AMÉLIORÉ AVEC B&H ET ROI CORRIGÉ)
# =============================
with tab4:
    st.subheader("🔍 Analyse par Crypto")

    if not df.empty:
        symbols = sorted(df["symbol"].unique())
        selected_symbol = st.selectbox("Sélectionner une crypto", symbols)

        # Données du symbole sélectionné
        symbol_realized = realized_pnl[
            realized_pnl["symbol"] == selected_symbol].copy() if not realized_pnl.empty else pd.DataFrame()
        symbol_open = open_positions_enriched[open_positions_enriched[
                                                  "symbol"] == selected_symbol].copy() if not open_positions_enriched.empty else pd.DataFrame()

        # =============================
        # NOUVEAU : Comparaison avec B&H pour cette crypto (ROI SUR CAPITAL INITIAL)
        # =============================
        st.markdown("---")
        st.subheader(f"🎯 Performance Trading vs Buy & Hold - {selected_symbol}")

        # Calculer le PnL trading (réalisé + latent) pour cette crypto
        symbol_realized_pnl = float(symbol_realized["pnl_$"].sum()) if not symbol_realized.empty else 0
        symbol_latent_pnl = float(symbol_open["pnl_live_$"].sum()) if not symbol_open.empty and symbol_open[
            "pnl_live_$"].notna().any() else 0
        symbol_trading_pnl = symbol_realized_pnl + symbol_latent_pnl

        # CORRIGÉ : Calculer le VRAI B&H - Premier achat et on garde jusqu'à maintenant
        symbol_trades = df[df["symbol"] == selected_symbol].copy()
        symbol_buys = symbol_trades[symbol_trades["action"] == "buy"].sort_values("datetime")

        has_bh_data = False
        symbol_bh_pnl = 0
        symbol_bh_cost = 0
        symbol_bh_value = 0
        symbol_bh_pct = 0
        first_buy_date = None
        first_buy_qty = 0
        first_buy_price = 0

        if not symbol_buys.empty:
            # Prendre le PREMIER achat (le plus ancien)
            first_buy = symbol_buys.iloc[0]
            first_buy_date = first_buy["datetime"]
            first_buy_qty = float(first_buy["quantity"])
            first_buy_price = float(first_buy["price"])
            first_buy_cost = first_buy_qty * first_buy_price

            # Prix actuel
            current_price = price_cache.get_with_fallback(selected_symbol)

            if current_price is not None:
                # Valeur si on avait gardé depuis le premier achat
                symbol_bh_cost = first_buy_cost
                symbol_bh_value = first_buy_qty * current_price
                symbol_bh_pnl = symbol_bh_value - symbol_bh_cost
                symbol_bh_pct = (symbol_bh_pnl / symbol_bh_cost * 100) if symbol_bh_cost > 0 else 0
                has_bh_data = True

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            card_type = "positive" if symbol_trading_pnl > 0 else "negative" if symbol_trading_pnl < 0 else "neutral"

            # CORRIGÉ : ROI trading = PnL / Capital initial (qui tourne)
            symbol_trading_roi = (symbol_trading_pnl / initial_capital * 100) if initial_capital > 0 else 0

            display_metric_card(
                "PnL Trading",
                format_currency(symbol_trading_pnl),
                format_percentage(symbol_trading_roi),
                card_type=card_type
            )

        with col2:
            if has_bh_data:
                bh_type = "positive" if symbol_bh_pnl > 0 else "negative" if symbol_bh_pnl < 0 else "neutral"
                display_metric_card(
                    "PnL Buy & Hold",
                    format_currency(symbol_bh_pnl),
                    format_percentage(symbol_bh_pct),
                    card_type=bh_type
                )
            else:
                display_metric_card("PnL Buy & Hold", "N/A", "Aucun achat historique", card_type="neutral")

        with col3:
            if has_bh_data:
                diff = symbol_trading_pnl - symbol_bh_pnl
                diff_type = "positive" if diff > 0 else "negative" if diff < 0 else "neutral"
                display_metric_card(
                    "Différence ($)",
                    format_currency(diff),
                    "Meilleur" if diff > 0 else "Moins bon" if diff < 0 else "Égal",
                    card_type=diff_type
                )
            else:
                display_metric_card("Différence ($)", "N/A", card_type="neutral")

        with col4:
            if has_bh_data:
                roi_diff = symbol_trading_roi - symbol_bh_pct
                roi_type = "positive" if roi_diff > 0 else "negative" if roi_diff < 0 else "neutral"
                display_metric_card(
                    "Différence ROI",
                    format_percentage(roi_diff),
                    card_type=roi_type
                )
            else:
                display_metric_card("Différence ROI", "N/A", card_type="neutral")

        # Graphique de comparaison
        if has_bh_data:
            st.markdown("#### 📊 Comparaison Visuelle")

            fig_compare = go.Figure()

            categories = ['Trading (Réalisé + Latent)', 'Buy & Hold (Depuis 1er achat)']
            pnl_values = [symbol_trading_pnl, symbol_bh_pnl]
            colors = ['#3b82f6', '#f59e0b']

            fig_compare.add_trace(go.Bar(
                x=categories,
                y=pnl_values,
                marker_color=colors,
                text=[format_currency(v) for v in pnl_values],
                textposition='outside'
            ))

            fig_compare.update_layout(
                title=f"PnL Comparé - {selected_symbol}",
                yaxis_title="PnL ($)",
                height=350,
                template="plotly_white",
                showlegend=False
            )

            fig_compare.add_hline(y=0, line_dash="dot", line_color="gray")

            st.plotly_chart(fig_compare, use_container_width=True)

            # Tableau détaillé
            with st.expander("🔍 Détail des calculs"):
                st.markdown(f"""
                **Trading (Réalisé + Latent) :**
                - Capital initial : {format_currency(initial_capital)}
                - PnL Réalisé : {format_currency(symbol_realized_pnl)}
                - PnL Latent : {format_currency(symbol_latent_pnl)}
                - **Total PnL : {format_currency(symbol_trading_pnl)}** ({format_percentage(symbol_trading_roi)} ROI)

                **Buy & Hold (Premier achat uniquement) :**
                - Date du premier achat : {first_buy_date.strftime('%Y-%m-%d %H:%M:%S') if first_buy_date else 'N/A'}
                - Capital initial : {format_currency(initial_capital)}
                - Quantité achetée : {first_buy_qty:.6f} {selected_symbol.replace('USDT', '').replace('USDC', '')}
                - Prix d'achat : {format_currency(first_buy_price)}
                - Coût initial : {format_currency(symbol_bh_cost)}
                - Prix actuel : {format_currency(current_price) if 'current_price' in locals() and current_price else 'N/A'}
                - Valeur actuelle : {format_currency(symbol_bh_value)}
                - **PnL B&H : {format_currency(symbol_bh_pnl)}** ({format_percentage(symbol_bh_pct)} ROI)

                ℹ️ *Le B&H simule : "J'achète lors du premier trade avec {format_currency(initial_capital)} et je ne vends JAMAIS"*
                ℹ️ *Le Trading réinvestit le même capital initial ({format_currency(initial_capital)}) à chaque cycle*
                """)

            # Analyse textuelle
            if symbol_trading_pnl > symbol_bh_pnl:
                st.success(
                    f"🎉 **Votre stratégie de trading performe mieux que Buy & Hold sur {selected_symbol}** avec une différence de {format_currency(symbol_trading_pnl - symbol_bh_pnl)} ({format_percentage(roi_diff)} de ROI supplémentaire)")
            elif symbol_trading_pnl < symbol_bh_pnl:
                st.warning(
                    f"⚠️ **Buy & Hold aurait été plus performant sur {selected_symbol}** avec une différence de {format_currency(symbol_bh_pnl - symbol_trading_pnl)} ({format_percentage(-roi_diff)} de ROI en plus)")
            else:
                st.info(f"➡️ **Votre stratégie de trading est équivalente à Buy & Hold sur {selected_symbol}**")

        st.caption(
            "💡 **Note :** Le Buy & Hold compare comme si vous aviez acheté lors de votre tout premier trade sur cette crypto avec le même capital initial et que vous n'aviez JAMAIS vendu depuis.")

        st.markdown("---")

        # Métriques du symbole
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if not symbol_realized.empty:
                pnl = float(symbol_realized["pnl_$"].sum())
                card_type = "positive" if pnl > 0 else "negative" if pnl < 0 else "neutral"
                display_metric_card(
                    f"PnL Réalisé",
                    format_currency(pnl),
                    card_type=card_type
                )
            else:
                display_metric_card(f"PnL Réalisé", "$0.00", card_type="neutral")

        with col2:
            if not symbol_open.empty and symbol_open["pnl_live_$"].notna().any():
                latent = float(symbol_open["pnl_live_$"].sum())
                card_type = "positive" if latent > 0 else "negative" if latent < 0 else "neutral"
                display_metric_card(
                    f"PnL Latent",
                    format_currency(latent),
                    card_type=card_type
                )
            else:
                display_metric_card(f"PnL Latent", "$0.00", card_type="neutral")

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
            if not symbol_open.empty and symbol_open["value_$"].notna().any():
                value = float(symbol_open["value_$"].sum())
                display_metric_card(
                    "Valeur Position",
                    format_currency(value),
                    card_type="info"
                )
            else:
                display_metric_card("Valeur Position", "$0.00", card_type="neutral")

        st.markdown("---")

        # Graphiques
        if not symbol_realized.empty:
            # Identifier les sorties par stop loss
            symbol_realized["is_stop"] = symbol_realized["confiance"].apply(
                lambda x: True if pd.notna(x) and float(x) == 0.0 else False
            )

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
# TAB 5: STATISTIQUES
# =============================
with tab5:
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
    Dashboard Crypto Portfolio v3.2 | ROI calculé sur le capital initial (même $1000 qui tourne)
</div>
""", unsafe_allow_html=True)