import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import re
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# ==========================================
# 1. CORE LOGIC CLASSES (Backend)
# ==========================================

# --- A. Duplicate Detector ---
class DuplicateTicketDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        if 'open_tickets' not in st.session_state:
            st.session_state.open_tickets = [
                {"id": 101, "text": "Internet connection is down wifi not working", "user": "Ali", "status": "Open"},
                {"id": 102, "text": "Printer in HR office is jamming paper", "user": "Mona", "status": "Open"},
                {"id": 103, "text": "Cannot login to ERP system wrong password", "user": "Hassan", "status": "Open"}
            ]
        self.GLOBAL_KEYWORDS = ['wifi', 'internet', 'network', 'server', 'power', 'system']

    def _is_global_issue(self, text):
        for kw in self.GLOBAL_KEYWORDS:
            if kw in text.lower(): return True
        return False

    def check_is_duplicate(self, new_text, new_user):
        if not st.session_state.open_tickets: return False, None, 0.0
        
        existing_texts = [t['text'] for t in st.session_state.open_tickets]
        corpus = existing_texts + [new_text]
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        
        sims = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
        best_idx = np.argmax(sims)
        best_score = sims[best_idx]
        
        match_ticket = st.session_state.open_tickets[best_idx]
        
        if best_score > 0.5:
            if match_ticket['user'] == new_user:
                return True, match_ticket, best_score 
            elif self._is_global_issue(new_text):
                return True, match_ticket, best_score
        
        return False, None, best_score

    def add_ticket(self, text, user):
        new_id = st.session_state.open_tickets[-1]['id'] + 1 if st.session_state.open_tickets else 101
        st.session_state.open_tickets.append({"id": new_id, "text": text, "user": user, "status": "Open"})
        return new_id

# --- B. Task Dispatcher ---
DATA_FILE = "technicians_state.csv"

class SmartDispatcher:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.df = self._load_state()

    def _load_state(self):
        if os.path.exists(DATA_FILE):
            df = pd.read_csv(DATA_FILE)
            for col in ['keyword_weights', 'is_present', 'total_finished_tickets', 'solved_history_text']:
                if col not in df.columns: 
                    df[col] = "{}" if 'weights' in col else (True if 'present' in col else (0 if 'total' in col else ""))
            df['solved_history_text'] = df['solved_history_text'].fillna("")
            return df
        else:
            return pd.DataFrame([
                {"id": 1, "name": "Ahmed", "current_floor": 1, "active_tickets": 0, "is_present": True, "solved_history_text": "wifi internet router connection", "keyword_weights": "{}"},
                {"id": 2, "name": "Sara", "current_floor": 3, "active_tickets": 0, "is_present": True, "solved_history_text": "password login excel windows office", "keyword_weights": "{}"},
                {"id": 3, "name": "Khaled", "current_floor": 5, "active_tickets": 0, "is_present": True, "solved_history_text": "printer paper jam toner hardware screen", "keyword_weights": "{}"},
                {"id": 4, "name": "Mona", "current_floor": 2, "active_tickets": 1, "is_present": True, "solved_history_text": "design css html ui", "keyword_weights": "{}"},
                {"id": 5, "name": "Omar", "current_floor": 4, "active_tickets": 0, "is_present": True, "solved_history_text": "server linux cloud aws", "keyword_weights": "{}"},
                {"id": 6, "name": "Hassan", "current_floor": 1, "active_tickets": 2, "is_present": True, "solved_history_text": "network cabling lan port", "keyword_weights": "{}"},
                {"id": 7, "name": "Laila", "current_floor": 6, "active_tickets": 0, "is_present": False, "solved_history_text": "security firewall vpn", "keyword_weights": "{}"},
            ])

    def save(self):
        self.df.to_csv(DATA_FILE, index=False)

    def assign(self, text, floor):
        try:
            corpus = self.df['solved_history_text'].tolist() + [text]
            matrix = self.vectorizer.fit_transform(corpus)
            sim_scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
        except:
            sim_scores = np.zeros(len(self.df))

        dists = abs(self.df['current_floor'] - floor)
        prox_scores = 1 - (dists / 10)
        
        final = (sim_scores * 0.6) + (prox_scores * 0.4) - (self.df['active_tickets'] * 0.2)
        final[self.df['is_present'] == False] = -99
        
        best_idx = np.argmax(final)
        winner = self.df.iloc[best_idx]
        
        self.df.at[best_idx, 'active_tickets'] += 1
        self.save()
        return winner['name'], final[best_idx]

    def train(self, name, text):
        if name in self.df['name'].values:
            idx = self.df[self.df['name'] == name].index[0]
            curr = str(self.df.at[idx, 'solved_history_text'])
            self.df.at[idx, 'solved_history_text'] = curr + " " + text
            if self.df.at[idx, 'active_tickets'] > 0:
                self.df.at[idx, 'active_tickets'] -= 1
            self.save()

# --- C. Stock Forecaster (Updated with more products) ---
class StockAI:
    def __init__(self):
        # Catalog of products with their simulation parameters
        # type: Pattern type (Stable, Volatile, Fast, Slow)
        # start: Initial quantity
        # rate: Daily consumption rate
        # noise: Random fluctuation magnitude
        self.catalog = {
            "HP LaserJet Toner": {"type": "Stable", "start": 50, "rate": 0.5, "noise": 0.1},
            "A4 Paper (Box)": {"type": "Fast", "start": 200, "rate": 3.0, "noise": 2.0},
            "Wireless Mouse": {"type": "Volatile", "start": 40, "rate": 0.2, "noise": 1.0},
            "HDMI Cable (2m)": {"type": "Volatile", "start": 60, "rate": 0.1, "noise": 0.5},
            "Ethernet Cable (Cat6)": {"type": "Stable", "start": 100, "rate": 0.8, "noise": 0.2},
            "USB-C Adapters": {"type": "Stable", "start": 30, "rate": 0.4, "noise": 0.3},
            "Screen Cleaning Kit": {"type": "Slow", "start": 25, "rate": 0.05, "noise": 0.01},
            "Laptop Stand": {"type": "Slow", "start": 15, "rate": 0.1, "noise": 0.1},
        }

    def get_product_list(self):
        return list(self.catalog.keys())

    def generate_dummy_data(self, item_name):
        props = self.catalog.get(item_name, {"type": "Stable", "start": 100, "rate": 1.0, "noise": 1.0})
        
        # Simulate last 90 days
        dates = pd.date_range(end=datetime.date.today(), periods=90)
        
        # Determine trend based on type
        qty = []
        current = props['start']
        
        for _ in range(90):
            # Base consumption
            consumption = props['rate']
            
            # Add randomness
            noise = np.random.normal(0, props['noise'])
            
            # Apply consumption
            current = max(0, current - (consumption + noise))
            
            # Random restocking event (small probability) to make it realistic
            if np.random.rand() > 0.98: 
                current += props['start'] * 0.2 # Restock 20%
                
            qty.append(current)
            
        # Ensure we sort by date (though range is already sorted)
        df = pd.DataFrame({'date': dates, 'quantity': qty})
        return df

    def predict(self, df):
        df['day_num'] = np.arange(len(df))
        X = df[['day_num']]
        y = df['quantity']
        
        model = LinearRegression()
        model.fit(X, y)
        
        slope = model.coef_[0]
        intercept = model.intercept_
        
        # If slope is positive, stock is growing
        if slope >= 0: return None, None, slope
        
        # Predict when y = 0
        days_until_zero = -intercept / slope
        remaining = days_until_zero - df['day_num'].max()
        
        return int(remaining), model, slope

# ==========================================
# 2. STREAMLIT UI (Full English)
# ==========================================

st.set_page_config(page_title="AI Ops Dashboard", layout="wide", page_icon="🤖")

st.title("🤖 AI IT Operations Dashboard")
st.markdown("Intelligent Task Dispatching & Resource Forecasting")
st.divider()

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to:", ["🎫 Ticket Dispatcher", "📈 Stock Forecast"])
    st.markdown("---")
    st.caption("AI Engine v3.1 | Enterprise Edition")

# --- PAGE 1: Dispatcher ---
if page == "🎫 Ticket Dispatcher":
    st.subheader("🚀 Smart Ticket Assignment")
    col1, col2 = st.columns([2, 1], gap="medium")
    
    with col1:
        st.markdown("### 📝 New Ticket Entry")
        with st.container(border=True):
            ticket_text = st.text_area("Issue Description", placeholder="e.g., Internet connection lost on the 3rd floor...", height=100)
            c1, c2 = st.columns(2)
            user_name = c1.text_input("Reported By", "User 1")
            floor = c2.number_input("Floor Number", min_value=1, max_value=10, value=1)
            
            if st.button("Analyze & Assign ⚡", type="primary", use_container_width=True):
                if not ticket_text:
                    st.warning("Please enter a description.")
                else:
                    dup_sys = DuplicateTicketDetector()
                    is_dup, match, score = dup_sys.check_is_duplicate(ticket_text, user_name)
                    
                    if is_dup:
                        st.error(f"🚫 **Ticket Rejected!** (Duplicate Confidence: {score*100:.1f}%)")
                        st.warning(f"Similar to open ticket #{match['id']}: '{match['text']}' reported by {match['user']}")
                    else:
                        st.success("✅ Ticket Accepted. Processing assignment...")
                        dispatcher = SmartDispatcher()
                        winner_name, score = dispatcher.assign(ticket_text, floor)
                        st.balloons()
                        st.markdown(f"""
                        <div style="padding: 20px; background-color: #e8f5e9; border-radius: 10px; margin-top: 10px;">
                            <h3 style="color: #2e7d32; margin:0;">👤 Assigned to: {winner_name}</h3>
                            <p style="margin:0;"><strong>AI Confidence Score:</strong> {score:.2f}</p>
                            <p style="margin:0; font-size: 0.9em; color: #555;">Reasoning: Optimal match based on expertise history, proximity, and current workload.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        dup_sys.add_ticket(ticket_text, user_name)

    with col2:
        st.markdown("### 👨‍💻 Team Status")
        dispatcher = SmartDispatcher()
        display_df = dispatcher.df[['name', 'current_floor', 'active_tickets', 'is_present']].copy()
        display_df['Status'] = display_df.apply(lambda x: "🔴 Absent" if not x['is_present'] else ("🟠 Busy" if x['active_tickets'] > 1 else "🟢 Available"), axis=1)
        final_view = display_df[['name', 'current_floor', 'Status']]
        final_view.columns = ['Name', 'Floor', 'Status']
        st.dataframe(final_view, hide_index=True, use_container_width=True)
        st.divider()
        st.markdown("### 🎓 Training (Feedback)")
        with st.form("training_form"):
            tech_to_train = st.selectbox("Solved By?", dispatcher.df['name'])
            if st.form_submit_button("Mark as Solved & Train AI ✅", use_container_width=True):
                dispatcher.train(tech_to_train, "sample problem solved")
                st.toast(f"Knowledge Base updated for {tech_to_train}!", icon="🧠")
                st.rerun()

# --- PAGE 2: Stock AI ---
elif page == "📈 Stock Forecast":
    st.subheader("📦 Inventory Demand Forecasting")
    
    stock_ai = StockAI()
    product_list = stock_ai.get_product_list()
    
    col_config, col_chart = st.columns([1, 3])
    
    with col_config:
        st.markdown("### ⚙️ Settings")
        selected_item = st.selectbox("Select Inventory Item", product_list)
        st.info("AI analyzes consumption trends from the last 90 days to predict stock depletion.")
        
        # Display Item Metadata
        props = stock_ai.catalog[selected_item]
        st.caption(f"**Item Profile:** {props['type']} Consumption")

    # Generate Data & Predict
    df = stock_ai.generate_dummy_data(selected_item)
    days_left, model, slope = stock_ai.predict(df)

    with col_chart:
        # Metrics Row
        m1, m2, m3 = st.columns(3)
        current_stock = int(df['quantity'].iloc[-1])
        m1.metric("Current Stock", f"{current_stock} units")
        m2.metric("Burn Rate", f"{abs(slope):.2f} units/day")
        
        if days_left is not None:
            if days_left < 7:
                m3.metric("Depletion In", f"{days_left} Days", delta="-Critical", delta_color="inverse")
                st.error(f"🔴 **CRITICAL WARNING:** {selected_item} will run out in **{days_left} days**!")
            elif days_left < 30:
                m3.metric("Depletion In", f"{days_left} Days", delta="-Warning", delta_color="normal")
                st.warning(f"⚠️ **Low Stock Warning:** Plan restocking within **{days_left} days**.")
            else:
                m3.metric("Depletion In", f"{days_left} Days", delta="Safe")
                st.success(f"🟢 **Status Safe:** Sufficient stock for **{days_left} days**.")
        else:
            m3.metric("Trend", "Positive", delta="Growing")
            st.info("Stock level is increasing or stable. No immediate action required.")

        st.markdown("#### 📉 Consumption Trend & AI Forecast")
        
        # Advanced Charting
        st.line_chart(df.set_index('date')['quantity'], color="#FF4B4B" if days_left and days_left < 14 else "#4CAF50")
        
        with st.expander("View Raw Data"):
            st.dataframe(df.sort_values(by='date', ascending=False).head(10), use_container_width=True)
            