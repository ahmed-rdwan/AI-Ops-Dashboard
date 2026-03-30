import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import re
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression, Ridge
from sqlalchemy import create_engine, text
import urllib
import time
import google.generativeai as genai

# ==========================================
# 0. إعدادات الاتصال & API
# ==========================================
# 🔑 هام: ضع مفتاح Gemini API هنا
GOOGLE_API_KEY = "AIzaSyCJW2MbVag9kMaMhwpdZwPAqjyxgVSi5pc" # 🔴 استبدل هذا بمفتاحك الحقيقي
try:
    genai.configure(api_key=GOOGLE_API_KEY)
except: pass

SERVER_NAME = r"Ahmed-Radwan\SQLEXPRESS" 
DATABASE_NAME = "project_management"

params = urllib.parse.quote_plus(
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={SERVER_NAME};"
    f"DATABASE={DATABASE_NAME};"
    "Trusted_Connection=yes;"
)
DB_CONNECTION_STR = f"mssql+pyodbc:///?odbc_connect={params}"

db_engine = None
USE_MOCK_DATA = False

try:
    db_engine = create_engine(DB_CONNECTION_STR)
    with db_engine.connect() as conn: pass
    print("✅ Connected to Local SQL Server")
except Exception as e:
    USE_MOCK_DATA = True
    st.error(f"❌ Database Error: {e}")
    st.stop()

# ==========================================
# 1. AI Logic Core (Dispatcher & Time)
# ==========================================
class DuplicateTicketDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer() 
        self.GLOBAL_KEYWORDS = ['wifi', 'internet', 'network', 'server', 'power', 'system', 'database', 'نت', 'سيرفر', 'كهرباء', 'نظام']

    def _get_open_tickets(self):
        try: return pd.read_sql("SELECT id, name, description, created_by FROM ticket WHERE status = 'Open'", db_engine)
        except: return pd.DataFrame()

    def _is_global_issue(self, text):
        for kw in self.GLOBAL_KEYWORDS:
            if kw in text.lower(): return True
        return False

    def check_is_duplicate(self, new_text, new_user_id):
        df_open = self._get_open_tickets()
        if df_open.empty: return False, None, 0.0
        
        existing_texts = (df_open['name'] + " " + df_open['description']).fillna("").tolist()
        corpus = existing_texts + [new_text]
        
        try:
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
            sims = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
        except: return False, None, 0.0
        
        if len(sims) == 0: return False, None, 0.0
        best_idx = np.argmax(sims)
        best_score = sims[best_idx]
        match_row = df_open.iloc[best_idx]
        
        if best_score > 0.5:
            if match_row['created_by'] == new_user_id: return True, match_row, best_score 
            elif self._is_global_issue(new_text): return True, match_row, best_score
        return False, None, best_score

    def create_ticket_in_sql(self, title, desc, created_by, assigned_to):
        with db_engine.connect() as conn:
            query = text("INSERT INTO ticket (name, description, created_by, assign_to, priority, status) VALUES (:name, :desc, :creator, :assignee, 'Medium', 'Open')")
            conn.execute(query, {"name": title, "desc": desc, "creator": created_by, "assignee": assigned_to})
            conn.commit()

class SmartDispatcher:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.time_model = Ridge(alpha=1.0)
        self.is_time_model_trained = False
        self._ensure_profiles()
        self.df = self._load_data_from_sql()
        self._train_time_predictor()

    def _ensure_profiles(self):
        with db_engine.connect() as conn:
            conn.execute(text("INSERT INTO technician_profile (user_id) SELECT id FROM [user] WHERE id NOT IN (SELECT user_id FROM technician_profile)"))
            conn.commit()

    def _load_data_from_sql(self):
        query = """
        SELECT u.id, u.name, p.solved_history_text, p.keyword_weights, p.active_tickets, p.is_present, p.current_floor, p.total_finished_tickets, p.avg_resolution_time
        FROM [user] u JOIN technician_profile p ON u.id = p.user_id
        """
        df = pd.read_sql(query, db_engine)
        
        # Load Balancing Logic
        load_query = "SELECT user_id, COUNT(*) as real_load FROM working_task WHERE end_date IS NULL OR end_date > GETDATE() GROUP BY user_id"
        try:
            load_df = pd.read_sql(load_query, db_engine)
            df = df.merge(load_df, left_on='id', right_on='user_id', how='left')
            df['real_load'] = df['real_load'].fillna(0)
        except: df['real_load'] = 0

        df['solved_history_text'] = df['solved_history_text'].fillna("")
        df['is_present'] = df['is_present'].apply(lambda x: True if x==1 else False)
        return df

    def _update_sql_profile(self, user_id, updates):
        set_clause = ", ".join([f"{k} = :{k}" for k in updates.keys()])
        query = text(f"UPDATE technician_profile SET {set_clause} WHERE user_id = :uid")
        updates['uid'] = user_id
        with db_engine.connect() as conn:
            conn.execute(query, updates)
            conn.commit()
    
    def _train_time_predictor(self):
        X_text = []
        y_time = []
        try:
            sql_query = "SELECT t.name + ' ' + t.description as full_text, (p.avg_resolution_time * 1.0) as actual_time FROM ticket t JOIN working_task wt ON t.id = wt.task_id JOIN technician_profile p ON wt.user_id = p.user_id WHERE t.status = 'Closed'"
            real_data = pd.read_sql(sql_query, db_engine)
            if len(real_data) > 5:
                X_text = real_data['full_text'].tolist()
                y_time = real_data['actual_time'].tolist()
        except: pass

        if len(X_text) < 10:
            external_data = [
                ("printer paper jam fix toner", 0.5), ("mouse not working usb", 0.2), 
                ("keyboard keys stuck clean", 0.5), ("screen black cable hdmi", 0.5),
                ("install windows os format", 2.0), ("internet wifi slow router", 1.0), 
                ("lan cable broken rj45", 1.0), ("vpn connection firewall", 2.0),
                ("reset password login fail", 0.2), ("fix bug api error 500", 4.0),
                ("develop login page react", 8.0), ("create database schema sql", 6.0),
                ("integrate payment gateway api", 12.0), ("train ai model dataset", 24.0)
            ]
            X_text += [d[0] for d in external_data]
            y_time += [d[1] for d in external_data]

        try:
            self.time_vectorizer = TfidfVectorizer()
            X_vectors = self.time_vectorizer.fit_transform(X_text)
            self.time_model.fit(X_vectors, y_time)
            self.is_time_model_trained = True
        except: pass

    def predict_duration(self, text_input):
        if not self.is_time_model_trained: return 2.0
        try:
            vec = self.time_vectorizer.transform([text_input])
            predicted = self.time_model.predict(vec)[0]
            return max(0.5, round(predicted, 1))
        except: return 1.0

    def assign(self, text, floor):
        try:
            corpus = self.df['solved_history_text'].tolist() + [text]
            matrix = self.vectorizer.fit_transform(corpus)
            sim_scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
        except: sim_scores = np.zeros(len(self.df))

        current_floors = self.df['current_floor'].fillna(1).astype(int)
        dists = abs(current_floors - floor)
        prox_scores = 1 - (dists / 10)
        
        loads = self.df['real_load'].values
        load_scores = 1 / (loads + 1) 

        # Fairness Formula: 60% Skill, 30% Load, 10% Location
        final = (sim_scores * 0.60) + (load_scores * 0.30) + (prox_scores * 0.10)
        final[~self.df['is_present']] = -99
        
        if len(final) == 0: return None, 0.0, 0.0, pd.DataFrame()

        best_idx = np.argmax(final)
        winner = self.df.iloc[best_idx]
        est_hours = self.predict_duration(text)

        details_df = self.df[['name', 'real_load']].copy()
        details_df['Final Score'] = np.round(final, 2)
        details_df = details_df.sort_values(by='Final Score', ascending=False)

        return winner, final[best_idx], est_hours, details_df

    def train(self, user_id, ticket_text, time_taken=1.0):
        user_row = self.df[self.df['id'] == user_id].iloc[0]
        new_text = str(user_row['solved_history_text']) + " " + ticket_text
        unique_words = set(new_text.split())
        updated_history = " ".join(unique_words)
        
        new_total = int(user_row['total_finished_tickets'] + 1)
        old_avg = float(user_row['avg_resolution_time']) if user_row['avg_resolution_time'] else 0.0
        new_avg = ((old_avg * (new_total - 1)) + time_taken) / new_total

        self._update_sql_profile(int(user_id), {
            'solved_history_text': updated_history,
            'total_finished_tickets': new_total,
            'avg_resolution_time': new_avg
        })

class StockAI:
    def get_items(self):
        try: return pd.read_sql("SELECT id, name, quantity FROM stock_item", db_engine)
        except: return pd.DataFrame()

    def predict(self, current_qty):
        dates = pd.date_range(end=datetime.date.today(), periods=60)
        trend = np.linspace(current_qty + 50, current_qty, 60)
        noise = np.random.normal(0, 2, 60)
        qty = trend + noise
        df = pd.DataFrame({'date': dates, 'quantity': qty})
        
        X = np.arange(len(df)).reshape(-1, 1)
        y = df['quantity'].values
        model = LinearRegression()
        model.fit(X, y)
        slope = model.coef_[0]
        
        if slope >= 0: return None, slope, df
        days = (-model.intercept_ / slope) - len(df)
        return int(days), slope, df

# ==========================================
# 2. Chatbot Core (From Notebook)
# ==========================================
class SQLBackendClient:
    """Reads real data from SQL for the Chatbot"""
    def get_tasks(self, filters=None):
        query = "SELECT t.name as title, t.status, u.name as assigned_to FROM task t LEFT JOIN working_task wt ON t.id = wt.task_id LEFT JOIN [user] u ON wt.user_id = u.id"
        df = pd.read_sql(query, db_engine)
        if filters and 'status' in filters:
            status_map = {'todo': 'To Do', 'in_progress': 'In Progress', 'completed': 'Completed'}
            target = status_map.get(filters['status'], filters['status']).lower()
            df = df[df['status'].str.lower() == target]
        return df.to_dict('records')

    def get_tickets(self, filters=None):
        query = "SELECT name as title, priority, status FROM ticket"
        df = pd.read_sql(query, db_engine)
        if filters and 'priority' in filters:
            df = df[df['priority'].str.lower() == filters['priority'].lower()]
        return df.to_dict('records')

    def get_stock(self, filters=None):
        query = "SELECT name as item_name, quantity, 'pieces' as unit FROM stock_item"
        df = pd.read_sql(query, db_engine)
        return df.to_dict('records')

class IntentDetector:
    TASKS_KEYWORDS = ['task', 'tasks', 'todo', 'progress', 'completed', 'work']
    TICKETS_KEYWORDS = ['ticket', 'tickets', 'bug', 'issue', 'problem']
    STOCK_KEYWORDS = ['stock', 'inventory', 'items', 'equipment']

    @classmethod
    def detect(cls, message: str):
        message_lower = message.lower()
        intent = {'type': 'general', 'filters': {}, 'needs_backend': False}
        
        if any(kw in message_lower for kw in cls.TASKS_KEYWORDS):
            intent['type'] = 'tasks'; intent['needs_backend'] = True
        elif any(kw in message_lower for kw in cls.TICKETS_KEYWORDS):
            intent['type'] = 'tickets'; intent['needs_backend'] = True
        elif any(kw in message_lower for kw in cls.STOCK_KEYWORDS):
            intent['type'] = 'stock'; intent['needs_backend'] = True
            
        if 'completed' in message_lower: intent['filters']['status'] = 'completed'
        if 'high' in message_lower: intent['filters']['priority'] = 'High'
        return intent
# ==========================================
# 2. Chatbot Core (Smart Hybrid Mode) 🧠
# ==========================================
class AIChatbot:
    def __init__(self):
        self.client = SQLBackendClient()
        self.intent_detector = IntentDetector()
        self.model = None
        
        # الاتصال التلقائي بأي موديل متاح
        try:
            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            selected_model = next((m for m in available_models if 'flash' in m), available_models[0] if available_models else None)
            if selected_model:
                print(f"✅ AI Connected: {selected_model}")
                self.model = genai.GenerativeModel(selected_model)
        except Exception as e:
            print(f"⚠️ AI Model Error: {e}")

    def chat(self, message):
        if not self.model: return "⚠️ AI Model not connected. Check API Key."
        
        intent = self.intent_detector.detect(message)
        context = ""
        
        # 1. لو السؤال يخص الشغل، هات الداتابيز
        if intent['needs_backend']:
            data = []
            if intent['type'] == 'tasks': data = self.client.get_tasks(intent['filters'])
            elif intent['type'] == 'tickets': data = self.client.get_tickets(intent['filters'])
            elif intent['type'] == 'stock': data = self.client.get_stock(intent['filters'])
            
            if data:
                context = f"Here is the real-time data from SQL Database: {json.dumps(data)}"
            else:
                context = "Database query returned no results."

        # 2. التعليمات الجديدة (السر هنا 💡)
        # بنقوله: انت خبير IT، لو فيه داتابيز استخدمها، لو مفيش جاوب من عندك.
        prompt = f"""
        You are a helpful IT Team Assistant. 
        
        YOUR INSTRUCTIONS:
        1. **Data Questions:** If the user asks about tasks, tickets, or stock, use the 'DATABASE CONTEXT' below. If it's empty, say you didn't find records.
        2. **General Questions:** If the user asks general IT questions (e.g., "how to fix printer", "code help"), IGNORE the database and answer from your general IT knowledge.
        3. **Language:** Always answer in the same language as the user (Arabic or English).

        USER QUESTION: "{message}"
        
        DATABASE CONTEXT: 
        {context}
        
        YOUR ANSWER (Professional & Helpful):
        """
        
        try: return self.model.generate_content(prompt).text
        except Exception as e: return f"AI Error: {e}"

# ==========================================
# 3. UI Layout
# ==========================================

st.set_page_config(page_title="AI Ops Center", layout="wide", page_icon="🏢")

st.title("🏢 Smart IT Operations Center")
st.caption(f"Server: {SERVER_NAME} | Database: {DATABASE_NAME}")
st.divider()

with st.sidebar:
    st.header("📍 Context")
    current_company = st.text_input("Company", "Headquarters")
    my_current_floor = st.number_input("Floor", 1, 20, 1)
    st.divider()
    page = st.radio("Navigation", ["🎫 Dispatcher", "📈 Stock Forecast", "📊 Analytics", "📌 Task Board", "💬 AI Assistant"])

# --- PAGE 1: Dispatcher ---
if page == "🎫 Dispatcher":
    st.subheader(f"🚀 Ticket Dispatcher")
    col1, col2 = st.columns([2, 1])
    with col1:
        with st.container(border=True):
            ticket_text = st.text_area("Describe Issue")
            users_df = pd.read_sql("SELECT id, name FROM [user]", db_engine)
            reporter_id = st.selectbox("Reported By", users_df['id'], format_func=lambda x: users_df[users_df['id'] == x]['name'].values[0])
            floor = st.number_input("Issue Floor", 1, 20, my_current_floor)
            
            if st.button("Analyze & Assign ⚡", type="primary", use_container_width=True):
                if not ticket_text: st.warning("Required")
                else:
                    dup = DuplicateTicketDetector()
                    is_dup, match, score = dup.check_is_duplicate(ticket_text, reporter_id)
                    if is_dup: st.error(f"Duplicate ({score:.2f})")
                    else:
                        dispatcher = SmartDispatcher()
                        winner, score, est_time, details = dispatcher.assign(ticket_text, floor)
                        if winner is not None:
                            st.balloons()
                            st.markdown(f"**Assigned to:** {winner['name']} | **Load:** {winner['real_load']} tasks | **Est Time:** {est_time}h")
                            dup.create_ticket_in_sql(ticket_text[:50], ticket_text, reporter_id, int(winner['id']))
                            st.toast("Saved")
                            with st.expander("AI Logic"): st.dataframe(details)
                        else: st.error("No staff.")

    with col2:
        st.subheader("👨‍💻 Team Load")
        dispatcher = SmartDispatcher()
        if not dispatcher.df.empty:
            view = dispatcher.df[['name', 'real_load', 'is_present']].copy()
            view.columns = ['Name', 'Tasks', 'Present']
            view['Status'] = view['Present'].apply(lambda x: "🟢" if x else "🔴")
            st.dataframe(view, hide_index=True)
        
        st.divider()
        st.subheader("✅ Close Ticket")
        try:
            open_tix = pd.read_sql("SELECT id, name, assign_to FROM ticket WHERE status='Open'", db_engine)
            if not open_tix.empty:
                tid = st.selectbox("Select Ticket", open_tix['id'], format_func=lambda x: f"#{x} {open_tix[open_tix['id']==x]['name'].values[0]}")
                hours = st.slider("Actual Hours", 0.5, 8.0, 1.0)
                if st.button("Close & Train"):
                    row = open_tix[open_tix['id']==tid].iloc[0]
                    dispatcher.train(row['assign_to'], row['name'], hours)
                    with db_engine.connect() as conn:
                        conn.execute(text(f"UPDATE ticket SET status='Closed' WHERE id={tid}"))
                        conn.commit()
                    st.success("Updated"); time.sleep(1); st.rerun()
        except: pass

# --- PAGE 2: Stock ---
elif page == "📈 Stock Forecast":
    st.subheader("📦 Stock Forecast")
    ai = StockAI()
    items = ai.get_items()
    if not items.empty:
        c1, c2 = st.columns([1, 2])
        with c1:
            item = st.selectbox("Select Item", items['name'])
            curr = items[items['name']==item]['quantity'].values[0]
            st.metric("Current Stock", f"{curr}")
        days, slope, chart = ai.predict(curr)
        with c2: st.line_chart(chart.set_index('date')['quantity'])

# --- PAGE 3: Analytics ---
elif page == "📊 Analytics":
    st.subheader("📊 Team Performance")
    try:
        df_perf = pd.read_sql("SELECT u.name, p.total_finished_tickets, ROUND(p.avg_resolution_time, 1) as avg_time FROM [user] u JOIN technician_profile p ON u.id = p.user_id WHERE p.total_finished_tickets > 0", db_engine)
        if not df_perf.empty:
            c1, c2 = st.columns(2)
            with c1: st.bar_chart(df_perf.set_index('name')['total_finished_tickets'])
            with c2: st.bar_chart(df_perf.set_index('name')['avg_time'])
    except: st.error("Error loading analytics")

# --- PAGE 4: Task Board ---
elif page == "📌 Task Board":
    st.subheader("📌 Smart Task Board (AI Load Balanced)")
    col_filter1, col_filter2 = st.columns(2)
    with col_filter1: team_id = st.selectbox("Select Team", [1, 2, 3], format_func=lambda x: f"Team {x}")
    with col_filter2: sprint_id = st.selectbox("Current Sprint", [1], format_func=lambda x: "Sprint 1")

    if st.button("🤖 AI Auto-Assign (Fair & Balanced)"):
        with st.spinner("Balancing workload..."):
            sql_unassigned = f"SELECT id, name, description FROM task WHERE sprint_id = {sprint_id} AND (assigned = 0 OR assigned IS NULL)"
            try:
                unassigned_df = pd.read_sql(sql_unassigned, db_engine)
                if unassigned_df.empty: st.success("Everything assigned! 🎉")
                else:
                    dispatcher = SmartDispatcher()
                    cnt = 0
                    for idx, row in unassigned_df.iterrows():
                        winner, score, est_time, _ = dispatcher.assign(str(row['description']), my_current_floor)
                        if winner is not None:
                            with db_engine.connect() as conn:
                                conn.execute(text("INSERT INTO working_task (task_id, team_id, user_id, start_date) VALUES (:tid, :teid, :uid, GETDATE())"), {'tid': int(row['id']), 'teid': int(team_id), 'uid': int(winner['id'])})
                                conn.execute(text(f"UPDATE task SET assigned=1 WHERE id={int(row['id'])}"))
                                conn.commit()
                                dispatcher.df.loc[dispatcher.df['id'] == winner['id'], 'real_load'] += 1
                            cnt += 1
                    st.success(f"Assigned {cnt} tasks!"); time.sleep(1); st.rerun()
            except Exception as e: st.error(f"Error: {e}")
    
    st.divider()
    sql_tasks = f"SELECT t.id, t.name, t.description, t.status, t.priority, u.name as assignee FROM task t LEFT JOIN working_task wt ON t.id = wt.task_id LEFT JOIN [user] u ON wt.user_id = u.id WHERE t.sprint_id = {sprint_id}"
    df_tasks = pd.read_sql(sql_tasks, db_engine)
    col_todo, col_prog, col_done = st.columns(3)
    
    def render_card(row):
        with st.container(border=True):
            st.markdown(f"**{row['name']}**")
            st.caption(f"👤 {row['assignee'] if row['assignee'] else '⚪ Unassigned'} | 🔥 {row['priority']}")
            c1, c2 = st.columns(2)
            if row['status'] == 'To Do' and c2.button("➡️", key=f"s{row['id']}"): update_status(row['id'], 'In Progress')
            elif row['status'] == 'In Progress':
                 if c1.button("⬅️", key=f"b{row['id']}"): update_status(row['id'], 'To Do')
                 if c2.button("✅", key=f"d{row['id']}"): update_status(row['id'], 'Completed')

    def update_status(tid, new_status):
        with db_engine.connect() as conn:
            conn.execute(text(f"UPDATE task SET status='{new_status}' WHERE id={tid}"))
            conn.commit(); st.rerun()

    with col_todo: 
        st.header("📝 To Do")
        for _, r in df_tasks[df_tasks['status'] == 'To Do'].iterrows(): render_card(r)
    with col_prog: 
        st.header("⏳ In Progress")
        for _, r in df_tasks[df_tasks['status'] == 'In Progress'].iterrows(): render_card(r)
    with col_done: 
        st.header("✅ Completed")
        for _, r in df_tasks[df_tasks['status'] == 'Completed'].iterrows(): render_card(r)

# --- PAGE 5: AI Assistant (New) ---
elif page == "💬 AI Assistant":
    st.subheader("💬 AI Project Assistant")
    if "messages" not in st.session_state: st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])

    if prompt := st.chat_input("Ask about tasks, tickets, or stock..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Checking Database..."):
                bot = AIChatbot()
                response = bot.chat(prompt)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})