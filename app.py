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
# 0. CONFIG & DATABASE
# ==========================================
# 🔑 مفتاح Gemini API
GOOGLE_API_KEY = "AIzaSyAE6iQAebVhe8n01yiE0Mz0GooZA0PCUUE" 
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
try:
    db_engine = create_engine(DB_CONNECTION_STR)
    with db_engine.connect() as conn: pass
    print("✅ Connected to Local SQL Server")
except Exception as e:
    st.error(f"❌ Database Error: {e}")
    st.stop()

# ==========================================
# 1. AI LOGIC (Dispatcher & Stock)
# ==========================================
class DuplicateTicketDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer() 
        self.GLOBAL_KEYWORDS = ['wifi', 'internet', 'network', 'server', 'power', 'system', 'database', 'نت', 'سيرفر', 'كهرباء', 'نظام']

    def _get_open_tickets(self):
        try: return pd.read_sql("SELECT id, name, description, created_by FROM ticket WHERE status = 'Open'", db_engine)
        except: return pd.DataFrame()

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
            return True, match_row, best_score
        return False, None, best_score

    def create_ticket_in_sql(self, title, desc, created_by, assigned_to):
        with db_engine.connect() as conn:
            query = text("INSERT INTO ticket (name, description, created_by, assign_to, priority, status) VALUES (:name, :desc, :creator, :assignee, 'Medium', 'Open')")
            conn.execute(query, {"name": title, "desc": desc, "creator": int(created_by), "assignee": int(assigned_to)})
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
        SELECT u.id, u.name, p.solved_history_text, p.total_finished_tickets, p.avg_resolution_time
        FROM [user] u JOIN technician_profile p ON u.id = p.user_id
        """
        df = pd.read_sql(query, db_engine)
        
        load_query = "SELECT user_id, COUNT(*) as real_load FROM working_task WHERE end_date IS NULL OR end_date > GETDATE() GROUP BY user_id"
        try:
            load_df = pd.read_sql(load_query, db_engine)
            df = df.merge(load_df, left_on='id', right_on='user_id', how='left')
            df['real_load'] = df['real_load'].fillna(0)
        except: df['real_load'] = 0

        df['solved_history_text'] = df['solved_history_text'].fillna("")
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
                ("printer paper jam", 0.5), ("mouse not working", 0.2), 
                ("keyboard issue", 0.5), ("screen black", 0.5),
                ("install windows", 2.0), ("internet slow", 1.0), 
                ("reset password", 0.2), ("api error", 4.0),
                ("create database", 6.0), ("train ai model", 24.0)
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

    def assign(self, text):
        try:
            corpus = self.df['solved_history_text'].tolist() + [text]
            matrix = self.vectorizer.fit_transform(corpus)
            sim_scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
        except: sim_scores = np.zeros(len(self.df))
        
        loads = self.df['real_load'].values
        load_scores = 1 / (loads + 1) 

        final = (sim_scores * 0.70) + (load_scores * 0.30)
        
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
# 2. Chatbot Core (RAG: Retrieval-Augmented Generation) 🧠
# ==========================================
class SQLBackendClient:
    """يجلب البيانات كاملة ويعطيها للـ AI ليفكر فيها"""
    def get_all_context(self):
        context = {}
        try:
            # 1. المهام (Tasks)
            tasks_df = pd.read_sql("SELECT t.name as Task, t.status, u.name as Assigned_To FROM task t LEFT JOIN working_task wt ON t.id = wt.task_id LEFT JOIN [user] u ON wt.user_id = u.id", db_engine)
            context['tasks'] = tasks_df.to_dict('records')
            
            # 2. التيكتات (Tickets)
            tickets_df = pd.read_sql("SELECT name as Ticket, priority, status FROM ticket WHERE status='Open'", db_engine)
            context['open_tickets'] = tickets_df.to_dict('records')
            
            # 3. المخزون (Stock)
            stock_df = pd.read_sql("SELECT name as Item, quantity FROM stock_item", db_engine)
            context['stock'] = stock_df.to_dict('records')
            
        except Exception as e:
            context['error'] = str(e)
            
        return context

class AIChatbot:
    def __init__(self):
        self.client = SQLBackendClient()
        self.model = None
        try:
            available = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            model_name = next((m for m in available if 'flash' in m), available[0] if available else None)
            if model_name: self.model = genai.GenerativeModel(model_name)
        except: pass

    def chat(self, message):
        if not self.model: return "⚠️ AI Model not connected. Check API Key."
        
        # 🔥 السحر هنا: نرسل الداتابيز كلها للـ AI ونقوله اتصرف
        db_context = self.client.get_all_context()
        
        prompt = f"""
        You are a highly intelligent IT Project Assistant named "AI Ops".
        
        SYSTEM CONTEXT (Your Live Database):
        {json.dumps(db_context)}
        
        INSTRUCTIONS:
        1. **Check the Database First:** If the user asks about stock (items, quantities), tasks, or tickets, Look deeply into the 'SYSTEM CONTEXT' JSON above and answer accurately. Count items if asked (e.g., "how many?").
        2. **General/Technical Questions:** If the user asks a technical question (e.g., "how to fix printer", "write python code", "what is agile"), IGNORE the database and answer using your general knowledge as an expert.
        3. **Tone:** Be professional, concise, and helpful.
        4. **Language:** Reply in the same language as the user (Arabic or English).
        
        USER QUESTION: "{message}"
        
        YOUR ANSWER:
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
    st.header("Navigation")
    page = st.radio("Go to:", ["🎫 Dispatcher", "📈 Stock Forecast", "📊 Analytics", "📌 Task Board", "💬 AI Assistant"])

# --- PAGE 1: Dispatcher ---
if page == "🎫 Dispatcher":
    st.subheader(f"🚀 Ticket Dispatcher")
    col1, col2 = st.columns([2, 1])
    with col1:
        with st.container(border=True):
            ticket_text = st.text_area("Describe Issue")
            users_df = pd.read_sql("SELECT id, name FROM [user]", db_engine)
            reporter_id = st.selectbox("Reported By", users_df['id'], format_func=lambda x: users_df[users_df['id'] == x]['name'].values[0])
            
            if st.button("Analyze & Assign ⚡", type="primary", use_container_width=True):
                if not ticket_text: st.warning("Required")
                else:
                    dup = DuplicateTicketDetector()
                    is_dup, match, score = dup.check_is_duplicate(ticket_text, int(reporter_id))
                    if is_dup: st.error(f"Duplicate ({score:.2f})")
                    else:
                        dispatcher = SmartDispatcher()
                        winner, score, est_time, details = dispatcher.assign(ticket_text)
                        if winner is not None:
                            st.balloons()
                            st.markdown(f"**Assigned to:** {winner['name']} | **Load:** {winner['real_load']} tasks | **Est Time:** {est_time}h")
                            dup.create_ticket_in_sql(ticket_text[:50], ticket_text, int(reporter_id), int(winner['id']))
                            st.toast("Saved")
                            with st.expander("AI Logic"): st.dataframe(details)
                        else: st.error("No staff.")

    with col2:
        st.subheader("👨‍💻 Team Load")
        dispatcher = SmartDispatcher()
        if not dispatcher.df.empty:
            view = dispatcher.df[['name', 'real_load']].copy()
            view.columns = ['Name', 'Active Tasks']
            st.dataframe(view, hide_index=True, use_container_width=True)
        
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
        with c2:
            if days:
                if days < 7: st.error(f"⚠️ Runs out in {days} days!")
                else: st.success(f"✅ Enough for {days} days")
            else: st.success("✅ Stock is stable")
            st.line_chart(chart.set_index('date')['quantity'])

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
    st.subheader("📌 Smart Task Board")
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
                        winner, score, est_time, _ = dispatcher.assign(str(row['description']))
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

# --- PAGE 5: AI Assistant ---
elif page == "💬 AI Assistant":
    st.subheader("💬 AI Project Assistant")
    if "messages" not in st.session_state: st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])

    if prompt := st.chat_input("Ask about tasks, tickets, stock, or technical help..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                bot = AIChatbot()
                response = bot.chat(prompt)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})