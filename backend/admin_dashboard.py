import streamlit as st
import sqlite3
import pandas as pd
import json
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.config import PROJECT_ROOT

# Page Configuration
st.set_page_config(page_title="Astrology AI Ops Hub", page_icon="🪐", layout="wide")

DB_PATH = os.path.join(PROJECT_ROOT, "logs", "sessions.db")

# ── Data Fetching Logic (Cached) ──

@st.cache_data(ttl=30)
def get_all_sessions():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT session_id, mobile_number, user_profile, created_at, last_activity 
        FROM profiles 
        ORDER BY last_activity DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    def extract_name(p_str):
        try: return json.loads(p_str).get("name", "Unknown")
        except: return "Unknown"
    
    df["User Name"] = df["user_profile"].apply(extract_name)
    return df

@st.cache_data(ttl=10)
def get_interaction_graph(session_id):
    conn = sqlite3.connect(DB_PATH)
    # Fetch from the new unified interactions table
    query = f"SELECT * FROM interactions WHERE session_id = '{session_id}' ORDER BY timestamp ASC"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

@st.cache_data(ttl=10)
def get_audit_logs(session_id=None, limit=500):
    conn = sqlite3.connect(DB_PATH)
    if session_id:
        query = f"SELECT * FROM audit_logs WHERE session_id = '{session_id}' ORDER BY timestamp DESC LIMIT {limit}"
    else:
        query = f"SELECT * FROM audit_logs ORDER BY timestamp DESC LIMIT {limit}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

@st.cache_data(ttl=60)
def get_table_data(table_name):
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 1000", conn)
    except:
        df = pd.DataFrame()
    conn.close()
    return df

# ── Sidebar: Navigation & Session Selection ──

st.sidebar.title("🪐 AI Ops Hub")
st.sidebar.markdown("Unified Operations & Logging")

df_sessions = get_all_sessions()

if df_sessions.empty:
    st.sidebar.warning("No sessions found in DB.")
    st.stop()

st.sidebar.subheader("Sessions")
search_user = st.sidebar.text_input("🔍 Search Name/ID", "")
filtered_sessions = df_sessions[
    df_sessions["User Name"].str.contains(search_user, case=False) | 
    df_sessions["session_id"].str.contains(search_user, case=False)
]

# Session List in Sidebar
selected_sid = st.sidebar.radio(
    "Select Session",
    options=filtered_sessions["session_id"].tolist(),
    format_func=lambda sid: f"{filtered_sessions[filtered_sessions['session_id']==sid]['User Name'].values[0]} ({sid[:6]})"
)

# ── Main Content Area ──

if selected_sid:
    # Fetch current session details
    s_row = df_sessions[df_sessions["session_id"] == selected_sid].iloc[0]
    
    st.title(f"👤 {s_row['User Name']}")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Session ID", selected_sid)
    col_b.metric("Mobile", s_row["mobile_number"] or "N/A")
    col_c.metric("Last Active", pd.to_datetime(s_row["last_activity"]).strftime("%Y-%m-%d %H:%M"))

    tabs = st.tabs(["💬 Chat Graph", "📋 User Profile", "📜 Audit Logs", "🕵️ RAG & Matching", "⚙️ DB Explorer"])

    # 1. Chat Graph Tab (Linked Q&A)
    with tabs[0]:
        st.subheader("Conversation Graph")
        df_graph = get_interaction_graph(selected_sid)
        
        if df_graph.empty:
            st.info("No unified interactions found. (Legacy sessions may be in DB Explorer)")
        else:
            for _, row in df_graph.iterrows():
                # Display Q/A with IDs
                with st.container(border=True):
                    st.markdown(f"**ID:** `{row['interaction_id']}` | **Category:** `{row['category'].upper()}`")
                    if row['parent_id']:
                        st.caption(f"🔗 Linked to Parent: `{row['parent_id']}`")
                    
                    with st.chat_message("user"):
                        st.write(f"**Q ({row['question_id'][:8]}):** {row['user_query']}")
                    
                    with st.chat_message("assistant"):
                        st.write(f"**A ({row['answer_id'][:8]}):** {row['assistant_response']}")
                    
                    st.caption(f"🕒 {row['timestamp']}")

    # 2. User Profile Tab
    with tabs[1]:
        st.subheader("User Profile Metadata")
        view_mode = st.radio("View Format", ["Pretty", "Raw JSON"], horizontal=True, key="prof_view")
        
        try:
            profile_data = json.loads(s_row["user_profile"])
            if view_mode == "Pretty":
                rows = []
                for k, v in profile_data.items():
                    rows.append({"Field": k, "Value": str(v)})
                st.table(pd.DataFrame(rows))
            else:
                st.json(profile_data)
        except:
            st.write(s_row["user_profile"])

    # 3. Audit Logs Tab
    with tabs[2]:
        st.subheader("Terminal Audit Logs")
        scope = st.radio("Scope", ["Current Session", "Global (Last 500)"], horizontal=True)
        
        df_audit = get_audit_logs(selected_sid if scope == "Current Session" else None)
        
        if df_audit.empty:
            st.info("No audit logs found for this scope.")
        else:
            cats = ["ALL"] + sorted(df_audit["category"].unique().tolist())
            selected_cat = st.selectbox("Filter Category", cats)
            
            if selected_cat != "ALL":
                df_audit = df_audit[df_audit["category"] == selected_cat]
            
            st.dataframe(df_audit[["timestamp", "category", "message"]], use_container_width=True)
            
            st.markdown("### Metadata Inspector")
            selected_log_id = st.selectbox("Select Log ID to view details", df_audit["id"].tolist())
            if selected_log_id:
                meta = df_audit[df_audit["id"] == selected_log_id]["metadata"].values[0]
                if meta:
                    st.json(json.loads(meta))

    # 4. RAG & Matching Inspector
    with tabs[3]:
        st.subheader("Deep Data Insight")
        df_init = get_audit_logs(selected_sid, limit=50)
        df_init = df_init[df_init["category"].isin(["SESSION_INIT", "MATCHING", "HORARY_INIT"])]
        
        if df_init.empty:
            st.info("No RAG Init or Matching trace records found.")
        else:
            for _, row in df_init.iterrows():
                with st.expander(f"Trace: {row['category']} at {row['timestamp']}"):
                    meta = json.loads(row["metadata"]) if row["metadata"] else {}
                    if "records" in meta and "records2" in meta:
                        c1, c2 = st.columns(2)
                        c1.markdown("**Male (Records 1)**")
                        c1.json(meta["records"])
                        c2.markdown("**Female (Records 2)**")
                        c2.json(meta["records2"])
                        if "matchrecords" in meta:
                            st.json(meta["matchrecords"])
                    else:
                        st.json(meta)

    # 5. DB Explorer Tab
    with tabs[4]:
        st.subheader("Raw Database Table Browser")
        tables = ["profiles", "interactions", "audit_logs", "kundali_sessions", "horary_sessions", "remedies_sessions", "matching_sessions", "messages"]
        selected_tbl = st.selectbox("Select Table", tables)
        if selected_tbl:
            df_tbl = get_table_data(selected_tbl)
            st.dataframe(df_tbl, use_container_width=True)

else:
    st.info("👈 Select a session from the sidebar to view details.")
