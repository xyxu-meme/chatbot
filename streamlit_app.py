import streamlit as st
from openai import OpenAI


# Show title and description.
st.title("💬 Chatbot")
st.write(
    "This is a simple chatbot that uses OpenAI's GPT-5 model to generate responses. "
    "This chatbot is purely for the students of 40.318 SCDD 2025 at ESD, SUTD. "
    "Please only use this app for the ESD Supply Chain Game."
)

"""
Streamlit Classroom Chat (OpenAI API, per-student quotas)
---------------------------------------------------------
Quick start:
1) pip install -r requirements.txt
2) Set environment variable OPENAI_API_KEY
3) streamlit run app.py

⚠️ Classroom demo security only. Use behind campus SSO if possible.
"""

import os
import sqlite3
import datetime as dt
import streamlit as st

try:
    from openai import OpenAI
except Exception:
    st.error("OpenAI SDK not found. Run: pip install openai")
    raise

APP_TITLE = "🎓 Classroom GPT Portal"
DB_PATH = "usage.db"
USERS_CSV = "users.csv"

# ---------- Helpers ----------

def get_openai_api_key() -> str:
    # Try Streamlit secrets first, then env var (no hard-coded default!)
    key = None
    try:
        key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        key = os.getenv("OPENAI_API_KEY")
    if not key:
        st.error("Missing OPENAI_API_KEY. Add it to .streamlit/secrets.toml or set the env var.")
        st.stop()
    return key.strip()


def ensure_state_defaults():
    st.session_state.setdefault("chat", [])       # [(role, content), ...]
    st.session_state.setdefault("user_msg", "")   # bound to the text_area
    st.session_state.setdefault("sys_msg", "")    # optional: persist system prompt

def estimate_input_tokens(sys_msg: str, chat_pairs: list, user_msg: str) -> int:
    chunks = []
    if sys_msg and sys_msg.strip():
        chunks.append(sys_msg.strip())
    for role, content in chat_pairs:
        if content:
            chunks.append(content)
    if user_msg and user_msg.strip():
        chunks.append(user_msg.strip())
    return approx_tokens("\n".join(chunks))

def approx_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token)."""
    if not text:
        return 0
    return max(1, int(len(text) / 4))

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute('''
    CREATE TABLE IF NOT EXISTS usage (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        date TEXT,
        prompts INTEGER DEFAULT 0,
        input_tokens INTEGER DEFAULT 0,
        output_tokens INTEGER DEFAULT 0
    )
    ''')
    return conn

def load_users():
    import csv
    users = {}
    if not os.path.exists(USERS_CSV):
        with open(USERS_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["username","password","role","max_requests_per_day","max_tokens_per_day"])
            w.writerow(["alice","alice123","student",25,20000])
            w.writerow(["bob","bob123","student",25,20000])
            w.writerow(["instructor","teach123","admin",99999,1000000])
    with open(USERS_CSV, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            users[row["username"]] = {
                "password": row["password"],
                "role": row.get("role","student"),
                "max_requests_per_day": int(row.get("max_requests_per_day", "25")),
                "max_tokens_per_day": int(row.get("max_tokens_per_day", "20000")),
            }
    return users

def ensure_today_row(conn, username, today_str):
    cur = conn.cursor()
    cur.execute("SELECT id FROM usage WHERE username=? AND date=?", (username, today_str))
    row = cur.fetchone()
    if not row:
        cur.execute("INSERT INTO usage (username, date, prompts, input_tokens, output_tokens) VALUES (?,?,?,?,?)",
                    (username, today_str, 0, 0, 0))
        conn.commit()

def get_usage(conn, username, today_str):
    cur = conn.cursor()
    cur.execute("SELECT prompts, input_tokens, output_tokens FROM usage WHERE username=? AND date=?",
                (username, today_str))
    row = cur.fetchone()
    if not row:
        return 0, 0, 0
    return row[0], row[1], row[2]

def add_usage(conn, username, today_str, add_prompts, add_in_tokens, add_out_tokens):
    cur = conn.cursor()
    cur.execute("""
        UPDATE usage
        SET prompts = prompts + ?,
            input_tokens = input_tokens + ?,
            output_tokens = output_tokens + ?
        WHERE username=? AND date=?
        """,
        (add_prompts, add_in_tokens, add_out_tokens, username, today_str)
    )
    conn.commit()

def reset_today_usage(conn, username, today_str):
    cur = conn.cursor()
    cur.execute("""
        UPDATE usage
        SET prompts = 0, input_tokens = 0, output_tokens = 0
        WHERE username=? AND date=?
        """, (username, today_str))
    conn.commit()

def list_all_usage(conn, today_str=None):
    cur = conn.cursor()
    if today_str:
        cur.execute("SELECT username, date, prompts, input_tokens, output_tokens FROM usage WHERE date=? ORDER BY username", (today_str,))
    else:
        cur.execute("SELECT username, date, prompts, input_tokens, output_tokens FROM usage ORDER BY date DESC, username")
    return cur.fetchall()

# ---------- Auth ----------

def login(users):
    st.sidebar.subheader("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Sign in"):
        if username in users and password == users[username]["password"]:
            st.session_state["user"] = username
            st.session_state["role"] = users[username]["role"]
            st.sidebar.success(f"Welcome, {username}!")
        else:
            st.sidebar.error("Invalid credentials")
    return st.session_state.get("user"), st.session_state.get("role")

def logout():
    if st.sidebar.button("Sign out"):
        for k in ("user","role","chat"):
            if k in st.session_state:
                del st.session_state[k]
        st.success("Signed out.")

# ---------- Chat Page ----------
def page_chat(client, users, conn):
    st.header("Chat")
    user = st.session_state["user"]
    today_str = dt.date.today().isoformat()
    ensure_today_row(conn, user, today_str)

    # ---- Session defaults (before any widgets) ----
    if "chat" not in st.session_state:
        st.session_state["chat"] = []
    if "user_msg" not in st.session_state:
        st.session_state["user_msg"] = ""
    if "sys_msg" not in st.session_state:
        st.session_state["sys_msg"] = (
            "You are a helpful AI teaching assistant. "
            "Be concise, numeric, and specific when appropriate."
        )

    # ---- Quotas ----
    used_prompts, used_in, used_out = get_usage(conn, user, today_str)
    max_req = users[user]["max_requests_per_day"]
    max_tok = users[user]["max_tokens_per_day"]
    remain_req = max(0, max_req - used_prompts)
    remain_tok = max(0, max_tok - (used_in + used_out))

    # ---- Controls (model & settings) ----
    with st.expander("Model & Settings", expanded=True):
        model = st.selectbox("Model", ["gpt-4o", "gpt-5", "gpt-4o-mini"], index=0)

        # Define a default once, but don't assign and bind simultaneously
        default_sys_msg = st.session_state.get(
            "sys_msg",
            "You are a helpful AI teaching assistant. Be concise, numeric, and specific when appropriate."
        )
        sys_msg = st.text_area(
            "System prompt (optional)",
            value=default_sys_msg,
            height=90,
            key="sys_msg"  # widget owns this key
        )

        cols = st.columns(3)
        with cols[0]:
            max_output_tokens = st.slider("Max output tokens", 2580, 12800, 6400, 128)
        with cols[1]:
            temperature = st.slider("Temperature (4o only)", 0.0, 1.5, 0.2, 0.05)
        with cols[2]:
            top_p = st.slider("top_p (4o only)", 0.0, 1.0, 1.0, 0.05)

    # ---- Quota panel ----
    with st.expander("Today’s quota & usage"):
        c = st.columns(4)
        c[0].metric("Prompts used", used_prompts)
        c[1].metric("Prompts remaining", remain_req)
        c[2].metric("Tokens used (est.)", used_in + used_out)
        c[3].metric("Tokens remaining (est.)", remain_tok)

    # ---- 1) Show conversation first ----
    for role, content in st.session_state["chat"]:
        with st.chat_message(role):
            st.markdown(content)

    # ---- 2) Prompt area (SAFE: form with clear_on_submit) ----
    with st.form("chat_form", clear_on_submit=True):
        user_msg = st.text_area(
            "Your message",
            key="user_msg",                  # widget owns this key
            height=110,
            placeholder="Type your message here…"
        )
        send_clicked = st.form_submit_button(
            "Send",
            type="primary",
            disabled=remain_req <= 0 or remain_tok <= 0
        )

    # ---- 3) Handle Send (do NOT write to st.session_state['user_msg']) ----
    if send_clicked:
        if not user_msg.strip():
            st.warning("Please enter a prompt.")
            st.stop()

        # Budget guard
        est_in = estimate_input_tokens(st.session_state["sys_msg"], st.session_state["chat"], user_msg)
        needed_total = est_in + max_output_tokens
        if needed_total > remain_tok:
            st.error(
                "Not enough daily tokens for this request.\n"
                f"Needed: {needed_total} (input {est_in} + output {max_output_tokens}); "
                f"Remaining: {remain_tok}.\n\n"
                "Reduce Max output tokens or click 'Clear conversation', then try again."
            )
            st.stop()

        try:
            # Build messages list (system + history + new user)
            messages = []
            if st.session_state["sys_msg"].strip():
                messages.append({"role": "system", "content": st.session_state["sys_msg"].strip()})
            messages.extend([{"role": r, "content": c} for r, c in st.session_state["chat"]])
            messages.append({"role": "user", "content": user_msg.strip()})

            with st.spinner("Contacting GPT..."):
                reply = ""

                if "gpt-5" in model:
                    # GPT-5 → Responses API
                    joined = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
                    resp = client.responses.create(
                        model=model,
                        input=joined,
                        max_output_tokens=max_output_tokens,
                        reasoning={"effort": "low"},
                        text={"verbosity": "low"},
                    )

                    inc = getattr(resp, "incomplete_details", None)
                    if inc and getattr(inc, "reason", "") == "max_output_tokens":
                        st.info("Hit output limit. Continuing…")
                        resp2 = client.responses.create(
                            model=model,
                            input="Continue from where you left off. Use concise bullet points only.",
                            max_output_tokens=min(max_output_tokens, 1024),
                            reasoning={"effort": "low"},
                            text={"verbosity": "low"},
                        )
                        reply = (getattr(resp2, "output_text", None) or "").strip()
                    else:
                        reply = (getattr(resp, "output_text", None) or "").strip()

                else:
                    # GPT-4o etc. → Chat Completions API
                    resp = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_output_tokens,
                    )
                    reply = (resp.choices[0].message.content or "").strip()

            if not reply:
                st.warning("The model returned no text. Showing raw response for debugging:")
                import json
                st.code(json.dumps(resp.model_dump(), indent=2) if hasattr(resp, "model_dump") else str(resp), language="json")
                st.stop()

            # Record history (append only)
            st.session_state["chat"].append(("user", user_msg.strip()))
            st.session_state["chat"].append(("assistant", reply))
            out_tokens = approx_tokens(reply)
            add_usage(conn, user, today_str, 1, est_in, out_tokens)

            # Input box is cleared automatically by the form; just refresh layout
            st.rerun()

        except Exception as e:
            st.error(f"API error: {e}")
            st.stop()

    # ---- Clear conversation ----
    if st.button("Clear conversation"):
        st.session_state["chat"] = []
        st.success("Conversation cleared.")
        st.experimental_rerun()


# ---------- Admin Page ----------

def page_admin(users, conn):
    st.header("Admin")
    if st.session_state.get("role") != "admin":
        st.warning("Admins only.")
        return
    today_str = dt.date.today().isoformat()
    rows = list_all_usage(conn, today_str)
    if rows:
        import pandas as pd
        df = pd.DataFrame(rows, columns=["username","date","prompts","input_tokens","output_tokens"])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No usage yet today.")

    st.subheader("Reset a user's usage for today")
    user_to_reset = st.selectbox("User", options=sorted(users.keys()))
    if st.button("Reset today's usage"):
        reset_today_usage(conn, user_to_reset, today_str)
        st.success(f"Reset {user_to_reset}'s usage for {today_str}.")

    st.subheader("Upload new users.csv")
    uploaded = st.file_uploader("Upload users.csv", type=["csv"])
    if uploaded is not None:
        with open(USERS_CSV, "wb") as f:
            f.write(uploaded.read())
        st.success("users.csv replaced. Refresh to load new users.")

# ---------- Main ----------

def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🎓", layout="wide")
    st.title(APP_TITLE)

    api_key = get_openai_api_key()
    client = OpenAI(api_key=api_key)
    
    conn = get_db()
    users = load_users()

    user, role = login(users)
    if user:
        st.sidebar.success(f"Signed in as {user} ({role})")
        logout()
        tab1, tab2 = st.tabs(["💬 Chat", "🛠️ Admin"])
        with tab1:
            page_chat(client, users, conn)
        with tab2:
            page_admin(users, conn)
    else:
        st.info("Please sign in to continue.")

if __name__ == "__main__":
    main()
