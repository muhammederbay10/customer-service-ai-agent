# database.py
import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent / "db" / "tddi_db.sqlite3"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS conversations (
  id TEXT PRIMARY KEY,
  user_id TEXT,
  active_agent TEXT DEFAULT 'general',
  pending_switch TEXT,
  running_summary TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  conversation_id TEXT,
  idx INTEGER,
  role TEXT CHECK(role IN ('user','assistant')),
  text TEXT,
  archived INTEGER DEFAULT 0,
  created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_msgs_convo_idx ON messages(conversation_id, idx);
CREATE TABLE IF NOT EXISTS summaries (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  conversation_id TEXT,
  start_idx INTEGER,
  end_idx INTEGER,
  summary TEXT,
  created_at TEXT
);
"""

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

with get_conn() as c:
    c.executescript(SCHEMA)

# --- Conversation state ---

def upsert_conversation(convo_id: str, user_id: str, **fields):
    with get_conn() as c:
        cur = c.execute("SELECT id FROM conversations WHERE id=?", (convo_id,))
        exists = cur.fetchone() is not None
        if not exists:
            c.execute("INSERT INTO conversations(id, user_id) VALUES (?,?)", (convo_id, user_id))
        for k, v in fields.items():
            c.execute(f"UPDATE conversations SET {k}=? WHERE id=?", (v, convo_id))


def get_conversation(convo_id: str):
    with get_conn() as c:
        cur = c.execute("SELECT * FROM conversations WHERE id=?", (convo_id,))
        return cur.fetchone()

# --- Messages ---

def save_message(convo_id: str, role: str, text: str) -> int:
    with get_conn() as c:
        cur = c.execute("SELECT COALESCE(MAX(idx), -1)+1 FROM messages WHERE conversation_id=?",
                        (convo_id,))
        idx = cur.fetchone()[0]
        c.execute(
            "INSERT INTO messages(conversation_id, idx, role, text, created_at) VALUES (?,?,?,?,?)",
            (convo_id, idx, role, text, datetime.utcnow().isoformat())
        )
        return idx


def get_unarchived(convo_id: str, limit: int = 8):
    with get_conn() as c:
        cur = c.execute(
            "SELECT * FROM messages WHERE conversation_id=? AND archived=0 ORDER BY idx DESC LIMIT ?",
            (convo_id, limit)
        )
        rows = list(cur.fetchall())[::-1]
        return rows


def next_four_unarchived(convo_id: str):
    with get_conn() as c:
        cur = c.execute(
            "SELECT * FROM messages WHERE conversation_id=? AND archived=0 ORDER BY idx ASC LIMIT 4",
            (convo_id,)
        )
        rows = list(cur.fetchall())
        return rows if len(rows) == 4 else []


def archive_ids(ids: list[int]):
    if not ids:
        return
    with get_conn() as c:
        q = "UPDATE messages SET archived=1 WHERE id IN (%s)" % ",".join(["?"] * len(ids))
        c.execute(q, ids)

# --- Summaries ---

def append_chunk_summary(convo_id: str, start_idx: int, end_idx: int, summary: str):
    with get_conn() as c:
        c.execute(
            "INSERT INTO summaries(conversation_id, start_idx, end_idx, summary, created_at) VALUES (?,?,?,?,?)",
            (convo_id, start_idx, end_idx, summary, datetime.utcnow().isoformat())
        )


def get_running_summary(convo_id: str) -> str:
    with get_conn() as c:
        cur = c.execute("SELECT running_summary FROM conversations WHERE id=?", (convo_id,))
        row = cur.fetchone()
        return row[0] if row else ""


def set_running_summary(convo_id: str, summary: str):
    with get_conn() as c:
        c.execute("UPDATE conversations SET running_summary=? WHERE id=?", (summary, convo_id))