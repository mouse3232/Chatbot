import json
import datetime
import logging
import os
import sqlite3
import uuid
from typing import List, Dict, Any, Optional

from backend.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

class SessionManager:
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(PROJECT_ROOT, "logs", "sessions.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Initialize SQLite tables for sessions and messages."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            # Profiles table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS profiles (
                    session_id TEXT PRIMARY KEY,
                    mobile_number TEXT,
                    user_profile TEXT,
                    row_refs TEXT,
                    conversation_summary TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Check if mobile_number column exists (migration)
            cursor.execute("PRAGMA table_info(profiles)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'mobile_number' not in columns:
                logger.info("Adding mobile_number column to profiles table.")
                cursor.execute("ALTER TABLE profiles ADD COLUMN mobile_number TEXT")
            # Messages table (Legacy - keeping for safety but will transition)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES profiles (session_id)
                )
            """)

            # New Category-specific Session Tables
            categories = ["kundali", "horary", "remedies", "matching"]
            for cat in categories:
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {cat}_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        user_query TEXT,
                        assistant_response TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES profiles (session_id)
                    )
                """)
            
            # Unified Interactions table (Production Grade)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    interaction_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    category TEXT,
                    question_id TEXT,
                    answer_id TEXT,
                    parent_id TEXT,
                    user_query TEXT,
                    assistant_response TEXT,
                    metadata TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES profiles (session_id)
                )
            """)
            
            # Audit Logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    category TEXT,
                    message TEXT,
                    session_id TEXT,
                    metadata TEXT
                )
            """)
            conn.commit()

    def save_session(self, session_id: str, user_profile: Dict[str, Any], row_refs: Optional[List[Dict[str, Any]]] = None, mobile_number: Optional[str] = None):
        """Save or update user profile, row refs, and mobile number in SQLite."""
        profile_json = json.dumps(user_profile)
        refs_json = json.dumps(row_refs) if row_refs is not None else None
        now = datetime.datetime.now().isoformat()
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO profiles (session_id, mobile_number, user_profile, row_refs, last_activity)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    mobile_number = COALESCE(excluded.mobile_number, profiles.mobile_number),
                    user_profile = excluded.user_profile,
                    row_refs = COALESCE(excluded.row_refs, profiles.row_refs),
                    last_activity = excluded.last_activity
            """, (session_id, mobile_number, profile_json, refs_json, now))
            conn.commit()

    def update_summary(self, session_id: str, summary: str):
        """Update the conversation summary for a session."""
        now = datetime.datetime.now().isoformat()
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE profiles SET conversation_summary = ?, last_activity = ? WHERE session_id = ?
            """, (summary, now, session_id))
            conn.commit()

    def get_summary(self, session_id: str) -> str:
        """Retrieve the current conversation summary."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT conversation_summary FROM profiles WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            return row[0] if row and row[0] else ""

    def add_message(self, session_id: str, role: str, content: str):
        """
        [DEPRECATED] Add a message to the legacy messages table.
        Used only for backward compatibility during transition.
        """
        now = datetime.datetime.now().isoformat()
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO messages (session_id, role, content, timestamp)
                VALUES (?, ?, ?, ?)
            """, (session_id, role, content, now))
            cursor.execute("UPDATE profiles SET last_activity = ? WHERE session_id = ?", (now, session_id))
            conn.commit()

    def add_interaction(self, session_id: str, category: str, user_query: str, assistant_response: str):
        """
        Adds a complete interaction (User Q + Assistant A) to the appropriate category table.
        Prevents duplication and organizes data by intent.
        """
        category = category.lower()
        valid_categories = ["kundali", "horary", "remedies", "matching"]
        if category not in valid_categories:
            category = "kundali" # Fallback
            
        table_name = f"{category}_sessions"
        now = datetime.datetime.now().isoformat()
        
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                INSERT INTO {table_name} (session_id, user_query, assistant_response, timestamp)
                VALUES (?, ?, ?, ?)
            """, (session_id, user_query, assistant_response, now))
            # Update last activity in profile
            cursor.execute("UPDATE profiles SET last_activity = ? WHERE session_id = ?", (now, session_id))
            conn.commit()

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve full session data (profile + aggregated interactions) from SQLite."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            # 1. Get profile
            cursor.execute("SELECT user_profile, row_refs, conversation_summary, mobile_number FROM profiles WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            if not row:
                return None
            user_profile = json.loads(row[0])
            row_refs = json.loads(row[1]) if row[1] else []
            summary = row[2] if row[2] else ""
            mobile_number = row[3]

            # 2. Aggregation from all category tables
            all_messages = []
            categories = ["kundali", "horary", "remedies", "matching"]
            for cat in categories:
                cursor.execute(f"""
                    SELECT user_query, assistant_response, timestamp FROM {cat}_sessions 
                    WHERE session_id = ?
                """, (session_id,))
                for r in cursor.fetchall():
                    # Map to legacy format for UI compatibility if needed
                    # but also preserving the pair structure
                    all_messages.append({"role": "user", "content": r[0], "timestamp": r[2]})
                    all_messages.append({"role": "assistant", "content": r[1], "timestamp": r[2]})
            
            # Sort by timestamp to maintain conversation flow
            all_messages.sort(key=lambda x: x.get("timestamp", ""))

        return {
            "session_id": session_id,
            "mobile_number": mobile_number,
            "user_profile": user_profile,
            "row_refs": row_refs,
            "messages": all_messages,
            "conversation_summary": summary
        }

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions from SQLite."""
        sessions = []
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT session_id, user_profile, last_activity, mobile_number 
                FROM profiles 
                ORDER BY last_activity DESC
            """)
            for row in cursor.fetchall():
                profile = json.loads(row[1])
                sessions.append({
                    "session_id": row[0],
                    "user_name": profile.get("name", "User"),
                    "last_activity": row[2],
                    "mobile_number": row[3]
                })
        return sessions

    def delete_session(self, session_id: str):
        """Delete session data from SQLite."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            # Delete from category tables
            categories = ["kundali", "horary", "remedies", "matching"]
            for cat in categories:
                cursor.execute(f"DELETE FROM {cat}_sessions WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM profiles WHERE session_id = ?", (session_id,))
            conn.commit()
        
        # Also delete potential log files in the sessions directory
        from backend.config import settings
        logs_root = settings.SESSION_LOGS_ROOT
        import glob
        for f in glob.glob(os.path.join(logs_root, f"{session_id}*")):
            try:
                os.remove(f)
            except Exception as e:
                logger.warning(f"Failed to remove session file {f}: {e}")

    def cleanup_retention(self):
        """
        Executes data retention policies:
        1. Delete messages older than 180 days.
        2. Delete messages for sessions not accessed in the last 30 days.
        3. Delete entire sessions not accessed in the last 30 days.
        """
        logger.info("Executing data retention cleanup...")
        now = datetime.datetime.now()
        thirty_days_ago = (now - datetime.timedelta(days=30)).isoformat()
        one_eighty_days_ago = (now - datetime.timedelta(days=180)).isoformat()

        with self._get_conn() as conn:
            cursor = conn.cursor()
            
            # 1. Delete messages older than 180 days (Legacy messages table)
            cursor.execute("DELETE FROM messages WHERE timestamp < ?", (one_eighty_days_ago,))
            
            # Category-specific sessions cleanup
            categories = ["kundali", "horary", "remedies", "matching"]
            for cat in categories:
                # 1. Delete category interactions older than 180 days
                cursor.execute(f"DELETE FROM {cat}_sessions WHERE timestamp < ?", (one_eighty_days_ago,))
                
                # 2. Delete messages for sessions not accessed in last 30 days
                # Find session_ids that are old
                cursor.execute(f"""
                    DELETE FROM {cat}_sessions 
                    WHERE session_id IN (SELECT session_id FROM profiles WHERE last_activity < ?)
                """, (thirty_days_ago,))

            # 3. Identify and delete sessions not accessed in last 30 days
            cursor.execute("SELECT session_id FROM profiles WHERE last_activity < ?", (thirty_days_ago,))
            old_sessions = [row[0] for row in cursor.fetchall()]
            
            conn.commit()

        # Perform full deletion for each old session (including files)
        for sid in old_sessions:
            logger.info(f"Cleaning up inactive session: {sid}")
            self.delete_session(sid)

        self.cleanup_temp_context()
        logger.info("Data retention cleanup completed.")

    def cleanup_temp_context(self):
        """Deletes files in /logs/temp_context older than 2 hours."""
        try:
            temp_dir = os.path.join(PROJECT_ROOT, "logs", "temp_context")
            if not os.path.exists(temp_dir):
                return
                
            import time
            now = time.time()
            two_hours_in_seconds = 2 * 3600
            
            deleted_count = 0
            for f in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, f)
                if os.path.isfile(file_path):
                    if os.stat(file_path).st_mtime < (now - two_hours_in_seconds):
                        os.remove(file_path)
                        deleted_count += 1
            
            if deleted_count > 0:
                logger.info(f"[SessionManager] Cleaned up {deleted_count} expired LLM context logs.")
        except Exception as e:
            logger.error(f"[SessionManager] Failed to cleanup temp context: {e}")

    def save_interaction(self, session_id: str, category: str, query: str, response: str, 
                         parent_id: Optional[str] = None, metadata: Optional[Dict] = None,
                         interaction_id: Optional[str] = None) -> Dict[str, str]:
        """
        Saves a Q/A interaction with unique IDs and optional parent linking.
        Writes to both Redis (speed) and SQLite (persistence).
        """
        import backend.astro_rag.session.chat_history as ch
        
        interaction_id = interaction_id or str(uuid.uuid4())
        question_id = str(uuid.uuid4())
        answer_id = str(uuid.uuid4())
        
        # 1. Save to Redis (Hot Data)
        ch.record_interaction(
            session_id=session_id,
            query=query,
            response=response
        )
        
        # 2. Save to SQLite (Permanent Record)
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO interactions (
                    interaction_id, session_id, category, question_id, answer_id, 
                    parent_id, user_query, assistant_response, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                interaction_id, session_id, category, question_id, answer_id,
                parent_id, query, response, json.dumps(metadata) if metadata else None
            ))
            
            # Also update last_activity for the profile
            cursor.execute("UPDATE profiles SET last_activity = CURRENT_TIMESTAMP WHERE session_id = ?", (session_id,))
            conn.commit()
            
        logger.info(f"[SessionManager] Saved interaction {interaction_id} for session {session_id}")
        
        return {
            "interaction_id": interaction_id,
            "question_id": question_id,
            "answer_id": answer_id,
            "parent_id": parent_id
        }

    def get_interaction_graph(self, session_id: str) -> List[Dict[str, Any]]:
        """Fetches the full conversation history for a session from SQLite."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM interactions 
                WHERE session_id = ? 
                ORDER BY timestamp ASC
            """, (session_id,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
