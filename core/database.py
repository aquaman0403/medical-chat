import os
import psycopg2
from psycopg2.extras import RealDictCursor
import uuid


class SupabaseDB:
    def __init__(self):
        self.db_url = os.environ.get("DATABASE_URL")
        if not self.db_url:
            raise ValueError("DATABASE_URL environment variable not set")

    def _get_connection(self):
        return psycopg2.connect(self.db_url)

    def save_message(self, session_id: str, role: str, content: str):
        """
        Save a message to the database.
        Maps 'session_id' -> 'conversation_id'
        Maps 'role' -> 'sender' ('user' or 'bot')
        Timestamps are stored in UTC timezone
        """
        sender = 'user' if role == 'user' else 'bot'
        message_id = str(uuid.uuid4())

        conn = None
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            # Insert message with UTC timestamp
            # NOW() AT TIME ZONE 'UTC' ensures UTC timestamp regardless of server timezone
            cur.execute("""
                        INSERT INTO messages (id, conversation_id, content, sender, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, NOW() AT TIME ZONE 'UTC', NOW() AT TIME ZONE 'UTC')
                        """, (message_id, session_id, content, sender))

            conn.commit()
            cur.close()
        except Exception as e:
            print(f"Error saving message to DB: {e}")
        finally:
            if conn:
                conn.close()

    def get_chat_history(self, session_id: str):
        """
        Retrieve chat history for a session.
        Returns a list of dicts with 'role', 'content', 'timestamp'.
        """
        conn = None
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            cur.execute("""
                        SELECT content, sender, created_at
                        FROM messages
                        WHERE conversation_id = %s
                        ORDER BY created_at ASC
                        """, (session_id,))

            rows = cur.fetchall()
            messages = []
            for row in rows:
                role = 'user' if row['sender'] == 'user' else 'assistant'
                messages.append({
                    'role': role,
                    'content': row['content'],
                    'timestamp': row['created_at'].isoformat() if row['created_at'] else None
                })

            cur.close()
            return messages
        except Exception as e:
            print(f"Error fetching chat history from DB: {e}")
            return []
        finally:
            if conn:
                conn.close()

    def get_all_sessions(self):
        """
        Get all conversations.
        """
        conn = None
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Assuming the 'conversations' table has 'id', 'created_at', 'updated_at', 'title'
            cur.execute("""
                        SELECT id, created_at, updated_at, title
                        FROM conversations
                        ORDER BY updated_at DESC
                        """)

            rows = cur.fetchall()
            sessions = []
            for row in rows:
                sessions.append({
                    'session_id': row['id'],
                    'created_at': row['created_at'].isoformat() if row['created_at'] else None,
                    'last_active': row['updated_at'].isoformat() if row['updated_at'] else None,
                    'preview': row['title'] or 'No Title'
                })

            cur.close()
            return sessions
        except Exception as e:
            print(f"Error fetching sessions from DB: {e}")
            return []
        finally:
            if conn:
                conn.close()

    def delete_session(self, session_id: str):
        """
        Delete a conversation.
        """
        conn = None
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            cur.execute("DELETE FROM conversations WHERE id = %s", (session_id,))

            conn.commit()
            cur.close()
        except Exception as e:
            print(f"Error deleting session from DB: {e}")
        finally:
            if conn:
                conn.close()
