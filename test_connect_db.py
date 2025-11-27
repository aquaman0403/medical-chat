import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()


def test_connection():
    db_url = os.environ.get("DATABASE_URL")
    print(f"Testing connection to: {db_url}")

    if not db_url:
        print("Error: DATABASE_URL not found in .env")
        return

    try:
        conn = psycopg2.connect(db_url)
        print("Connection successful!")

        # Check if the 'messages' table exists
        cur = conn.cursor()
        cur.execute("""
                    SELECT EXISTS (SELECT
                                   FROM information_schema.tables
                                   WHERE table_schema = 'public'
                                     AND table_name = 'messages');
                    """)
        exists = cur.fetchone()[0]

        if exists:
            print("\nTable 'messages' found!")

            # Print columns to verify schema
            cur.execute("""
                        SELECT column_name, data_type
                        FROM information_schema.columns
                        WHERE table_name = 'messages';
                        """)
            columns = cur.fetchall()
            print("Columns:")
            for col in columns:
                print(f"- {col[0]} ({col[1]})")
        else:
            print("\nTable 'messages' NOT found!")

        cur.close()
        conn.close()

    except Exception as e:
        print(f"Connection failed: {e}")


if __name__ == "__main__":
    test_connection()
