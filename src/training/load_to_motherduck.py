import duckdb
from dotenv import load_dotenv
import os
load_dotenv()

con = duckdb.connect(f"md:?motherduck_token={os.getenv('MOTHERDUCK_TOKEN')}")

con.execute("CREATE DATABASE IF NOT EXISTS titanic_db")
con.execute("USE titanic_db")

con.execute("""
    CREATE OR REPLACE TABLE test_data AS 
    SELECT * FROM read_csv_auto('data/raw/test.csv')
""")

print("✅ Done!")
con.close()