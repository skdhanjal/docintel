import psycopg

conn = psycopg.connect("postgresql://docintel:docintel_dev@localhost/docintel")
conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
conn.commit()
print("pgvector extension enabled")
conn.close()