CREATE INDEX IF NOT EXISTS {}
  ON {}
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = {});