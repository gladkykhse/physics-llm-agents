CREATE TABLE IF NOT EXISTS {} (
  id        bigserial PRIMARY KEY,
  source    text NOT NULL,
  chunk_id  integer NOT NULL,
  text      text NOT NULL,
  embedding vector({}) NOT NULL,
  meta      jsonb DEFAULT '{{}}'
);
