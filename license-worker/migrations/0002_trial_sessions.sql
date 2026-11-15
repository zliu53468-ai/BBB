-- Anonymous free trial records; deliberately independent of paid licenses.
-- The device ID and secret session token are stored as HMAC-SHA256 hashes.
CREATE TABLE IF NOT EXISTS trial_sessions (
  id TEXT PRIMARY KEY,
  device_hash TEXT NOT NULL UNIQUE,
  token_hash TEXT NOT NULL UNIQUE,
  started_at INTEGER NOT NULL,
  expires_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS trial_sessions_expiry_idx ON trial_sessions(expires_at);
