-- The activation code is NEVER persisted in plaintext; code_hash is HMAC-SHA256.
CREATE TABLE IF NOT EXISTS licenses (
  id TEXT PRIMARY KEY,
  code_hash TEXT NOT NULL UNIQUE,
  code_suffix TEXT NOT NULL,
  label TEXT NOT NULL DEFAULT '',
  created_at INTEGER NOT NULL,
  expires_at INTEGER,
  revoked_at INTEGER,
  public_key TEXT,
  device_hash TEXT,
  bound_at INTEGER,
  generation INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS licenses_status_idx ON licenses(revoked_at,expires_at);
CREATE TABLE IF NOT EXISTS challenges (
  id TEXT PRIMARY KEY,
  code_hash TEXT NOT NULL,
  nonce TEXT NOT NULL,
  expires_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS challenges_expiration_idx ON challenges(expires_at);
CREATE TABLE IF NOT EXISTS rate_limits (
  key TEXT PRIMARY KEY,
  window_start INTEGER NOT NULL,
  hits INTEGER NOT NULL
);
