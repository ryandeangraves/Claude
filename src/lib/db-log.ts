import { neon } from "@neondatabase/serverless";

// FIX #18 (LOW): Module-level singleton – see db.ts for rationale.
// FIX #12 (MEDIUM): Connection errors are caught so that a bad DATABASE_URL
// does not leak credentials through an unhandled exception.
let _db: ReturnType<typeof neon>;

function getDb() {
  if (!_db) {
    try {
      _db = neon(process.env.DATABASE_URL!);
    } catch {
      console.error("[DB-LOG] Failed to initialise database connection.");
      throw new Error("Database unavailable");
    }
  }
  return _db;
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface LogEntry {
  id: number;
  action: string;
  category: string;
  summary: string;
  details: Record<string, unknown>;
  created_at: string;
}

export interface CreateLogEntryInput {
  action: string;
  category: string;
  summary: string;
  details?: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

export async function createLogEntry(
  input: CreateLogEntryInput
): Promise<LogEntry> {
  const db = getDb();
  const rows = await db(
    `INSERT INTO frank_log (action, category, summary, details)
     VALUES ($1, $2, $3, $4)
     RETURNING id, action, category, summary, details, created_at`,
    [
      input.action,
      input.category,
      input.summary,
      JSON.stringify(input.details ?? {}),
    ]
  );
  return rows[0] as LogEntry;
}

export async function getLogEntries(
  limit = 100,
  category?: string
): Promise<LogEntry[]> {
  const db = getDb();
  if (category) {
    const rows = await db(
      `SELECT id, action, category, summary, details, created_at
       FROM frank_log
       WHERE category = $1
       ORDER BY created_at DESC
       LIMIT $2`,
      [category, limit]
    );
    return rows as LogEntry[];
  }
  const rows = await db(
    `SELECT id, action, category, summary, details, created_at
     FROM frank_log
     ORDER BY created_at DESC
     LIMIT $1`,
    [limit]
  );
  return rows as LogEntry[];
}
