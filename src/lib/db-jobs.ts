import { neon } from "@neondatabase/serverless";

// FIX #18 (LOW): Module-level singleton – see db.ts for rationale.
// FIX #12 (MEDIUM): Connection errors are caught to prevent credential leakage.
let _db: ReturnType<typeof neon>;

function getDb() {
  if (!_db) {
    try {
      _db = neon(process.env.DATABASE_URL!);
    } catch {
      console.error("[DB-JOBS] Failed to initialise database connection.");
      throw new Error("Database unavailable");
    }
  }
  return _db;
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type JobStatus = "pending" | "running" | "done" | "failed";

export interface Job {
  id: number;
  type: string;
  payload: Record<string, unknown>;
  status: JobStatus;
  result: Record<string, unknown> | null;
  error: string | null;
  created_at: string;
  updated_at: string;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

export async function createJob(
  type: string,
  payload: Record<string, unknown>
): Promise<Job> {
  const db = getDb();
  // FIX #3 (CRITICAL): All values are passed as bind parameters – no raw
  // string concatenation.  The prior version of this file used template
  // literals to build SQL strings, creating injection risk.
  const rows = await db(
    `INSERT INTO jobs (type, payload, status)
     VALUES ($1, $2, 'pending')
     RETURNING id, type, payload, status, result, error, created_at, updated_at`,
    [type, JSON.stringify(payload)]
  );
  return rows[0] as Job;
}

export async function getJob(id: number): Promise<Job | null> {
  const db = getDb();
  const rows = await db(
    `SELECT id, type, payload, status, result, error, created_at, updated_at
     FROM jobs
     WHERE id = $1`,
    [id]
  );
  return (rows[0] as Job) ?? null;
}

export async function getPendingJobs(limit = 10): Promise<Job[]> {
  const db = getDb();
  const rows = await db(
    `SELECT id, type, payload, status, result, error, created_at, updated_at
     FROM jobs
     WHERE status = 'pending'
     ORDER BY created_at ASC
     LIMIT $1`,
    [limit]
  );
  return rows as Job[];
}

export async function updateJobStatus(
  id: number,
  status: JobStatus,
  resultOrError?: { result?: Record<string, unknown>; error?: string }
): Promise<void> {
  const db = getDb();
  await db(
    `UPDATE jobs
     SET status     = $1,
         result     = $2,
         error      = $3,
         updated_at = now()
     WHERE id = $4`,
    [
      status,
      resultOrError?.result ? JSON.stringify(resultOrError.result) : null,
      resultOrError?.error ?? null,
      id,
    ]
  );
}
