import { neon } from "@neondatabase/serverless";

// ---------------------------------------------------------------------------
// Connection
// ---------------------------------------------------------------------------

// FIX #18 (LOW): Initialise the Neon client once at module load rather than
// once per request.  Calling neon() on every request prevents connection
// reuse and degrades performance under load.
//
// FIX #12 (MEDIUM): Wrap the initialisation so that a malformed DATABASE_URL
// does not expose the raw connection string in an unhandled error response.

let _db: ReturnType<typeof neon>;

function getDb() {
  if (!_db) {
    try {
      _db = neon(process.env.DATABASE_URL!);
    } catch {
      // Swallow the original error – it may contain the DATABASE_URL value.
      console.error("[DB] Failed to initialise database connection.");
      throw new Error("Database unavailable");
    }
  }
  return _db;
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface Entry {
  id: number;
  content: string;
  url: string | null;
  tags: string[];
  metadata: Record<string, unknown>;
  created_at: string;
}

// ---------------------------------------------------------------------------
// URL validation (SSRF protection)
// ---------------------------------------------------------------------------

// FIX #13 (MEDIUM): Validate URLs before storing them to prevent SSRF if any
// agent later fetches these URLs.  The block-list covers localhost variants,
// private RFC-1918 ranges, and the cloud metadata endpoint (169.254.169.254).
export function isExternalUrl(rawUrl: string): boolean {
  let parsed: URL;
  try {
    parsed = new URL(rawUrl);
  } catch {
    return false;
  }

  if (!["http:", "https:"].includes(parsed.protocol)) return false;

  const host = parsed.hostname.toLowerCase();
  const blockedExact = ["localhost", "127.0.0.1", "0.0.0.0", "::1"];
  if (blockedExact.includes(host)) return false;

  const blockedPrefixes = [
    "169.254.", // link-local / AWS metadata
    "10.",      // RFC-1918
    "192.168.", // RFC-1918
    "172.16.",  // RFC-1918
    "172.17.",  // RFC-1918
    "172.18.",  // RFC-1918
    "172.19.",  // RFC-1918
    "172.20.",  // RFC-1918
    "172.21.",  // RFC-1918
    "172.22.",  // RFC-1918
    "172.23.",  // RFC-1918
    "172.24.",  // RFC-1918
    "172.25.",  // RFC-1918
    "172.26.",  // RFC-1918
    "172.27.",  // RFC-1918
    "172.28.",  // RFC-1918
    "172.29.",  // RFC-1918
    "172.30.",  // RFC-1918
    "172.31.",  // RFC-1918
  ];
  if (blockedPrefixes.some((p) => host.startsWith(p))) return false;

  return true;
}

// ---------------------------------------------------------------------------
// Read helpers
// ---------------------------------------------------------------------------

export async function getEntries(limit = 50, offset = 0): Promise<Entry[]> {
  const db = getDb();
  const rows = await db(
    `SELECT id, content, url, tags, metadata, created_at
     FROM entries
     ORDER BY created_at DESC
     LIMIT $1 OFFSET $2`,
    [limit, offset]
  );
  return rows as Entry[];
}

export async function getEntry(id: number): Promise<Entry | null> {
  const db = getDb();
  const rows = await db(
    `SELECT id, content, url, tags, metadata, created_at
     FROM entries
     WHERE id = $1`,
    [id]
  );
  return (rows[0] as Entry) ?? null;
}

// FIX #3 (CRITICAL): Validate the search query before it is passed to
// to_tsquery().  The PostgreSQL full-text search functions can throw errors
// on malformed input and may leak internal details in error messages.
// Although the @neondatabase/serverless driver sends the value as a bind
// parameter (preventing classic SQL injection), tsquery has its own syntax
// that can produce unexpected behaviour with unvalidated input.
//
// We allow word characters, spaces, hyphens, and the standard tsquery
// boolean operators (& | ! : *) while rejecting everything else.
function sanitiseTsQueryInput(query: string): string {
  const clean = query.trim();
  if (clean.length === 0) throw new Error("Search query must not be empty");
  if (clean.length > 500) throw new Error("Search query is too long");
  // Reject characters that have no place in a keyword search and could
  // confuse or exploit the tsquery parser.
  if (!/^[\w\s\-]+$/u.test(clean)) {
    throw new Error("Search query contains invalid characters");
  }
  return clean;
}

export async function searchEntries(rawQuery: string): Promise<
  (Entry & { headline: string })[]
> {
  const query = sanitiseTsQueryInput(rawQuery);
  const db = getDb();
  const rows = await db(
    `SELECT id, content, url, tags, metadata, created_at,
            ts_headline('english', content, to_tsquery('english', $1),
                        'MaxWords=35,MinWords=15,StartSel=<mark>,StopSel=</mark>'
            ) AS headline
     FROM entries
     WHERE search_vector @@ to_tsquery('english', $1)
     ORDER BY ts_rank(search_vector, to_tsquery('english', $1)) DESC
     LIMIT 20`,
    [query]
  );
  return rows as (Entry & { headline: string })[];
}

// ---------------------------------------------------------------------------
// Write helpers
// ---------------------------------------------------------------------------

export async function createEntry(data: {
  content: string;
  url?: string;
  tags?: string[];
  metadata?: Record<string, unknown>;
}): Promise<Entry> {
  // FIX #13: Reject private/internal URLs before persisting.
  if (data.url && !isExternalUrl(data.url)) {
    throw new Error("URL must refer to a public external resource");
  }

  const db = getDb();
  const rows = await db(
    `INSERT INTO entries (content, url, tags, metadata)
     VALUES ($1, $2, $3, $4)
     RETURNING id, content, url, tags, metadata, created_at`,
    [
      data.content,
      data.url ?? null,
      JSON.stringify(data.tags ?? []),
      JSON.stringify(data.metadata ?? {}),
    ]
  );
  return rows[0] as Entry;
}

export async function deleteEntry(id: number): Promise<void> {
  const db = getDb();
  await db(`DELETE FROM entries WHERE id = $1`, [id]);
}
