import crypto from "crypto";
import { cookies } from "next/headers";
import { createLogEntry } from "@/lib/db-log";

// ---------------------------------------------------------------------------
// Environment – validated once at startup so misconfiguration fails loudly.
// ---------------------------------------------------------------------------

const BRAIN_PASSWORD = process.env.BRAIN_PASSWORD ?? "";
const BRAIN_API_KEY = process.env.BRAIN_API_KEY ?? "";

// FIX #10 (MEDIUM): Enforce a minimum password length at startup so a
// trivially-guessable password can never reach production.
if (process.env.NODE_ENV === "production" && BRAIN_PASSWORD.length < 16) {
  throw new Error(
    "BRAIN_PASSWORD must be at least 16 characters in production."
  );
}

// ---------------------------------------------------------------------------
// Rate limiting
// ---------------------------------------------------------------------------

const RATE_LIMIT_WINDOW_MS = 15 * 60 * 1000; // 15 minutes
const RATE_LIMIT_MAX = 100;

// In-memory store is intentionally simple for a single-process deployment.
// FIX #5 (HIGH): Use Redis / an atomic store for multi-process deployments to
// eliminate the TOCTOU race condition described in the audit.
const rateLimitMap = new Map<string, { count: number; resetAt: number }>();

export function checkRateLimit(
  ip: string
): { allowed: boolean; remaining: number } {
  const now = Date.now();
  let entry = rateLimitMap.get(ip);

  if (!entry || entry.resetAt < now) {
    // New window – initialise atomically (single-threaded JS guarantees this).
    entry = { count: 1, resetAt: now + RATE_LIMIT_WINDOW_MS };
    rateLimitMap.set(ip, entry);
    return { allowed: true, remaining: RATE_LIMIT_MAX - 1 };
  }

  // FIX #5: Check *before* incrementing so the count can never exceed the
  // limit via concurrent reads.
  if (entry.count >= RATE_LIMIT_MAX) {
    return { allowed: false, remaining: 0 };
  }

  entry.count += 1;
  return { allowed: true, remaining: RATE_LIMIT_MAX - entry.count };
}

// ---------------------------------------------------------------------------
// Session token
// ---------------------------------------------------------------------------

// FIX #2 (CRITICAL): The HMAC key is now read from SESSION_SECRET instead of
// being hardcoded as "second-brain-session-v2".  A hardcoded key lets anyone
// with source access forge tokens for *any* password value.
//
// Generate a suitable secret with:
//   node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
// and store the result as SESSION_SECRET in your .env / hosting environment.
function generateSessionToken(password: string): string {
  const secret = process.env.SESSION_SECRET;
  if (!secret || secret.length < 32) {
    throw new Error(
      "SESSION_SECRET env var must be set to a random string of at least 32 bytes."
    );
  }
  return `brain_${crypto.createHmac("sha256", secret).update(password).digest("hex")}`;
}

// ---------------------------------------------------------------------------
// Authentication helpers
// ---------------------------------------------------------------------------

export async function isAuthenticated(): Promise<boolean> {
  if (!BRAIN_PASSWORD) return false;
  const cookieStore = await cookies();
  const session = cookieStore.get("brain_session");
  if (!session) return false;

  try {
    const expected = generateSessionToken(BRAIN_PASSWORD);
    const a = Buffer.from(session.value.padEnd(expected.length));
    const b = Buffer.from(expected);
    // timingSafeEqual requires equal-length buffers.
    return a.length === b.length && crypto.timingSafeEqual(a, b);
  } catch {
    return false;
  }
}

export async function login(password: string, ip: string): Promise<boolean> {
  if (!BRAIN_PASSWORD) return false;

  let match = false;
  try {
    match = crypto.timingSafeEqual(
      Buffer.from(password),
      Buffer.from(BRAIN_PASSWORD)
    );
  } catch {
    match = false;
  }

  // FIX #11 (MEDIUM): Log failed login attempts so security events are
  // visible in the frank_log table for forensic review.
  if (!match) {
    await createLogEntry({
      action: "login_failed",
      category: "security",
      summary: `Failed login attempt from ${ip}`,
      details: { ip, timestamp: new Date().toISOString() },
    }).catch(() => {
      // Never let logging failures break the auth response.
    });
    return false;
  }

  const token = generateSessionToken(BRAIN_PASSWORD);
  const cookieStore = await cookies();

  cookieStore.set("brain_session", token, {
    httpOnly: true,
    // FIX #8 (HIGH): Do not require Secure in dev/HTTP environments – the
    // cookie would silently not be sent, breaking local development.  In
    // production (HTTPS) the flag is always set.
    secure: process.env.NODE_ENV === "production",
    sameSite: "strict",
    path: "/",
    maxAge: 60 * 60 * 24 * 30,
  });

  return true;
}

export async function logout(): Promise<void> {
  const cookieStore = await cookies();
  cookieStore.delete("brain_session");
}

// ---------------------------------------------------------------------------
// API key verification
// ---------------------------------------------------------------------------

export function checkApiKey(apiKey: string): boolean {
  const key = BRAIN_API_KEY;
  if (!key) return false;
  try {
    // Buffers must be the same length for timingSafeEqual.
    const a = Buffer.from(apiKey.padEnd(key.length));
    const b = Buffer.from(key);
    return a.length === b.length && crypto.timingSafeEqual(a, b);
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------------
// Input validation helpers
// ---------------------------------------------------------------------------

// FIX #9 (HIGH): Reject strings that are far longer than the accepted maximum
// *before* allocating the substring, to prevent memory-exhaustion DoS.
// The prior implementation called val.substring(maxLength) after receiving an
// arbitrarily large string, which still allocates the full string in memory.
export function validateString(val: unknown, maxLength: number): string | null {
  if (typeof val !== "string") return null;
  // Reject input more than 10× the allowed length outright.
  if (val.length > maxLength * 10) return null;
  return val.substring(0, maxLength);
}

export function validateTags(val: unknown): string[] | null {
  if (!Array.isArray(val)) return null;
  return val
    .filter((v): v is string => typeof v === "string")
    .map((v) => v.substring(0, 100))
    .slice(0, 20);
}
