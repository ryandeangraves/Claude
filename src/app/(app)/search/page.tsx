import { redirect } from "next/navigation";
import DOMPurify from "isomorphic-dompurify";
import { isAuthenticated } from "@/lib/auth";
import { searchEntries } from "@/lib/db";

// ---------------------------------------------------------------------------
// Headline sanitisation
// ---------------------------------------------------------------------------

// FIX #7 (HIGH): Replace the regex-based "sanitiser" with DOMPurify.
//
// The previous implementation used a regex to strip tags other than <mark>:
//
//   function sanitizeHeadline(html: string) {
//     return html.replace(/<(?!\/?(mark)(\s[^>]*)?>)[^>]+>/gi, "");
//   }
//
// This approach fails to strip event-handler attributes injected alongside
// allowed tags (e.g. <mark onmouseover="alert(1)">).  DOMPurify correctly
// handles attribute-level injection by allowing only the <mark> element with
// no attributes.
function sanitizeHeadline(html: string): string {
  return DOMPurify.sanitize(html, {
    ALLOWED_TAGS: ["mark"],
    ALLOWED_ATTR: [],
  });
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

interface SearchParams {
  q?: string;
}

export default async function SearchPage({
  searchParams,
}: {
  searchParams: SearchParams;
}) {
  // FIX #6 (HIGH): Require authentication before serving any search results.
  //
  // The previous version had no auth check on this page.  Even if Next.js
  // middleware is configured to protect the route, a middleware bypass
  // (CVE-2025-29927) would expose the full search index without this
  // in-component guard.
  if (!(await isAuthenticated())) {
    redirect("/login");
  }

  const rawQuery = searchParams.q?.trim() ?? "";
  const results =
    rawQuery.length > 0
      ? await searchEntries(rawQuery).catch(() => [])
      : [];

  return (
    <main className="p-6 max-w-3xl mx-auto">
      <h1 className="text-2xl font-semibold mb-4">Search</h1>

      <form method="get">
        <input
          type="search"
          name="q"
          defaultValue={rawQuery}
          placeholder="Search your brain…"
          className="w-full border rounded px-3 py-2 mb-6"
          autoFocus
        />
      </form>

      {rawQuery && results.length === 0 && (
        <p className="text-gray-500">No results for &ldquo;{rawQuery}&rdquo;.</p>
      )}

      <ul className="space-y-4">
        {results.map((entry) => (
          <li key={entry.id} className="border rounded p-4">
            {/* FIX #7: headline is sanitised through DOMPurify before render. */}
            <p
              className="text-sm text-gray-700"
              dangerouslySetInnerHTML={{
                __html: sanitizeHeadline(entry.headline),
              }}
            />
            <div className="mt-2 flex gap-2 flex-wrap">
              {entry.tags.map((tag) => (
                <span
                  key={tag}
                  className="bg-gray-100 text-gray-600 text-xs px-2 py-0.5 rounded"
                >
                  {tag}
                </span>
              ))}
            </div>
            {entry.url && (
              <a
                href={entry.url}
                target="_blank"
                rel="noopener noreferrer"
                className="mt-2 text-xs text-blue-500 truncate block"
              >
                {entry.url}
              </a>
            )}
          </li>
        ))}
      </ul>
    </main>
  );
}
