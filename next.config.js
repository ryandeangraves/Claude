/** @type {import('next').NextConfig} */
const nextConfig = {
  // ---------------------------------------------------------------------------
  // FIX #15 (LOW): Content Security Policy headers
  // ---------------------------------------------------------------------------
  // A strong CSP is the primary defence-in-depth control against XSS.  Even if
  // a sanitisation step is bypassed, the browser will refuse to execute
  // injected scripts when a restrictive CSP is in place.
  //
  // Adjust `script-src` if you load scripts from trusted third-party origins
  // (e.g. analytics, fonts).  Remove 'unsafe-inline' from script-src once you
  // migrate any inline scripts to external files with nonces.
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          {
            key: "Content-Security-Policy",
            value: [
              "default-src 'self'",
              // 'unsafe-inline' required for Next.js App Router inline styles;
              // remove once all styles are external.
              "style-src 'self' 'unsafe-inline'",
              // No inline scripts – Next.js injects nonces automatically when
              // you set a nonce-based CSP (see Next.js docs for nonce setup).
              "script-src 'self'",
              "img-src 'self' data: https:",
              "font-src 'self'",
              "connect-src 'self'",
              "frame-ancestors 'none'",
              "base-uri 'self'",
              "form-action 'self'",
            ].join("; "),
          },
          // Prevent browsers from MIME-sniffing responses away from declared
          // content types, reducing the risk of drive-by download attacks.
          { key: "X-Content-Type-Options", value: "nosniff" },
          // Deny framing to mitigate clickjacking.
          { key: "X-Frame-Options", value: "DENY" },
          // Enforce HTTPS for one year, including sub-domains.
          {
            key: "Strict-Transport-Security",
            value: "max-age=31536000; includeSubDomains",
          },
          // FIX #8: Force HTTPS redirects in production.
          // (Handled at the HTTP server / hosting layer, not in Next.js config,
          // but the HSTS header above instructs browsers to upgrade future
          // requests automatically.)
        ],
      },
    ];
  },

  // ---------------------------------------------------------------------------
  // Recommended hardening
  // ---------------------------------------------------------------------------
  poweredByHeader: false, // Remove the X-Powered-By: Next.js header.
};

module.exports = nextConfig;
