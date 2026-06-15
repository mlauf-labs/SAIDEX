"""Website Categorization Benchmark 03 — Technical API documentation page (English).

Run standalone:
    python benchmarks/website_cat_03.py
"""

from __future__ import annotations

import asyncio
from typing import Literal

from pydantic import BaseModel, Field
from saidex import LanguageCodeStr

from ._base import AtLeast, BenchmarkScenario, run_all_models

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class WebsiteCategory(BaseModel):
    """Classification of a website based on its HTML content."""

    category: Literal["e-commerce", "news", "documentation", "blog", "social-media", "corporate", "other"] = Field(
        description="Primary category of the website"
    )
    subcategory: str = Field(
        description="More specific subcategory, e.g. 'api reference', 'tutorial', 'developer docs', 'user manual'"
    )
    language: LanguageCodeStr = Field(description="Primary language of the page as an ISO 639-1 code, e.g. 'de', 'en', 'fr'")
    target_audience: str = Field(
        description="Brief description of the intended audience"
    )
    has_shopping_cart: bool = Field(description="Whether the page contains a shopping cart or buy functionality")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the classification")


# ---------------------------------------------------------------------------
# Test HTML — developer documentation page
# ---------------------------------------------------------------------------

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Authentication – REST API Reference | DataBridge Developer Docs</title>
  <meta name="description" content="Learn how to authenticate with the DataBridge REST API using API keys and OAuth 2.0 bearer tokens.">
  <link rel="stylesheet" href="/static/docs.css">
  <link rel="canonical" href="https://docs.databridge.io/api/authentication">
</head>
<body class="docs-layout">
  <div class="docs-wrapper">

    <!-- Sidebar Navigation -->
    <nav class="docs-sidebar" aria-label="Documentation navigation">
      <div class="sidebar-header">
        <a href="/"><img src="/img/databridge-logo.svg" alt="DataBridge"></a>
        <span class="version-badge">v3.2</span>
      </div>
      <div class="sidebar-search">
        <input type="search" placeholder="Search docs…" aria-label="Search documentation">
      </div>
      <ul class="sidebar-nav">
        <li class="nav-section">Getting Started</li>
        <li><a href="/docs/quickstart">Quickstart Guide</a></li>
        <li><a href="/docs/sdks">SDK Overview</a></li>
        <li class="nav-section">REST API Reference</li>
        <li class="active"><a href="/docs/api/authentication">Authentication</a></li>
        <li><a href="/docs/api/pagination">Pagination</a></li>
        <li><a href="/docs/api/errors">Error Codes</a></li>
        <li><a href="/docs/api/rate-limits">Rate Limits</a></li>
        <li class="nav-section">Resources</li>
        <li><a href="/docs/api/pipelines">Pipelines</a></li>
        <li><a href="/docs/api/connectors">Connectors</a></li>
        <li><a href="/docs/api/transformations">Transformations</a></li>
        <li><a href="/docs/api/webhooks">Webhooks</a></li>
        <li class="nav-section">Guides</li>
        <li><a href="/docs/guides/streaming">Real-time Streaming</a></li>
        <li><a href="/docs/guides/batch">Batch Processing</a></li>
        <li><a href="/docs/changelog">Changelog</a></li>
      </ul>
    </nav>

    <!-- Main content -->
    <main class="docs-main">
      <div class="docs-content">
        <div class="breadcrumb">
          <a href="/docs">Docs</a> / <a href="/docs/api">REST API</a> / Authentication
        </div>

        <h1>Authentication</h1>
        <p class="lead">The DataBridge API uses API keys and OAuth 2.0 bearer tokens.
        All API requests must be authenticated; unauthenticated requests return
        <code>401 Unauthorized</code>.</p>

        <div class="alert alert-info">
          <strong>Base URL:</strong> <code>https://api.databridge.io/v3</code>
        </div>

        <h2 id="api-keys">API Keys</h2>
        <p>API keys are the simplest authentication method. Generate a key in your
        <a href="https://app.databridge.io/settings/api-keys">Dashboard → Settings → API Keys</a>.</p>

        <p>Pass the key in the <code>Authorization</code> header:</p>

        <div class="code-block">
          <div class="code-header">
            <span>HTTP</span>
            <button class="copy-btn" data-target="ex-apikey">Copy</button>
          </div>
          <pre id="ex-apikey"><code>GET /v3/pipelines HTTP/1.1
Host: api.databridge.io
Authorization: Bearer db_live_sk_4f8a2c1e9b3d7f0e5a2c4d8b1e6f3a9c
Content-Type: application/json</code></pre>
        </div>

        <div class="code-block">
          <div class="code-header">
            <span>Python</span>
            <button class="copy-btn" data-target="ex-python">Copy</button>
          </div>
          <pre id="ex-python"><code>import httpx

API_KEY = "db_live_sk_4f8a2c1e9b3d7f0e5a2c4d8b1e6f3a9c"
headers = {"Authorization": f"Bearer {API_KEY}"}

response = httpx.get("https://api.databridge.io/v3/pipelines", headers=headers)
response.raise_for_status()
print(response.json())</code></pre>
        </div>

        <h2 id="oauth">OAuth 2.0 (Client Credentials)</h2>
        <p>For server-to-server integrations, use the OAuth 2.0 client credentials flow to
        obtain a short-lived access token.</p>

        <h3>Step 1 — Request a token</h3>
        <div class="code-block">
          <div class="code-header"><span>cURL</span><button class="copy-btn">Copy</button></div>
          <pre><code>curl -X POST https://auth.databridge.io/oauth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials" \
  -d "client_id=YOUR_CLIENT_ID" \
  -d "client_secret=YOUR_CLIENT_SECRET" \
  -d "scope=pipelines:read pipelines:write"</code></pre>
        </div>

        <p>Response:</p>
        <div class="code-block">
          <div class="code-header"><span>JSON</span><button class="copy-btn">Copy</button></div>
          <pre><code>{
  "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "scope": "pipelines:read pipelines:write"
}</code></pre>
        </div>

        <h3>Step 2 — Use the token</h3>
        <p>Include the access token in subsequent requests exactly like an API key:</p>
        <div class="code-block">
          <div class="code-header"><span>HTTP</span><button class="copy-btn">Copy</button></div>
          <pre><code>Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...</code></pre>
        </div>

        <h2 id="scopes">Available Scopes</h2>
        <table class="docs-table">
          <thead>
            <tr><th>Scope</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>pipelines:read</code></td><td>List and retrieve pipeline configurations</td></tr>
            <tr><td><code>pipelines:write</code></td><td>Create, update, and delete pipelines</td></tr>
            <tr><td><code>connectors:read</code></td><td>Read connector definitions</td></tr>
            <tr><td><code>connectors:write</code></td><td>Manage connector configurations</td></tr>
            <tr><td><code>events:read</code></td><td>Read pipeline run events and logs</td></tr>
            <tr><td><code>admin</code></td><td>Full account access (use with caution)</td></tr>
          </tbody>
        </table>

        <h2 id="errors">Authentication Errors</h2>
        <table class="docs-table">
          <thead><tr><th>HTTP Status</th><th>Error Code</th><th>Meaning</th></tr></thead>
          <tbody>
            <tr><td>401</td><td><code>invalid_token</code></td><td>Token is missing, malformed, or expired</td></tr>
            <tr><td>401</td><td><code>token_expired</code></td><td>OAuth access token has expired; refresh it</td></tr>
            <tr><td>403</td><td><code>insufficient_scope</code></td><td>Token lacks the required scope for this endpoint</td></tr>
          </tbody>
        </table>

        <div class="docs-footer-nav">
          <a href="/docs/quickstart" class="prev">&larr; Quickstart</a>
          <a href="/docs/api/pagination" class="next">Pagination &rarr;</a>
        </div>
      </div>

      <div class="docs-toc">
        <p>On this page</p>
        <ul>
          <li><a href="#api-keys">API Keys</a></li>
          <li><a href="#oauth">OAuth 2.0</a></li>
          <li><a href="#scopes">Available Scopes</a></li>
          <li><a href="#errors">Authentication Errors</a></li>
        </ul>
      </div>
    </main>
  </div>

  <footer class="docs-global-footer">
    <a href="https://databridge.io">databridge.io</a>
    <a href="/docs/changelog">Changelog</a>
    <a href="https://status.databridge.io">Status</a>
    <a href="mailto:support@databridge.io">Support</a>
    <span>&copy; 2024 DataBridge Inc.</span>
  </footer>
</body>
</html>"""


SCENARIO = BenchmarkScenario(
    name="Website Cat 03 — API Documentation (EN)",
    description="HTML of an English developer documentation page with API reference, code examples, and sidebar navigation.",
    schema=WebsiteCategory,
    text=HTML_CONTENT,
    system_prompt="Analyze the following HTML page content and classify the website. Focus on the structure, navigation, and content clues.",
    expected={
        "category": "documentation",
        "language": "en",
        "has_shopping_cart": False,
        "confidence": AtLeast(0.7),
    },
)


async def main() -> None:
    print("Benchmark: Website Cat 03 — API Documentation (EN)")
    await run_all_models(SCENARIO)


if __name__ == "__main__":
    asyncio.run(main())
