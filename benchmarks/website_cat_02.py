"""Website Categorization Benchmark 02 — News portal article page (German).

Run standalone:
    python benchmarks/website_cat_02.py
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
        description="More specific subcategory, e.g. 'fashion', 'electronics', 'sports', 'tech news', 'politics'"
    )
    language: LanguageCodeStr = Field(description="Primary language of the page as an ISO 639-1 code, e.g. 'de', 'en', 'fr'")
    target_audience: str = Field(
        description="Brief description of the intended audience"
    )
    has_shopping_cart: bool = Field(description="Whether the page contains a shopping cart or buy functionality")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the classification")


# ---------------------------------------------------------------------------
# Test HTML — news portal article
# ---------------------------------------------------------------------------

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AI Regulation: European Parliament Passes Final AI Act Package | DailyTech.com</title>
  <meta name="description" content="The European Parliament has today passed the final AI Act regulatory package. What does this mean for businesses worldwide?">
  <meta property="og:type" content="article">
  <meta property="article:published_time" content="2024-03-12T09:15:00+01:00">
  <meta property="article:author" content="Dr. Lena Schreiber">
  <meta property="article:section" content="Technology">
</head>
<body>
  <header class="site-header">
    <div class="header-top">
      <a href="/" class="site-logo">
        <img src="/img/dailytech-logo.svg" alt="DailyTech.com">
      </a>
      <div class="header-meta">
        <span class="date">Tuesday, 12 March 2024</span>
        <div class="edition-selector">
          <select><option>Global Edition</option><option>US Edition</option><option>UK Edition</option></select>
        </div>
        <a href="/newsletter" class="btn-subscribe">Subscribe to newsletter</a>
        <a href="/login" class="btn-login">Sign in</a>
      </div>
    </div>
    <nav class="main-nav">
      <ul>
        <li><a href="/politics">Politics</a></li>
        <li><a href="/business">Business</a></li>
        <li class="active"><a href="/technology">Technology</a></li>
        <li><a href="/science">Science</a></li>
        <li><a href="/culture">Culture</a></li>
        <li><a href="/sport">Sport</a></li>
        <li><a href="/opinion">Opinion</a></li>
        <li><a href="/videos">Videos</a></li>
      </ul>
      <div class="breaking-ticker">
        <span class="breaking-label">BREAKING</span>
        <span>S&amp;P 500 closes at record high &bull; Election forecasts: tight race in three swing states</span>
      </div>
    </nav>
  </header>

  <div class="layout-content">
    <main class="article-main">
      <article itemscope itemtype="http://schema.org/NewsArticle">
        <div class="article-header">
          <div class="category-tag"><a href="/technology">Technology</a> &rsaquo; <a href="/technology/ai">Artificial Intelligence</a></div>
          <h1 itemprop="headline">EU AI Regulation: Parliament Passes the AI Act —
            What Businesses Need to Know Now</h1>
          <p class="article-deck">The European Parliament voted today with a large majority
          in favour of the final AI Act package. Companies deploying so-called "high-risk AI"
          are most affected. Here is an overview of the most important changes.</p>

          <div class="article-meta">
            <span class="author" itemprop="author">By <strong>Dr. Lena Schreiber</strong>, Technology Editor</span>
            <time itemprop="datePublished" datetime="2024-03-12T09:15:00">12 March 2024, 09:15</time>
            <span class="reading-time">Reading time: approx. 5 minutes</span>
          </div>

          <div class="social-share">
            <button class="share-btn share-twitter" aria-label="Share on X">X</button>
            <button class="share-btn share-facebook" aria-label="Share on Facebook">f</button>
            <button class="share-btn share-linkedin" aria-label="Share on LinkedIn">in</button>
            <button class="share-btn share-copy" aria-label="Copy link">🔗</button>
          </div>
        </div>

        <figure class="article-hero">
          <img src="/img/articles/eu-parliament-ai-act.jpg" alt="European Parliament plenary chamber">
          <figcaption>The European Parliament in Strasbourg voted 523 in favour of the AI Act.
          (Photo: EU Parliament / CC BY 4.0)</figcaption>
        </figure>

        <div class="article-body" itemprop="articleBody">
          <p><strong>Strasbourg.</strong> With a clear majority of 523 votes to 46, with
          37 abstentions, the European Parliament today adopted the AI Act in its final form.
          The regulation is the first comprehensive legal framework for artificial intelligence
          worldwide and is set to come into force in stages from 2026.</p>

          <h2>What does the AI Act regulate?</h2>
          <p>The framework classifies AI systems by risk level into four categories:
          minimal risk, limited risk, high risk, and unacceptable risk. For high-risk
          applications — such as in medicine, law enforcement, or automated credit decisions —
          particularly strict requirements apply regarding transparency, human oversight,
          and technical documentation.</p>

          <blockquote>
            "The AI Act is a milestone. We are setting global standards for a technology
            that will fundamentally transform our society."
            <cite>— Dragos Tudorache, European Parliament rapporteur</cite>
          </blockquote>

          <h2>What must companies do now?</h2>
          <p>Companies developing or deploying AI systems must complete a conformity
          assessment and provide corresponding documentation by 2026 at the latest. For
          prohibited AI practices — such as social scoring by public authorities or
          biometric mass surveillance in public spaces — an immediate ban applies
          as early as summer 2024.</p>

          <p>Experts estimate that small and medium-sized enterprises in particular face
          significant compliance costs. The tech industry association ITI is calling for
          practical guidelines and transition periods for smaller businesses.</p>
        </div>

        <div class="article-tags">
          <span>Tags:</span>
          <a href="/topics/eu">#EU</a>
          <a href="/topics/artificial-intelligence">#ArtificialIntelligence</a>
          <a href="/topics/regulation">#Regulation</a>
          <a href="/topics/ai-act">#AIAct</a>
        </div>
      </article>

      <section class="related-articles">
        <h3>More on this topic</h3>
        <ul>
          <li><a href="/technology/ai/chatgpt-gdpr-compliance-2024">ChatGPT and GDPR: What companies need to know</a></li>
          <li><a href="/business/digitalisation/ai-investment-europe">European firms to invest €12bn in AI by 2026</a></li>
        </ul>
      </section>
    </main>

    <aside class="sidebar">
      <section class="ad-slot"><div class="advertisement">Advertisement</div></section>
      <section class="most-read">
        <h3>Most read</h3>
        <ol>
          <li><a href="/business/inflation-april-2024">Inflation falls to 2.3 percent</a></li>
          <li><a href="/politics/coalition-budget-2025">Coalition dispute over 2025 budget</a></li>
          <li><a href="/technology/ai/eu-ai-act-passed">AI Act adopted</a></li>
        </ol>
      </section>
      <section class="newsletter-signup">
        <h3>Stay informed daily</h3>
        <p>Our free morning newsletter straight to your inbox.</p>
        <form action="/newsletter/subscribe" method="post">
          <input type="email" placeholder="Your email address" name="email">
          <button type="submit">Subscribe</button>
        </form>
      </section>
    </aside>
  </div>

  <footer>
    <nav class="footer-nav">
      <a href="/legal">Legal Notice</a>
      <a href="/privacy">Privacy Policy</a>
      <a href="/about">About the Newsroom</a>
      <a href="/media">Media Kit</a>
      <a href="/contact">Contact</a>
    </nav>
    <p>&copy; 2024 DailyTech Publishing Ltd</p>
  </footer>
</body>
</html>"""


SCENARIO = BenchmarkScenario(
    name="Website Cat 02 — News Portal Article",
    description="HTML of a news portal article page with Schema.org metadata, author byline, and category navigation.",
    schema=WebsiteCategory,
    text=HTML_CONTENT,
    system_prompt="Analyze the following HTML page content and classify the website. Focus on the structure, navigation, and content clues.",
    expected={
        "category": "news",
        "language": "en",
        "has_shopping_cart": False,
        "confidence": AtLeast(0.7),
    },
)


async def main() -> None:
    print("Benchmark: Website Cat 02 — News Portal Article")
    await run_all_models(SCENARIO)


if __name__ == "__main__":
    asyncio.run(main())
