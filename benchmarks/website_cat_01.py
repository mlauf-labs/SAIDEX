"""Website Categorization Benchmark 01 — E-Commerce product page (German online shop).

Run standalone:
    python benchmarks/website_cat_01.py
"""

from __future__ import annotations

import asyncio
from typing import Literal

from pydantic import BaseModel, Field

from ._base import BenchmarkScenario, run_all_models

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class WebsiteCategory(BaseModel):
    """Classification of a website based on its HTML content."""

    category: Literal["e-commerce", "news", "documentation", "blog", "social-media", "corporate", "other"] = Field(
        description="Primary category of the website"
    )
    subcategory: str | None = Field(
        None,
        description="More specific subcategory, e.g. 'fashion', 'electronics', 'sports', 'tech news'"
    )
    language: str = Field(description="Primary language of the page, e.g. 'de', 'en', 'fr'")
    target_audience: str = Field(
        description="Brief description of the intended audience, e.g. 'online shoppers', 'developers', 'general public'"
    )
    has_shopping_cart: bool = Field(description="Whether the page contains a shopping cart or buy functionality")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the classification")


# ---------------------------------------------------------------------------
# Test HTML — e-commerce product page
# ---------------------------------------------------------------------------

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Nike Air Zoom Pegasus 41 Running Shoes – Men's | SportWorld24</title>
  <meta name="description" content="Buy Nike Air Zoom Pegasus 41 men's running shoes. Free shipping over £49, 30-day returns, over 500 customer reviews.">
  <link rel="stylesheet" href="/static/css/shop.css">
</head>
<body>
  <header>
    <nav class="main-nav">
      <a href="/" class="logo"><img src="/img/sportworld24-logo.svg" alt="SportWorld24"></a>
      <ul>
        <li><a href="/running">Running</a></li>
        <li><a href="/fitness">Fitness</a></li>
        <li><a href="/outdoor">Outdoor</a></li>
        <li><a href="/sale">SALE</a></li>
      </ul>
      <div class="nav-actions">
        <a href="/search"><i class="icon-search"></i></a>
        <a href="/wishlist"><i class="icon-heart"></i> <span>3</span></a>
        <a href="/cart" class="cart-icon"><i class="icon-cart"></i> <span class="cart-count">2</span></a>
        <a href="/account">My Account</a>
      </div>
    </nav>
    <div class="breadcrumb">
      <a href="/">Home</a> &raquo; <a href="/running">Running</a> &raquo;
      <a href="/running/shoes">Men's Running Shoes</a> &raquo; Nike Air Zoom Pegasus 41
    </div>
  </header>

  <main class="product-page">
    <div class="product-gallery">
      <img src="/img/products/nike-pegasus-41-main.jpg" alt="Nike Air Zoom Pegasus 41 – Side view" class="main-image">
      <div class="thumbnails">
        <img src="/img/products/nike-pegasus-41-front.jpg" alt="Front view">
        <img src="/img/products/nike-pegasus-41-sole.jpg" alt="Outsole">
        <img src="/img/products/nike-pegasus-41-detail.jpg" alt="ReactX foam detail">
      </div>
    </div>

    <div class="product-info">
      <span class="brand">Nike</span>
      <h1 class="product-title">Air Zoom Pegasus 41 – Men's Running Shoe</h1>
      <div class="product-rating">
        <span class="stars">★★★★☆</span>
        <span class="rating-count">(487 reviews)</span>
        <a href="#reviews">Read all reviews</a>
      </div>

      <div class="price-box">
        <span class="price-old">£ 139.95</span>
        <span class="price-current">£ 119.95</span>
        <span class="price-badge">-14%</span>
      </div>
      <p class="availability in-stock"><i class="icon-check"></i> In Stock — Ships within 1–2 business days</p>

      <div class="variant-selector">
        <label>Select size:</label>
        <div class="size-grid">
          <button class="size-btn sold-out" disabled>6</button>
          <button class="size-btn">7</button>
          <button class="size-btn selected">8</button>
          <button class="size-btn">9</button>
          <button class="size-btn">10</button>
          <button class="size-btn sold-out" disabled>11</button>
          <button class="size-btn">12</button>
        </div>
      </div>

      <div class="color-selector">
        <label>Colour: <strong>Midnight Navy / White</strong></label>
        <div class="color-swatches">
          <span class="swatch selected" style="background:#1a237e" title="Midnight Navy"></span>
          <span class="swatch" style="background:#222" title="Black/Anthracite"></span>
          <span class="swatch" style="background:#e53935" title="University Red"></span>
        </div>
      </div>

      <div class="cta-buttons">
        <button class="btn-primary btn-add-to-cart" data-product-id="NK-PEG41-42-NVY">
          <i class="icon-cart"></i> Add to Cart
        </button>
        <button class="btn-secondary btn-buy-now">Buy Now</button>
        <button class="btn-wishlist" aria-label="Add to wishlist"><i class="icon-heart-outline"></i></button>
      </div>

      <div class="trust-badges">
        <span><i class="icon-truck"></i> Free shipping over £49</span>
        <span><i class="icon-return"></i> 30-day free returns</span>
        <span><i class="icon-lock"></i> Secure checkout: PayPal, Credit Card, Apple Pay</span>
      </div>
    </div>

    <div class="product-details">
      <h2>Product Description</h2>
      <p>The Nike Air Zoom Pegasus 41 continues the legendary Pegasus lineage with the new
      ReactX foam, which delivers 13% more energy return compared to its predecessor.
      The redesigned upper made from recycled fibres provides a comfortable fit, while
      the wider forefoot area offers more room for natural toe splay.</p>

      <ul class="features">
        <li>ReactX foam for improved cushioning and energy return</li>
        <li>Zoom Air unit in the forefoot for responsiveness at push-off</li>
        <li>Engineered Mesh upper — breathable and lightweight</li>
        <li>Waffle outsole for traction on various surfaces</li>
        <li>Weight: approx. 283 g (size UK 8)</li>
        <li>Drop: 10 mm</li>
        <li>Suitable for: neutral to mild overpronators, training runs 5–15 km</li>
      </ul>

      <table class="specs-table">
        <tr><th>SKU</th><td>NK-PEG41-42-NVY</td></tr>
        <tr><th>Brand</th><td>Nike</td></tr>
        <tr><th>Collection</th><td>Spring/Summer 2024</td></tr>
        <tr><th>Upper material</th><td>min. 20% recycled fibres</td></tr>
      </table>
    </div>

    <section id="reviews" class="reviews">
      <h2>Customer Reviews (487)</h2>
      <div class="review">
        <strong>MaxRuns</strong> – ★★★★★
        <p>My go-to shoe for long runs for years. The 41 version is noticeably
        softer and more responsive. Highly recommended!</p>
      </div>
      <div class="review">
        <strong>TrailRunner_UK</strong> – ★★★★☆
        <p>Great everyday shoe, fits well. Only the forefoot area could be slightly
        more generous for my wider feet.</p>
      </div>
      <a href="/reviews/nike-air-zoom-pegasus-41" class="btn-link">View all 487 reviews</a>
    </section>
  </main>

  <footer>
    <nav>
      <a href="/legal">Legal Notice</a>
      <a href="/privacy">Privacy Policy</a>
      <a href="/terms">Terms & Conditions</a>
      <a href="/contact">Contact</a>
    </nav>
    <p>&copy; 2024 SportWorld24 Ltd, London</p>
  </footer>
</body>
</html>"""


SCENARIO = BenchmarkScenario(
    name="Website Cat 01 — E-Commerce Product Page",
    description="HTML of an online shop product page with shopping cart, prices, and product details.",
    schema=WebsiteCategory,
    text=HTML_CONTENT,
    system_prompt="Analyze the following HTML page content and classify the website. Focus on the structure, navigation, and content clues.",
    expected={"category": "e-commerce", "language": "en", "has_shopping_cart": True},
)


async def main() -> None:
    print("Benchmark: Website Cat 01 — E-Commerce Product Page")
    await run_all_models(SCENARIO)


if __name__ == "__main__":
    asyncio.run(main())
