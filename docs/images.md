# Image Extraction (Multimodal)

→ [Documentation index](index.md)

> **Runnable example:** [`examples/05_image_analysis.py`](../examples/05_image_analysis.py)

`saidex` works with any vision-capable LLM.  Images are passed
as part of the LangChain message list using the standard multimodal
`HumanMessage` format — no library-specific API is needed.

The example file covers four scenarios:
- **A** — extract from an image URL with OpenAI GPT-4o
- **B** — extract from a local file (base64-encoded) with OpenAI GPT-4o
- **C** — extract from an image URL with a self-hosted vLLM vision model
- **D** — extract from multiple images combined into one schema

---

## Prerequisites

```bash
pip install langchain-openai       # for OpenAI / Azure OpenAI
```

For a local vLLM vision model:

```bash
pip install vllm
vllm serve Qwen/Qwen2-VL-7B-Instruct --port 8000
```

---

## Helper functions

The example file provides three helper functions for building multimodal
messages.  Copy them directly into your project:

### `image_message_from_url`

Creates a `HumanMessage` from a publicly accessible HTTPS image URL.

```python
from langchain_core.messages import HumanMessage

def image_message_from_url(
    url: str,
    prompt: str = "Analyse this image and extract the required information.",
    *,
    detail: str = "high",   # "low" | "high" | "auto" (OpenAI-specific)
) -> HumanMessage:
    return HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": url, "detail": detail}},
        ]
    )
```

### `image_message_from_file`

Reads a local image file, base64-encodes it, and creates a `HumanMessage`.
Works with JPEG, PNG, GIF, and WebP.

```python
import base64
from pathlib import Path
from langchain_core.messages import HumanMessage

def image_message_from_file(
    path: str | Path,
    prompt: str = "Analyse this image and extract the required information.",
    *,
    media_type: str | None = None,   # inferred from extension when None
) -> HumanMessage:
    path = Path(path)
    suffix_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    mime = media_type or suffix_map.get(path.suffix.lower(), "image/jpeg")
    data_url = f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"
    return HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
    )
```

### `multi_image_message`

Combines multiple images (URL or file) and a single prompt into one message:

```python
from pathlib import Path
from langchain_core.messages import HumanMessage

def multi_image_message(
    images: list[str | Path],
    prompt: str = "Analyse all the images and extract the required information.",
) -> HumanMessage:
    content: list[dict] = [{"type": "text", "text": prompt}]
    for image in images:
        if isinstance(image, str) and image.startswith("http"):
            content.append({"type": "image_url", "image_url": {"url": image}})
        else:
            path = Path(image)
            data_url = f"data:image/jpeg;base64,{base64.b64encode(path.read_bytes()).decode()}"
            content.append({"type": "image_url", "image_url": {"url": data_url}})
    return HumanMessage(content=content)
```

---

## Scenario A — extract from an image URL (OpenAI)

```python
import asyncio
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from saidex import get_structured_data

class ProductLabel(BaseModel):
    product_name: str  = Field(description="Full product name as printed on the label")
    brand:        str  = Field(description="Brand or manufacturer name")
    weight_grams: int | None = Field(default=None, description="Net weight in grams, or null")
    ingredients:  list[str]  = Field(default_factory=list, description="List of ingredients")
    allergens:    list[str]  = Field(default_factory=list, description="Allergens listed")

async def extract_from_image_url(url: str) -> ProductLabel | None:
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    messages = [
        SystemMessage(content="Extract product information from the label in the image."),
        image_message_from_url(url, prompt="Extract data from this product label."),
    ]

    result, stats = await get_structured_data(llm, ProductLabel, messages)
    print(f"Retries: {stats.total_retries}")
    return result

asyncio.run(extract_from_image_url("https://example.com/product-label.jpg"))
```

---

## Scenario B — extract from a local file (base64)

```python
async def extract_from_local_file(path: str) -> ProductLabel | None:
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    messages = [
        SystemMessage(content="Extract product information from the label."),
        image_message_from_file(path, prompt="Extract data from this product label."),
    ]

    result, stats = await get_structured_data(llm, ProductLabel, messages)
    return result

asyncio.run(extract_from_local_file("./product.jpg"))
```

---

## Scenario C — self-hosted vLLM vision model

`saidex` works with any OpenAI-compatible vision server,
including locally hosted models via [vLLM](https://docs.vllm.ai/).

```bash
# Start the vLLM server
pip install vllm
vllm serve Qwen/Qwen2-VL-7B-Instruct \
    --port 8000 \
    --trust-remote-code \
    --max-model-len 4096
```

```python
from langchain_openai import ChatOpenAI

def create_vllm_vision_client(
    model: str = "Qwen/Qwen2-VL-7B-Instruct",
    base_url: str = "http://localhost:8000/v1",
    temperature: float = 0.0,
) -> ChatOpenAI:
    """Create a LangChain client pointing at a local vLLM server."""
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key="not-needed",   # vLLM does not check the API key
        temperature=temperature,
        max_tokens=1024,
    )

async def extract_with_vllm(url: str) -> ProductLabel | None:
    llm = create_vllm_vision_client()

    messages = [
        SystemMessage(content="Extract product information from the image."),
        image_message_from_url(url, prompt="Extract data from this product label."),
    ]

    result, stats = await get_structured_data(llm, ProductLabel, messages)
    return result
```

> **Model recommendations for local vision extraction:**
> - `Qwen/Qwen2-VL-7B-Instruct` — good balance of speed and accuracy
> - `Qwen/Qwen2-VL-72B-Instruct` — highest accuracy, requires more VRAM
> - `microsoft/Phi-3.5-vision-instruct` — small footprint, fast inference

---

## Scenario D — multiple images combined

Pass multiple images in one message to extract a combined schema from all
of them at once:

```python
class ComparativeAnalysis(BaseModel):
    products: list[str] = Field(description="Product names found across all images")
    common_ingredients: list[str] = Field(
        default_factory=list,
        description="Ingredients present in all products",
    )
    unique_allergens: list[str] = Field(
        default_factory=list,
        description="All allergens found across all products (deduplicated)",
    )
    price_comparison: str | None = Field(
        default=None,
        description="Summary of price comparison if prices are visible",
    )

async def compare_products(image_paths: list[str]) -> ComparativeAnalysis | None:
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    messages = [
        SystemMessage(content="Analyse all images and provide a comparative analysis."),
        multi_image_message(
            image_paths,
            prompt="Compare these products and fill in the comparison schema.",
        ),
    ]

    result, stats = await get_structured_data(llm, ComparativeAnalysis, messages)
    return result

asyncio.run(compare_products(["product_a.jpg", "product_b.jpg", "product_c.jpg"]))
```

---

## Practical tips

### Use `detail="low"` to reduce cost

For large images where fine detail is not needed:

```python
msg = image_message_from_url(url, detail="low")
```

OpenAI charges significantly less for `detail="low"`.  Use `"high"` only
when the schema requires reading small text (serial numbers, ingredient lists,
fine print).

### Add a fallback for blurry images

A more capable model as fallback is especially useful for vision tasks where
image quality varies:

```python
primary  = ChatOpenAI(model="gpt-4o-mini", temperature=0)
fallback = ChatOpenAI(model="gpt-4o",      temperature=0)

result, stats = await get_structured_data(
    primary, ProductLabel, messages,
    fallback_llm_model=fallback,
    max_primary_retries=1,    # fail fast on the small model
    max_fallback_retries=3,
)
```

### Make fields optional for partially visible content

When the image might not show all fields (e.g. a label is partially obscured):

```python
class ProductLabel(BaseModel):
    product_name: str  = Field(description="Full product name")
    brand:        str  = Field(description="Brand name")
    # These might not be visible in every image:
    weight_grams: int | None  = Field(default=None, description="Net weight in grams or null if not visible")
    price:        float | None = Field(default=None, description="Price in EUR or null if not visible")
    barcode:      str | None  = Field(default=None, description="EAN/UPC barcode or null if not visible")
```

---

## Related

- [`examples/05_image_analysis.py`](../examples/05_image_analysis.py) — complete runnable example (all four scenarios)
- [Extraction](extraction.md) — how to build and pass message lists
- [Retry & Fallback](retry-and-fallback.md) — improving reliability for low-quality images
- [Schema Design](schema-design.md) — designing schemas for vision extraction
