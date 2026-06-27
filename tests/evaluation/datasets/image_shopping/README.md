# Image Shopping Dataset

Synthetic product-photo assets for image-shopping evaluations.

- Images: `assets/*.jpg`
- Metadata: `assets/*.yaml`
- Model: `gpt-image 2.0` from generated image C2PA metadata
- Date: 2026-06-27
- Style: product-only retail catalog shots on neutral studio backgrounds

## Licensing

Use of these generated images is governed by the OpenAI Services Agreement and
OpenAI policies. OpenAI states that, as between the customer and OpenAI and to
the extent permitted by law, the customer owns generated output, and OpenAI
does not claim copyright over API-generated content.

References:

- https://openai.com/policies/services-agreement/
- https://help.openai.com/en/articles/5008634-who-owns-the-output-of-image-generations

Scenarios should reference sidecar `id` values as `image_id`. Sidecar
descriptions are evaluation ground truth and are not sent to the shopping
agent by default.
