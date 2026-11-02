/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import React from "react";
import { createRoot } from "react-dom/client";

import ProductResultGrid, { COLLAPSED_PRODUCT_COUNT } from "./ProductResultGrid";

global.IS_REACT_ACT_ENVIRONMENT = true;

const products = (count) =>
  Array.from({ length: count }, (_, index) => ({
    productName: `Product ${index + 1}`,
    productUrl: `/images/${index + 1}.jpg`,
    price: { amount: 20 + index, currency: "USD" },
  }));

describe("ProductResultGrid", () => {
  let container;
  let root;

  beforeEach(() => {
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(() => {
    React.act(() => root.unmount());
    container.remove();
  });

  const cards = () => container.querySelectorAll(".product-result-card");
  const toggle = () => container.querySelector(".product-result-toggle");

  test("shows the first products and expands to all of them", () => {
    React.act(() => root.render(<ProductResultGrid images={products(12)} />));

    expect(cards()).toHaveLength(COLLAPSED_PRODUCT_COUNT);
    expect(toggle().textContent).toBe("Show all 12");
    expect(toggle().getAttribute("aria-expanded")).toBe("false");

    React.act(() => toggle().click());

    expect(cards()).toHaveLength(12);
    expect(toggle().textContent).toBe("Show fewer");
    expect(toggle().getAttribute("aria-expanded")).toBe("true");

    React.act(() => toggle().click());

    expect(cards()).toHaveLength(COLLAPSED_PRODUCT_COUNT);
  });

  test("has no toggle when every product already fits", () => {
    React.act(() =>
      root.render(<ProductResultGrid images={products(COLLAPSED_PRODUCT_COUNT)} />)
    );

    expect(cards()).toHaveLength(COLLAPSED_PRODUCT_COUNT);
    expect(toggle()).toBeNull();
  });

  test("selects a product and marks the selected card", () => {
    const onProductSelect = vi.fn();
    const images = products(3);
    React.act(() =>
      root.render(
        <ProductResultGrid
          images={images}
          selectedProductName="Product 2"
          onProductSelect={onProductSelect}
        />
      )
    );

    expect(cards()[1].classList.contains("is-selected")).toBe(true);
    expect(cards()[1].textContent).toContain("$21.00");

    React.act(() => cards()[0].click());

    expect(onProductSelect).toHaveBeenCalledWith(images[0]);
  });
});
