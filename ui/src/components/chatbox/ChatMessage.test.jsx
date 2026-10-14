/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import React from "react";
import { createRoot } from "react-dom/client";

import ChatMessage from "./ChatMessage";

global.IS_REACT_ACT_ENVIRONMENT = true;

describe("ChatMessage loader", () => {
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

  test("shows what the turn is doing while the reply is pending", () => {
    React.act(() =>
      root.render(
        <ChatMessage
          role="assistant"
          content="loader"
          productName=""
          progressLabel="Searching the catalog"
        />
      )
    );

    expect(container.querySelector('[aria-label="Loading spinner"]')).not.toBeNull();
    expect(container.textContent).toContain("Searching the catalog…");
  });

  test("shows only the spinner before any progress arrives", () => {
    React.act(() =>
      root.render(<ChatMessage role="assistant" content="loader" productName="" />)
    );

    expect(container.querySelector('[aria-label="Loading spinner"]')).not.toBeNull();
    expect(container.textContent).toBe("");
  });
});
