// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";
import { createRoot } from "react-dom/client";

import App from "./App";

vi.mock("@mui/icons-material", () => ({
  Menu: () => null,
  Send: () => null,
  AttachFile: () => null,
  Close: () => null,
  RestartAlt: () => null,
  ShoppingCart: () => null,
  DeleteOutlined: () => null,
}));
vi.mock("@mui/material/Switch", () => ({ default: () => null }));
vi.mock("@mui/material/styles", () => ({
  styled: (Component) => () => Component,
}));

global.IS_REACT_ACT_ENVIRONMENT = true;
global.TextDecoder = class {
  decode() {
    return "";
  }
};
Element.prototype.scrollIntoView = vi.fn();

const SESSION_STORAGE_KEY = "shopping_session_identity";
const click = (element) => {
  React.act(() => {
    element.click();
  });
};

const flushEffects = async () => {
  await React.act(async () => {
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
  });
};

const submitQuery = async (container, query) => {
  const input = container.querySelector(".input_test");
  const valueSetter = Object.getOwnPropertyDescriptor(
    HTMLInputElement.prototype,
    "value"
  ).set;

  React.act(() => {
    valueSetter.call(input, query);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
  click(container.querySelector('[aria-label="Send message"]'));
  await flushEffects();
};

describe("App session lifecycle", () => {
  let container;
  let root;
  let originalFetch;

  beforeEach(() => {
    vi.useFakeTimers();
    sessionStorage.clear();
    originalFetch = global.fetch;
    global.fetch = vi.fn((input) => {
      const url = String(input);
      if (url.includes("/cart")) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ lines: [], subtotal: null }),
        });
      }
      if (url.endsWith("/capabilities")) {
        return Promise.resolve({
          ok: false,
          json: () => Promise.resolve({}),
        });
      }
      if (url.endsWith("/query/stream")) {
        return Promise.resolve({
          ok: true,
          body: {
            getReader: () => ({
              read: () => Promise.resolve({ value: undefined, done: true }),
            }),
          },
        });
      }
      return Promise.reject(new Error(`Unexpected request: ${url}`));
    });

    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(() => {
    React.act(() => {
      root.unmount();
    });
    container.remove();
    sessionStorage.clear();
    global.fetch = originalFetch;
    vi.clearAllTimers();
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  const streamPayloads = () =>
    global.fetch.mock.calls
      .filter(([input]) => String(input).endsWith("/query/stream"))
      .map(([, request]) => JSON.parse(request.body));

  test("requests carry no shopper profile and there is no shopper picker", async () => {
    // This test uses React createRoot directly, so rendering must be inside act.
    React.act(() => {
      root.render(<App />);
    });
    await flushEffects();

    await submitQuery(container, "Show me bags");

    const [payload] = streamPayloads();
    expect(payload.query).toBe("Show me bags");
    expect(payload).not.toHaveProperty("shopper_profile_id");
    expect(container.textContent).not.toContain("Shop as");
    expect(
      global.fetch.mock.calls.some(([input]) =>
        String(input).includes("shopper-profiles")
      )
    ).toBe(false);
  });

  test("Reset clears the identity and the next request starts a new conversation", async () => {
    // This test uses React createRoot directly, so rendering must be inside act.
    React.act(() => {
      root.render(<App />);
    });
    await flushEffects();

    await submitQuery(container, "Show me bags");
    const [first] = streamPayloads();

    click(container.querySelector('[aria-label="Reset conversation"]'));
    expect(sessionStorage.getItem(SESSION_STORAGE_KEY)).toBeNull();
    await flushEffects();

    await submitQuery(container, "Show me shoes");
    const second = streamPayloads().at(-1);
    expect(second.query).toBe("Show me shoes");
    expect(second.conversation_id).not.toBe(first.conversation_id);
    expect(second.cart_id).not.toBe(first.cart_id);
  });
});
