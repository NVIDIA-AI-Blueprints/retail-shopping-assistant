// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";
import { createRoot } from "react-dom/client";

import App from "./App";
import { getOrCreateUserSession } from "./utils";

jest.mock("@mui/icons-material/Menu", () => () => null);
jest.mock("@mui/icons-material/Send", () => () => null);
jest.mock("@mui/icons-material/AttachFile", () => () => null);
jest.mock("@mui/icons-material/Close", () => () => null);
jest.mock("@mui/icons-material/RestartAlt", () => () => null);
jest.mock("@mui/material/Switch", () => () => null);
jest.mock("@mui/material/styles", () => ({
  styled: (Component) => () => Component,
}));

global.IS_REACT_ACT_ENVIRONMENT = true;
global.TextDecoder = class {
  decode() {
    return "";
  }
};
Element.prototype.scrollIntoView = jest.fn();

const SESSION_STORAGE_KEY = "shopping_session_identity";
const SHOPPER_PROFILE_STORAGE_KEY = "shopping_shopper_profile_id";

const profiles = [
  {
    shopper_profile_id: "shopper_morgan",
    display_name: "Morgan",
    shopper_type: "skeptical_researcher",
    behavior:
      "Probes for material, care burden, and repeated-wear practicality before choosing.",
    zipcode: "60601",
  },
  {
    shopper_profile_id: "shopper_alex",
    display_name: "Alex",
    shopper_type: "occasion_driven_explorer",
    behavior: "Starts from an occasion and asks for a complete look.",
    zipcode: "98101",
  },
  {
    shopper_profile_id: "shopper_casey",
    display_name: "Casey",
    shopper_type: "strict_budget_style_mixer",
    behavior: "Balances a complete look with an explicit budget.",
    zipcode: "85004",
  },
  {
    shopper_profile_id: "shopper_jordan",
    display_name: "Jordan",
    shopper_type: "impatient_decisive",
    behavior: "Uses concise references and changes cart decisions.",
    zipcode: "10001",
  },
  {
    shopper_profile_id: "shopper_riley",
    display_name: "Riley",
    shopper_type: "iterative_refiner",
    behavior: "Refines one part of an existing outfit direction.",
    zipcode: "33130",
  },
];

const buttonWithText = (container, text) =>
  Array.from(container.querySelectorAll("button")).find((button) =>
    button.textContent.includes(text)
  );

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

describe("App shopper identity lifecycle", () => {
  let container;
  let root;
  let originalFetch;

  beforeEach(() => {
    jest.useFakeTimers();
    sessionStorage.clear();
    originalFetch = global.fetch;
    global.fetch = jest.fn((input) => {
      const url = String(input);
      if (url.endsWith("/shopper-profiles")) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(profiles),
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
    jest.clearAllTimers();
    jest.useRealTimers();
    jest.restoreAllMocks();
  });

  test("Guest requests omit shopper_profile_id through App and Chatbox", async () => {
    // This test uses React createRoot directly, so rendering must be inside act.
    // eslint-disable-next-line testing-library/no-unnecessary-act
    React.act(() => {
      root.render(<App />);
    });
    await flushEffects();

    await submitQuery(container, "Show me bags");

    const streamCall = global.fetch.mock.calls.find(([input]) =>
      String(input).endsWith("/query/stream")
    );
    const payload = JSON.parse(streamCall[1].body);

    expect(payload.query).toBe("Show me bags");
    expect(payload).not.toHaveProperty("shopper_profile_id");
  });

  test("shopper switch keeps exactly its rotated identity and manual Reset clears it", async () => {
    const setItemSpy = jest.spyOn(Storage.prototype, "setItem");

    // This test uses React createRoot directly, so rendering must be inside act.
    // eslint-disable-next-line testing-library/no-unnecessary-act
    React.act(() => {
      root.render(<App />);
    });
    await flushEffects();

    const priorSession = getOrCreateUserSession();
    const writesBeforeSwitch = setItemSpy.mock.calls.filter(
      ([key]) => key === SESSION_STORAGE_KEY
    ).length;

    click(buttonWithText(container, "Shop as"));
    click(buttonWithText(container, "Morgan"));
    click(buttonWithText(container, "Shop as Morgan"));
    await flushEffects();

    const identityWrites = setItemSpy.mock.calls
      .filter(([key]) => key === SESSION_STORAGE_KEY)
      .slice(writesBeforeSwitch);
    const storedIdentity = sessionStorage.getItem(SESSION_STORAGE_KEY);

    expect(identityWrites).toHaveLength(1);
    expect(storedIdentity).toBe(identityWrites[0][1]);
    expect(sessionStorage.getItem(SHOPPER_PROFILE_STORAGE_KEY)).toBe(
      "shopper_morgan"
    );

    const shopperSession = getOrCreateUserSession();
    expect(shopperSession.sessionId).not.toBe(priorSession.sessionId);
    expect(shopperSession.conversationId).not.toBe(priorSession.conversationId);
    expect(shopperSession.cartId).not.toBe(priorSession.cartId);

    await submitQuery(container, "Show me bags");
    await submitQuery(container, "Which one is under $50?");

    const payloads = global.fetch.mock.calls
      .filter(([input]) => String(input).endsWith("/query/stream"))
      .map(([, request]) => JSON.parse(request.body));

    expect(payloads).toHaveLength(2);
    payloads.forEach((payload) => {
      expect(payload.shopper_profile_id).toBe("shopper_morgan");
      expect(payload.conversation_id).toBe(shopperSession.conversationId);
      expect(payload.cart_id).toBe(shopperSession.cartId);
      expect(payload).not.toHaveProperty("display_name");
      expect(payload).not.toHaveProperty("shopper_type");
      expect(payload).not.toHaveProperty("behavior");
      expect(payload).not.toHaveProperty("zipcode");
    });

    click(container.querySelector('[aria-label="Reset conversation"]'));
    expect(sessionStorage.getItem(SESSION_STORAGE_KEY)).toBeNull();
    expect(sessionStorage.getItem(SHOPPER_PROFILE_STORAGE_KEY)).toBe(
      "shopper_morgan"
    );
  });
});
