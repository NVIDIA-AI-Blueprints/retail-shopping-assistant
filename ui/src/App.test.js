// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";
import { createRoot } from "react-dom/client";

import App from "./App";
import {
  getOrCreateUserSession,
  getSelectedShopperProfileId,
} from "./utils";

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

const click = (element) => {
  React.act(() => {
    element.click();
  });
};

const chooseShopper = async (container, value) => {
  const select = container.querySelector('[aria-label="Shopper profile"]');
  React.act(() => {
    select.value = value;
    select.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await flushEffects();
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

  test("requires an explicit choice before Guest can start chatting", async () => {
    // This test uses React createRoot directly, so rendering must be inside act.
    // eslint-disable-next-line testing-library/no-unnecessary-act
    React.act(() => {
      root.render(<App />);
    });
    await flushEffects();

    expect(container.textContent).toContain("Choose how you’d like to shop");
    expect(container.querySelector(".input_test")).toBeNull();
    expect(
      global.fetch.mock.calls.some(([input]) =>
        String(input).endsWith("/query/stream")
      )
    ).toBe(false);

    await chooseShopper(container, "__guest__");
    await submitQuery(container, "Show me bags");

    const streamCall = global.fetch.mock.calls.find(([input]) =>
      String(input).endsWith("/query/stream")
    );
    const payload = JSON.parse(streamCall[1].body);

    expect(payload.query).toBe("Show me bags");
    expect(payload).not.toHaveProperty("shopper_profile_id");
    expect(sessionStorage.getItem(SHOPPER_PROFILE_STORAGE_KEY)).not.toBeNull();
  });

  test("shows named shopper context and keeps Guest mode uncluttered", async () => {
    // eslint-disable-next-line testing-library/no-unnecessary-act
    React.act(() => {
      root.render(<App />);
    });
    await flushEffects();

    expect(
      container.querySelector('[aria-label="Selected shopper profile"]')
    ).toBeNull();

    await chooseShopper(container, "shopper_alex");

    const summary = container.querySelector(
      '[aria-label="Selected shopper profile"]'
    );
    expect(summary.textContent).toContain("Alex");
    expect(summary.textContent).toContain("occasion driven explorer");
    expect(summary.textContent).toContain(profiles[1].behavior);
    expect(summary.textContent).toContain("Saved ZIP 98101");

    await chooseShopper(container, "__guest__");

    expect(
      container.querySelector('[aria-label="Selected shopper profile"]')
    ).toBeNull();
    expect(container.querySelector(".input_test")).not.toBeNull();
  });

  test("profile changes rotate identity while Reset retains the selected mode", async () => {
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

    await chooseShopper(container, "shopper_morgan");

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

    await chooseShopper(container, "shopper_alex");
    const alexSession = getOrCreateUserSession();
    expect(alexSession.sessionId).not.toBe(shopperSession.sessionId);
    expect(alexSession.conversationId).not.toBe(shopperSession.conversationId);
    expect(alexSession.cartId).not.toBe(shopperSession.cartId);
    expect(container.querySelectorAll(".messages__item--user")).toHaveLength(0);

    await submitQuery(container, "Find a wedding outfit");

    const payloads = global.fetch.mock.calls
      .filter(([input]) => String(input).endsWith("/query/stream"))
      .map(([, request]) => JSON.parse(request.body));

    expect(payloads).toHaveLength(2);
    expect(payloads[0].shopper_profile_id).toBe("shopper_morgan");
    expect(payloads[0].conversation_id).toBe(shopperSession.conversationId);
    expect(payloads[0].cart_id).toBe(shopperSession.cartId);
    expect(payloads[1].shopper_profile_id).toBe("shopper_alex");
    expect(payloads[1].conversation_id).toBe(alexSession.conversationId);
    expect(payloads[1].cart_id).toBe(alexSession.cartId);
    payloads.forEach((payload) => {
      expect(payload).not.toHaveProperty("display_name");
      expect(payload).not.toHaveProperty("shopper_type");
      expect(payload).not.toHaveProperty("behavior");
      expect(payload).not.toHaveProperty("zipcode");
    });

    await chooseShopper(container, "__guest__");
    const guestSession = getOrCreateUserSession();
    expect(guestSession.sessionId).not.toBe(alexSession.sessionId);
    expect(guestSession.conversationId).not.toBe(alexSession.conversationId);
    expect(guestSession.cartId).not.toBe(alexSession.cartId);

    await submitQuery(container, "Show me shoes");
    const guestPayload = JSON.parse(
      global.fetch.mock.calls
        .filter(([input]) => String(input).endsWith("/query/stream"))
        .at(-1)[1].body
    );
    expect(guestPayload.conversation_id).toBe(guestSession.conversationId);
    expect(guestPayload.cart_id).toBe(guestSession.cartId);
    expect(guestPayload).not.toHaveProperty("shopper_profile_id");

    click(container.querySelector('[aria-label="Reset conversation"]'));
    expect(sessionStorage.getItem(SESSION_STORAGE_KEY)).toBeNull();
    expect(sessionStorage.getItem(SHOPPER_PROFILE_STORAGE_KEY)).not.toBeNull();
    expect(getSelectedShopperProfileId()).toBeNull();
  });

  test("profile-service failure leaves Guest available and recovers in place", async () => {
    jest.spyOn(console, "warn").mockImplementation(() => {});
    global.fetch.mockImplementationOnce(() =>
      Promise.reject(new Error("profiles unavailable"))
    );

    // eslint-disable-next-line testing-library/no-unnecessary-act
    React.act(() => {
      root.render(<App />);
    });
    await flushEffects();

    let select = container.querySelector('[aria-label="Shopper profile"]');
    expect(select.textContent).toContain("Guest mode");
    expect(select.textContent).toContain("Profiles unavailable");
    expect(container.querySelector(".input_test")).toBeNull();

    await chooseShopper(container, "__guest__");
    const guestSession = getOrCreateUserSession();

    expect(container.querySelector(".input_test")).not.toBeNull();
    React.act(() => {
      jest.runOnlyPendingTimers();
    });
    await flushEffects();

    select = container.querySelector('[aria-label="Shopper profile"]');
    expect(Array.from(select.options).map((option) => option.text)).toEqual([
      "Choose shopper",
      "Guest mode",
      "Morgan",
      "Alex",
      "Casey",
      "Jordan",
      "Riley",
    ]);
    expect(select.textContent).not.toContain("Profiles unavailable");
    expect(select.value).toBe("__guest__");
    expect(container.querySelector(".input_test")).not.toBeNull();
    expect(getOrCreateUserSession()).toEqual(guestSession);
  });

  test("bounds automatic profile retries and retries again on focus", async () => {
    jest.spyOn(console, "warn").mockImplementation(() => {});
    global.fetch.mockImplementationOnce(() =>
      Promise.reject(new Error("profiles unavailable"))
    );
    global.fetch.mockImplementationOnce(() =>
      Promise.reject(new Error("profiles still unavailable"))
    );

    // eslint-disable-next-line testing-library/no-unnecessary-act
    React.act(() => {
      root.render(<App />);
    });
    await flushEffects();

    React.act(() => {
      jest.runOnlyPendingTimers();
    });
    await flushEffects();

    const profileRequestCount = () =>
      global.fetch.mock.calls.filter(([input]) =>
        String(input).endsWith("/shopper-profiles")
      ).length;

    expect(profileRequestCount()).toBe(2);

    React.act(() => {
      jest.advanceTimersByTime(30000);
    });
    await flushEffects();
    expect(profileRequestCount()).toBe(2);

    React.act(() => {
      window.dispatchEvent(new Event("focus"));
    });
    await flushEffects();

    expect(profileRequestCount()).toBe(3);
    expect(
      container.querySelector('[aria-label="Shopper profile"]').textContent
    ).toContain("Alex");
  });
});
