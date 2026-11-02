// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  clearUserSession,
  createApiRequest,
  getOrCreateUserSession,
  requestIdForTurn,
  settleTurn,
  turnFingerprint,
} from ".";

beforeEach(() => {
  sessionStorage.clear();
});

test("Query payload carries no shopper profile", () => {
  const payload = createApiRequest(getOrCreateUserSession(), "Show me bags", "", true);

  expect(payload).not.toHaveProperty("shopper_profile_id");
});

test("Query payload overrides guardrails only after an explicit toggle", () => {
  const session = getOrCreateUserSession();

  expect(createApiRequest(session, "safe", "", true).guardrails).toBe(true);
  expect(createApiRequest(session, "bypass", "", false).guardrails).toBe(false);
  expect(createApiRequest(session, "default")).not.toHaveProperty("guardrails");
});

test("A resend of an unfinished turn reuses its request ID", () => {
  sessionStorage.clear();
  const session = getOrCreateUserSession();
  const payload = createApiRequest(session, "remove one of the boots");
  expect(payload).not.toHaveProperty("request_id");

  const first = requestIdForTurn(turnFingerprint(payload));
  expect(first).toMatch(/^request-[A-Za-z0-9][A-Za-z0-9._:-]*$/);
  expect(requestIdForTurn(turnFingerprint(payload))).toBe(first);

  settleTurn(first);
  expect(requestIdForTurn(turnFingerprint(payload))).not.toBe(first);
});

test("A different message, image or guardrail setting is a new turn", () => {
  sessionStorage.clear();
  const session = getOrCreateUserSession();
  const base = turnFingerprint(createApiRequest(session, "add one", "", true));

  for (const other of [
    createApiRequest(session, "add two", "", true),
    createApiRequest(session, "add one", "", false),
    createApiRequest(session, "add one", "data:image/png;base64,AA", true),
  ]) {
    expect(turnFingerprint(other)).not.toBe(base);
  }

  const first = requestIdForTurn(base);
  requestIdForTurn(turnFingerprint(createApiRequest(session, "add two", "", true)));
  expect(requestIdForTurn(base)).not.toBe(first);
});

test("Settling a turn that is no longer pending leaves the pending one alone", () => {
  sessionStorage.clear();
  const session = getOrCreateUserSession();
  const fingerprint = turnFingerprint(createApiRequest(session, "add one"));
  const pending = requestIdForTurn(fingerprint);

  settleTurn("request-older");
  expect(requestIdForTurn(fingerprint)).toBe(pending);

  clearUserSession();
  expect(requestIdForTurn(fingerprint)).not.toBe(pending);
});
