// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  clearUserSession,
  createApiRequest,
  getOrCreateUserSession,
  getSelectedShopperProfileId,
  parseShopperProfiles,
  requestIdForTurn,
  rotateUserSession,
  setSelectedShopperProfileId,
  settleTurn,
  turnFingerprint,
} from ".";

const profile = {
  shopper_profile_id: "shopper_morgan",
  display_name: "Morgan",
  shopper_type: "skeptical_researcher",
  behavior:
    "Probes for material, care burden, and repeated-wear practicality before choosing.",
  zipcode: "60601",
};

const profiles = [
  profile,
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

beforeEach(() => {
  sessionStorage.clear();
});

test("shopper selection is tab-scoped and separate from chat identity", () => {
  const firstSession = getOrCreateUserSession();
  setSelectedShopperProfileId(profile.shopper_profile_id);

  const secondSession = rotateUserSession();

  expect(getSelectedShopperProfileId()).toBe(profile.shopper_profile_id);
  expect(secondSession.sessionId).not.toBe(firstSession.sessionId);
  expect(secondSession.conversationId).not.toBe(firstSession.conversationId);
  expect(secondSession.cartId).not.toBe(firstSession.cartId);
});

test("Guest removes the stored shopper selection", () => {
  setSelectedShopperProfileId(profile.shopper_profile_id);
  setSelectedShopperProfileId(null);

  expect(getSelectedShopperProfileId()).toBeNull();
});

test("shopper profile responses are validated without caching profile contents", () => {
  expect(parseShopperProfiles(profiles)).toEqual(profiles);
  expect(() =>
    parseShopperProfiles([
      { ...profile, zipcode: "6060" },
      ...profiles.slice(1),
    ])
  ).toThrow("invalid response");
  expect(() => parseShopperProfiles([profile, profile, ...profiles.slice(2)])).toThrow(
    "duplicate identifiers"
  );
});

test("selected shopper query payload contains only its profile ID", () => {
  setSelectedShopperProfileId(profile.shopper_profile_id);

  const payload = createApiRequest(
    getOrCreateUserSession(),
    "Show me bags",
    "",
    true,
    [],
    getSelectedShopperProfileId()
  );

  expect(payload.shopper_profile_id).toBe(profile.shopper_profile_id);
  expect(payload).not.toHaveProperty("display_name");
  expect(payload).not.toHaveProperty("shopper_type");
  expect(payload).not.toHaveProperty("behavior");
  expect(payload).not.toHaveProperty("zipcode");
  expect(JSON.stringify(payload)).not.toContain(profile.display_name);
  expect(JSON.stringify(payload)).not.toContain(profile.behavior);
});

test("Guest query payload omits shopper_profile_id", () => {
  const payload = createApiRequest(
    getOrCreateUserSession(),
    "Show me bags",
    "",
    true,
    [],
    null
  );

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

test("A different message, profile or guardrail setting is a new turn", () => {
  sessionStorage.clear();
  const session = getOrCreateUserSession();
  const base = turnFingerprint(createApiRequest(session, "add one", "", true));

  for (const other of [
    createApiRequest(session, "add two", "", true),
    createApiRequest(session, "add one", "", false),
    createApiRequest(session, "add one", "data:image/png;base64,AA", true),
    createApiRequest(session, "add one", "", true, [], profile.shopper_profile_id),
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
