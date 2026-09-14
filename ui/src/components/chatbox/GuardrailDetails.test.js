/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import React from "react";
import { createRoot } from "react-dom/client";

import GuardrailDetails from "./GuardrailDetails";

global.IS_REACT_ACT_ENVIRONMENT = true;

const report = {
  enabled: true,
  failure_mode: "closed",
  checks: [
    {
      stage: "input",
      status: "allow",
      violated_categories: [],
      latency_ms: 624.7,
      model_calls: { content_safety: 1, topic_control: 1 },
    },
    {
      stage: "output",
      status: "allow",
      violated_categories: [],
      latency_ms: 350.2,
      model_calls: { content_safety: 1 },
    },
  ],
};

describe("GuardrailDetails", () => {
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

  test("stays collapsed until the subtle safety summary is opened", () => {
    React.act(() => root.render(<GuardrailDetails report={report} />));

    const details = container.querySelector("details");
    expect(details.open).toBe(false);
    expect(container.querySelector("summary").textContent).toContain("Safety");
    expect(container.textContent).toContain("Content safety · Topic control");
    expect(container.textContent).toContain("625 ms");

    React.act(() => container.querySelector("summary").click());

    expect(details.open).toBe(true);
  });

  test("does not render when guardrails were disabled", () => {
    React.act(() =>
      root.render(<GuardrailDetails report={{ ...report, enabled: false }} />)
    );

    expect(container.firstChild).toBeNull();
  });
});
