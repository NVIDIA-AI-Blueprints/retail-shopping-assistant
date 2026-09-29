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

  test("shows the result in the quiet summary while details stay collapsed", () => {
    React.act(() => root.render(<GuardrailDetails report={report} />));

    const details = container.querySelector("details");
    expect(details.open).toBe(false);
    expect(container.querySelector("summary").textContent).toContain(
      "Safety · input passed · output passed · 975 ms"
    );
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

  test.each([
    ["non_retail", { content_safety: 1, topic_control: 1 }, "Topic control"],
    ["content_safety", { content_safety: 1 }, "Content safety"],
    ["unsafe_video", { multimodal_safety: 1 }, "Video safety"],
  ])("names the rail that blocked %s", (category, modelCalls, expected) => {
    const blocked = {
      ...report,
      checks: [
        {
          ...report.checks[0],
          status: "block",
          violated_categories: [category],
          model_calls: modelCalls,
        },
      ],
    };

    React.act(() => root.render(<GuardrailDetails report={blocked} />));

    expect(container.querySelector("summary").textContent).toContain(
      `input blocked by ${expected}`
    );
  });
});
