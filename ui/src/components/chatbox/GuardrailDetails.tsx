/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import React from "react";

import { GuardrailCheckResult, GuardrailReport } from "../../types";

const CHECK_LABELS: Record<string, string> = {
  content_safety: "Content safety",
  topic_control: "Topic control",
  multimodal_safety: "Video safety",
};

const statusLabel = (check: GuardrailCheckResult): string => {
  if (check.status === "allow") return "Passed";
  if (check.status === "block") return "Blocked";
  return "Unavailable";
};

const checkNames = (check: GuardrailCheckResult): string => {
  const names = Object.entries(check.model_calls)
    .filter(([, calls]) => calls > 0)
    .map(([name]) => CHECK_LABELS[name] || name.replace(/_/g, " "));
  return names.join(" · ");
};

const blockedBy = (check: GuardrailCheckResult): string => {
  if (check.status !== "block") return "";
  if (check.violated_categories.includes("non_retail")) return "Topic control";
  if (check.violated_categories.includes("content_safety")) {
    return "Content safety";
  }
  const called = Object.keys(check.model_calls).filter(
    (name) => check.model_calls[name] > 0
  );
  if (called.length === 1 && called[0] === "multimodal_safety") {
    return "Video safety";
  }
  return "Safety policy";
};

const GuardrailDetails: React.FC<{ report?: GuardrailReport }> = ({
  report,
}) => {
  if (!report?.enabled || report.checks.length === 0) return null;

  const summaryStatus = report.checks.some((check) => check.status !== "allow")
    ? "attention"
    : "passed";
  const summary = report.checks
    .map((check) => {
      const blocker = blockedBy(check);
      return `${check.stage} ${statusLabel(check).toLowerCase()}${
        blocker ? ` by ${blocker}` : ""
      }`;
    })
    .join(" · ");
  const totalLatency = report.checks.reduce(
    (total, check) => total + check.latency_ms,
    0
  );

  return (
    <details
      className={`guardrail-details guardrail-details--${summaryStatus}`}
    >
      <summary>
        <span className="guardrail-details__marker" aria-hidden="true" />
        Safety · {summary} · {Math.round(totalLatency)} ms
      </summary>
      <div className="guardrail-details__body">
        {report.checks.map((check) => {
          const names = checkNames(check);
          return (
            <div className="guardrail-details__check" key={check.stage}>
              <span className="guardrail-details__stage">{check.stage}</span>
              <span
                className={`guardrail-details__status guardrail-details__status--${check.status}`}
              >
                {statusLabel(check)}
              </span>
              {names && <span>{names}</span>}
              <span>{Math.round(check.latency_ms)} ms</span>
              {check.violated_categories.length > 0 && (
                <span>{check.violated_categories.join(", ")}</span>
              )}
            </div>
          );
        })}
        <div className="guardrail-details__mode">
          Provider errors{" "}
          {report.failure_mode === "closed" ? "stop" : "do not stop"} the
          request.
        </div>
      </div>
    </details>
  );
};

export default GuardrailDetails;
