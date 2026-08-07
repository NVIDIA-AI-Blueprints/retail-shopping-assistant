// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * What is running, and what this conversation has cost.
 *
 * Sits directly under the NVIDIA mark in the chat header, so "powered by" and
 * the models actually doing the work read as one statement rather than two
 * unrelated bits of chrome.
 */

import React, { useState } from "react";

import { ModelCapabilities, ModelUsage, SessionUsage } from "../../types";

interface ModelStripProps {
  models: ModelCapabilities;
  modelUsage: ModelUsage;
  sessionUsage: SessionUsage;
  /** The conversation id sent to the server, and the trace session's name. */
  conversationId: string;
}

/**
 * The roles a shopper is told about, and what each covers.
 *
 * Labels name the job, not the model. The specific models change; a row
 * reading "gpt-5.2" goes stale the day it is swapped, and a screenshot of it
 * goes stale immediately. The live model id is on hover, where it comes from
 * /capabilities and so cannot drift.
 *
 * `image_embedding` is deliberately absent: it is configured, but the catalog
 * is not using it for image search, and naming a model that never runs is a
 * claim rather than a status.
 */
const SHOWN_MODELS: Array<{ label: string; roles: string[] }> = [
  { label: "Nemotron VLM", roles: ["vlm"] },
  // The grounding editor is the same model doing a second job. A shopper
  // reading "LLM" wants the cost of the answer, not an org chart.
  { label: "LLM", roles: ["app_llm", "app_llm_grounding_editor"] },
  { label: "Embedding", roles: ["text_embedding"] },
  { label: "Guardrails", roles: ["content_safety", "topic_control"] },
];
const ModelStrip: React.FC<ModelStripProps> = ({
  models,
  modelUsage,
  sessionUsage,
  conversationId,
}) => {
  const [copied, setCopied] = useState(false);

  /**
   * The conversation id is the trace session's name, so copying it is how a
   * conversation on screen is found in the trace viewer. Reading it out of
   * sessionStorage in devtools was the alternative.
   */
  const copyConversationId = () => {
    const done = () => {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1400);
    };
    if (navigator.clipboard?.writeText) {
      navigator.clipboard.writeText(conversationId).then(done).catch(() => done());
      return;
    }
    done();
  };

  const shown = SHOWN_MODELS.filter((entry) =>
    entry.roles.some((role) => isConfigured(models[role]))
  );

  return (
    <div className="model-strip" aria-label="Models in use">
      <div className="model-strip__models">
        {shown.map((entry) => {
          const names = entry.roles
            .map((role) => models[role]?.model)
            .filter((name): name is string => Boolean(name));
          const calls = sum(entry.roles, (role) => modelUsage[role]?.calls);
          const tokens = sum(entry.roles, (role) => modelUsage[role]?.tokens);
          return (
            <div
              key={entry.label}
              className={`model-strip__model${calls > 0 ? " is-active" : ""}`}
              title={names.join("\n")}
            >
              <span className="model-strip__name">
                <i className="model-strip__dot" aria-hidden="true" />
                {entry.label}
              </span>
              {/* Always rendered, so the columns exist before the first turn
                  and the numbers appear in place rather than pushing the bar
                  around as models start firing. */}
              <span className="model-strip__stat">
                {calls === 0 ? (
                  <span className="model-strip__idle">not used yet</span>
                ) : (
                  <>
                    {formatNumber(calls)} {calls === 1 ? "call" : "calls"}
                    {/* Only chat models report tokens; embeddings and the
                        guardrails checks have none, and a zero there would
                        read as "used none" rather than "has none". */}
                    {tokens > 0 && (
                      <> · {formatCompact(tokens)} tokens</>
                    )}
                  </>
                )}
              </span>
            </div>
          );
        })}
      </div>

      <button
        type="button"
        className="model-strip__session"
        onClick={copyConversationId}
        disabled={!conversationId}
        title={`Trace session: ${conversationId}\nClick to copy`}
        aria-label={`Copy conversation id ${conversationId}`}
      >
        <span className="model-strip__session-label">session</span>
        <span className="model-strip__session-id">
          {/* Empty until the first turn: a reset clears the identity and the
              next one is minted when the shopper speaks. */}
          {!conversationId ? "—" : copied ? "copied" : shortId(conversationId)}
        </span>
      </button>

      <div className="model-strip__total">
        <span className="model-strip__total-label">This session</span>
        <span className="model-strip__total-value">
          <strong>{formatNumber(sessionUsage.modelCalls)}</strong> calls ·{" "}
          <strong>{formatCompact(sessionUsage.totalTokens)}</strong> tokens
        </span>
      </div>
    </div>
  );
};

const sum = (
  roles: string[],
  pick: (role: string) => number | undefined
): number => roles.reduce((total, role) => total + (pick(role) ?? 0), 0);

/** `conversation-3b0285ce-…` -> `3b0285ce`. Enough to recognise, short enough to sit in a bar. */
const shortId = (value: string): string => {
  const bare = value.replace(/^conversation-/, "");
  return bare.slice(0, 8) || value;
};

const isConfigured = (model: ModelCapabilities[string]): boolean =>
  Boolean(model?.enabled && model?.model);

const formatNumber = (value: number): string =>
  new Intl.NumberFormat("en-US").format(value);

/** 38200 -> "38.2k". A running total has to stay a stable width as it grows. */
const formatCompact = (value: number): string => {
  if (value < 1000) return String(value);
  if (value < 1_000_000) return `${(value / 1000).toFixed(1)}k`;
  return `${(value / 1_000_000).toFixed(1)}m`;
};

export default ModelStrip;
