// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Customer-facing inference workload summary for the current assistant turn.
 */

import React from "react";

import {
  InferenceActivity,
  InferenceCategory,
  ModelCapabilities,
  ModelUsage,
  ModelUsageStatus,
  TokenUsage,
} from "../../types";

interface InferenceActivityPanelProps {
  events: InferenceActivity[];
  models: ModelCapabilities;
  tokenUsage: TokenUsage | null;
  modelUsage: ModelUsage;
}

const InferenceActivityPanel: React.FC<InferenceActivityPanelProps> = ({
  events,
  models,
  tokenUsage,
  modelUsage,
}) => {
  const groupedCounts = countByCategory(events);
  const hasTokenUsage = Boolean(tokenUsage && tokenUsage.model_calls > 0);
  const llmCallCount = hasTokenUsage ? tokenUsage!.model_calls : groupedCounts.language;
  const usageRows = modelUsageRows(models, modelUsage, tokenUsage, events);
  const embeddingCallCount =
    usageCalls(modelUsage, ["text_embedding", "image_embedding"]) || groupedCounts.embedding;
  const omniCallCount = modelUsage.vlm?.calls ?? groupedCounts.vision;

  return (
    <aside className="inference-panel" aria-label="Inference activity">
      <div className="inference-panel__header">
        <div>
          <div className="inference-panel__eyebrow">NVIDIA inference</div>
          <h2>Model usage</h2>
        </div>
        <span className="inference-panel__total">
          {hasTokenUsage ? `${formatNumber(tokenUsage!.total_tokens)} tokens` : "Idle"}
        </span>
      </div>

      <div className="inference-panel__summary">
        <SummaryMetric label="Omni calls" value={omniCallCount} />
        <SummaryMetric label="LLM calls" value={llmCallCount} />
        <SummaryMetric label="Embedding calls" value={embeddingCallCount} />
        <SummaryMetric label="Safety calls" value={groupedCounts.safety} />
      </div>

      <div className="inference-panel__usage" aria-label="Token usage">
        <div className="inference-panel__section-title">Token usage</div>
        <div className="inference-panel__usage-grid">
          <SummaryMetric
            label="Input"
            value={hasTokenUsage ? formatNumber(tokenUsage!.input_tokens) : "--"}
          />
          <SummaryMetric
            label="Output"
            value={hasTokenUsage ? formatNumber(tokenUsage!.output_tokens) : "--"}
          />
          <SummaryMetric
            label="Total"
            value={hasTokenUsage ? formatNumber(tokenUsage!.total_tokens) : "--"}
          />
        </div>
      </div>

      {usageRows.length > 0 && (
        <div className="inference-panel__models" aria-label="Per-turn model usage">
          <div className="inference-panel__section-title">This turn</div>
          {usageRows.map((model) => (
            <div key={model.role} className="inference-panel__model">
              <div className="inference-panel__model-topline">
                <span>{model.label}</span>
                <strong
                  className={`inference-panel__status inference-panel__status--${model.status}`}
                >
                  {statusLabel(model.status)}
                </strong>
              </div>
              <p>{model.name}</p>
              <small>
                {model.calls > 0
                  ? `${model.calls} ${model.calls === 1 ? "call" : "calls"}`
                  : model.detail}
              </small>
            </div>
          ))}
        </div>
      )}
    </aside>
  );
};

interface SummaryMetricProps {
  label: string;
  value: number | string;
}

const SummaryMetric: React.FC<SummaryMetricProps> = ({ label, value }) => (
  <div className="inference-panel__metric">
    <span>{value}</span>
    <p>{label}</p>
  </div>
);

const countByCategory = (events: InferenceActivity[]): Record<InferenceCategory, number> => {
  const counts: Record<InferenceCategory, number> = {
    vision: 0,
    language: 0,
    embedding: 0,
    safety: 0,
    memory: 0,
    system: 0,
  };

  events.forEach((event) => {
    counts[event.category] += 1;
  });

  return counts;
};

type UsageRowStatus = ModelUsageStatus | "running" | "queued";
type ActivityUsageStatus = UsageRowStatus | "complete";

const modelUsageRows = (
  models: ModelCapabilities,
  modelUsage: ModelUsage,
  tokenUsage: TokenUsage | null,
  events: InferenceActivity[]
): Array<{
  role: string;
  label: string;
  name: string;
  status: UsageRowStatus;
  calls: number;
  detail: string;
}> => {
  const eventStatus = statusByCategory(events);
  return [
    {
      role: "app_llm",
      label: "LLM",
      fallbackCalls: tokenUsage?.model_calls ?? 0,
      fallbackStatus: eventStatus.language,
    },
    {
      role: "vlm",
      label: "Omni",
      fallbackCalls: 0,
      fallbackStatus: eventStatus.vision,
    },
    {
      role: "text_embedding",
      label: "Text embedding",
      fallbackCalls: 0,
      fallbackStatus: eventStatus.embedding,
    },
    {
      role: "image_embedding",
      label: "Image embedding",
      fallbackCalls: 0,
      fallbackStatus: undefined,
    },
    {
      role: "content_safety",
      label: "Safety",
      fallbackCalls: countByCategory(events).safety,
      fallbackStatus: eventStatus.safety,
    },
  ]
    .map((row) => usageRow(row, models, modelUsage))
    .filter((row) => Boolean(row.name || row.status !== "disabled"));
};

const configuredModelName = (model: ModelCapabilities[string]): string => {
  if (!model || !model.model) return "Not configured";
  return model.model;
};

const usageRow = (
  row: {
    role: string;
    label: string;
    fallbackCalls: number;
    fallbackStatus?: ActivityUsageStatus;
  },
  models: ModelCapabilities,
  modelUsage: ModelUsage
) => {
  const model = models[row.role];
  const entry = modelUsage[row.role];
  const isEnabled = Boolean(model?.enabled && model?.model);
  const calls = entry?.calls ?? row.fallbackCalls;
  let status: UsageRowStatus = entry?.status ?? "not_used";

  if (!entry) {
    if (row.fallbackStatus === "running" || row.fallbackStatus === "queued") {
      status = row.fallbackStatus;
    } else if (calls > 0 || row.fallbackStatus === "complete") {
      status = "used";
    } else if (!isEnabled) {
      status = "disabled";
    }
  }

  return {
    role: row.role,
    label: row.label,
    name: configuredModelName(model),
    status,
    calls,
    detail: entry?.detail || (isEnabled ? "Available" : "Off"),
  };
};

const statusByCategory = (
  events: InferenceActivity[]
): Partial<Record<InferenceCategory, ActivityUsageStatus>> => {
  const statuses: Partial<Record<InferenceCategory, ActivityUsageStatus>> = {};
  events.forEach((event) => {
    const existing = statuses[event.category];
    if (existing === "failed" || event.status === "failed") {
      statuses[event.category] = "failed";
    } else if (existing === "running" || event.status === "running") {
      statuses[event.category] = "running";
    } else if (existing === "queued" || event.status === "queued") {
      statuses[event.category] = "queued";
    } else if (event.status === "complete") {
      statuses[event.category] = "complete";
    }
  });
  return statuses;
};

const usageCalls = (modelUsage: ModelUsage, roles: string[]): number => {
  return roles.reduce((total, role) => total + (modelUsage[role]?.calls ?? 0), 0);
};

const statusLabel = (status: UsageRowStatus): string => {
  if (status === "used") return "Used";
  if (status === "failed") return "Failed";
  if (status === "disabled") return "Off";
  if (status === "running") return "Running";
  if (status === "queued") return "Queued";
  return "Available";
};

const formatNumber = (value: number): string => {
  return new Intl.NumberFormat("en-US").format(value);
};

export default InferenceActivityPanel;
