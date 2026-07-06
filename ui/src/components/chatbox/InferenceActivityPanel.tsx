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
  TokenUsage,
} from "../../types";

interface InferenceActivityPanelProps {
  events: InferenceActivity[];
  models: ModelCapabilities;
  tokenUsage: TokenUsage | null;
}

const InferenceActivityPanel: React.FC<InferenceActivityPanelProps> = ({
  events,
  models,
  tokenUsage,
}) => {
  const groupedCounts = countByCategory(events);
  const modelStack = modelRows(models);
  const hasTokenUsage = Boolean(tokenUsage && tokenUsage.model_calls > 0);
  const llmCallCount = hasTokenUsage ? tokenUsage!.model_calls : groupedCounts.language;

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
        <SummaryMetric label="Omni calls" value={groupedCounts.vision} />
        <SummaryMetric label="LLM calls" value={llmCallCount} />
        <SummaryMetric label="Embedding calls" value={groupedCounts.embedding} />
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

      {modelStack.length > 0 && (
        <div className="inference-panel__models" aria-label="Configured model stack">
          <div className="inference-panel__section-title">Models</div>
          {modelStack.map((model) => (
            <div key={model.role} className="inference-panel__model">
              <span>{model.label}</span>
              <p>{model.name}</p>
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

const modelRows = (models: ModelCapabilities): Array<{
  role: string;
  label: string;
  name: string;
}> => {
  const rows = [
    {
      role: "app_llm",
      label: "LLM",
      name: enabledModelName(models.app_llm),
    },
    {
      role: "vlm",
      label: "Omni",
      name: enabledModelName(models.vlm),
    },
    {
      role: "text_embedding",
      label: "Text embedding",
      name: enabledModelName(models.text_embedding),
    },
    {
      role: "image_embedding",
      label: "Image embedding",
      name: enabledModelName(models.image_embedding),
    },
    {
      role: "content_safety",
      label: "Safety",
      name: enabledModelName(models.content_safety),
    },
    {
      role: "topic_control",
      label: "Topic control",
      name: enabledModelName(models.topic_control),
    },
  ];

  return rows.filter((row) => Boolean(row.name));
};

const enabledModelName = (model: ModelCapabilities[string]): string => {
  if (!model || !model.enabled || !model.model) return "";
  return model.model;
};

const formatNumber = (value: number): string => {
  return new Intl.NumberFormat("en-US").format(value);
};

export default InferenceActivityPanel;
