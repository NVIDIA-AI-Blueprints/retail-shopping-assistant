// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * What the vision model saw in the shopper's photo or video.
 *
 * Arrives seconds before the products, while the catalog work is still running,
 * so the shopper watches their image being read rather than waiting at a blank
 * screen.
 *
 * Everything detected is shown. The items the assistant is actually searching
 * for are marked, so "I like the top" reads as *saw the jeans and boots too,
 * looking for the sweater* — which is what happened, rather than a claim about
 * it. Which items those are comes from the model's own searches, not from us.
 */

import React from "react";

import { MediaAnalysis } from "../../types";

interface MediaAnalysisCardProps {
  analysis: MediaAnalysis;
}

const MediaAnalysisCard: React.FC<MediaAnalysisCardProps> = ({ analysis }) => {
  const items = analysis.items ?? [];
  const pursued = items.filter((item) => item.pursued > 0);
  const alsoSeen = items.filter((item) => item.pursued === 0);

  return (
    <div className="media-analysis" aria-label="What was detected in your media">
      <div className="media-analysis__header">
        <span className="media-analysis__eyebrow">Nemotron VLM</span>
        <span className="media-analysis__title">saw this</span>
      </div>

      {analysis.summary && (
        <p className="media-analysis__summary">{analysis.summary}</p>
      )}

      {items.length > 0 && (
        <div className="media-analysis__items">
          {pursued.map((item) => (
            <span
              key={item.label}
              className="media-analysis__item is-pursued"
              title={`Searching for this — ${item.pursued} ${
                item.pursued === 1 ? "search" : "searches"
              }`}
            >
              {item.label}
            </span>
          ))}
          {alsoSeen.map((item) => (
            <span
              key={item.label}
              className="media-analysis__item"
              title="Seen, but not what you asked about"
            >
              {item.label}
            </span>
          ))}
        </div>
      )}

      {pursued.length > 0 && alsoSeen.length > 0 && (
        <p className="media-analysis__focus-note">
          Also seen, not searched: {alsoSeen.map((i) => i.label).join(", ")}
        </p>
      )}

      <Facts label="Colours" values={analysis.colors} />
      <Facts label="Materials" values={analysis.materials} />
      <Facts label="Style" values={analysis.style} />
      <Facts label="Occasion" values={analysis.occasion} />

      {analysis.queries.length > 0 && (
        <div className="media-analysis__row">
          <span className="media-analysis__label">Searching</span>
          <span className="media-analysis__values">
            {analysis.queries.map((query) => (
              <em key={query} className="media-analysis__query">
                {query}
              </em>
            ))}
          </span>
        </div>
      )}
    </div>
  );
};

const Facts: React.FC<{ label: string; values: string[] }> = ({
  label,
  values,
}) =>
  values.length === 0 ? null : (
    <div className="media-analysis__row">
      <span className="media-analysis__label">{label}</span>
      <span className="media-analysis__values">{values.join(" · ")}</span>
    </div>
  );

export default MediaAnalysisCard;
