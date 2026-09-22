// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";

import { MediaAnalysis } from "../../types";

interface MediaAnalysisCardProps {
  analysis: MediaAnalysis;
}

const MediaAnalysisCard: React.FC<MediaAnalysisCardProps> = ({ analysis }) => {
  const items = analysis.items ?? [];
  const strongest = items.reduce((best, item) => Math.max(best, item.pursued), 0);
  const hasSignal =
    strongest > 0 && items.some((item) => item.pursued < strongest);
  const pursued = hasSignal
    ? items.filter((item) => item.pursued === strongest)
    : [];
  const alsoSeen = hasSignal
    ? items.filter((item) => item.pursued < strongest)
    : items;
  const detail = dedupe([
    ...analysis.colors.slice(0, 3),
    ...analysis.materials.slice(0, 3),
    ...analysis.style.slice(0, 2),
    ...analysis.occasion.slice(0, 1),
  ]).slice(0, 8);

  return (
    <div className="media-analysis" aria-label="What was detected in your media">
      <div className="media-analysis__eyebrow">Nemotron VLM sees</div>
      {analysis.summary && (
        <p className="media-analysis__summary">{analysis.summary}</p>
      )}
      {items.length > 0 && (
        <div className="media-analysis__items">
          {pursued.map((item) => (
            <span key={item.label} className="media-analysis__item is-pursued">
              {item.label}
            </span>
          ))}
          {alsoSeen.map((item) => (
            <span key={item.label} className="media-analysis__item">
              {item.label}
            </span>
          ))}
        </div>
      )}
      {pursued.length > 0 && (
        <p className="media-analysis__focus">
          Looking for {joinWords(pursued.map((item) => item.label))}
          {alsoSeen.length > 0 && (
            <>
              {" — also saw "}
              {joinWords(alsoSeen.map((item) => item.label))}
            </>
          )}
        </p>
      )}
      {detail.length > 0 && (
        <p className="media-analysis__detail">{detail.join(" · ")}</p>
      )}
    </div>
  );
};

const dedupe = (values: string[]): string[] => {
  const seen = new Set<string>();
  return values.filter((value) => {
    const key = value.trim().toLowerCase();
    if (!key || seen.has(key)) return false;
    seen.add(key);
    return true;
  });
};

const joinWords = (values: string[]): string => {
  if (values.length <= 1) return values[0] ?? "";
  return `${values.slice(0, -1).join(", ")} and ${values[values.length - 1]}`;
};

export default MediaAnalysisCard;
