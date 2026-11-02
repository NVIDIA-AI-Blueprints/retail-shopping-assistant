// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * What the vision model saw in the shopper's photo or video.
 *
 * Arrives seconds before the products, while the catalog work is still running.
 *
 * Led by the model's own sentence, because it already writes the readable
 * version — "a woman models a cream cable-knit sweater with blue jeans and
 * brown block-heeled boots". An earlier draft printed colours, materials, style
 * and occasion as four labelled rows of comma-separated values and read like a
 * database record rather than someone describing a picture.
 *
 * The detail is still there, folded into one quiet line: enough to show the
 * model's range without asking anyone to read a schema.
 */

import React from "react";

import { MediaAnalysis } from "../../types";

interface MediaAnalysisCardProps {
  analysis: MediaAnalysis;
}

const MediaAnalysisCard: React.FC<MediaAnalysisCardProps> = ({ analysis }) => {
  const items = analysis.items ?? [];
  // The strongest signal, not any signal. The model often writes at least one
  // query touching every garment it saw, so "pursued at all" marks the whole
  // outfit and says nothing. What the shopper asked about is the item the
  // model chased *most*: for the fall video, the sweater at two against the
  // jeans and boots at one each.
  //
  // When everything ties there is no signal, and claiming one would be
  // inventing a focus the model did not have.
  const strongest = items.reduce((best, item) => Math.max(best, item.pursued), 0);
  const hasSignal =
    strongest > 0 && items.some((item) => item.pursued < strongest);
  const pursued = hasSignal
    ? items.filter((item) => item.pursued === strongest)
    : [];
  const alsoSeen = hasSignal
    ? items.filter((item) => item.pursued < strongest)
    : items;

  // One line rather than four rows. Deduplicated because the model repeats
  // itself across fields -- "cable knit" arrives as a material and as a style.
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

/** ["a","b","c"] -> "a, b and c" — a sentence, not a list. */
const joinWords = (values: string[]): string => {
  if (values.length <= 1) return values[0] ?? "";
  return `${values.slice(0, -1).join(", ")} and ${values[values.length - 1]}`;
};

export default MediaAnalysisCard;
