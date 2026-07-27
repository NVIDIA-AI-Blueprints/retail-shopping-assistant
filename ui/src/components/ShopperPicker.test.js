// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";
import { createRoot } from "react-dom/client";

import ShopperPicker from "./ShopperPicker";

global.IS_REACT_ACT_ENVIRONMENT = true;

const profiles = [
  {
    shopper_profile_id: "shopper_morgan",
    display_name: "Morgan",
    shopper_type: "skeptical_researcher",
    behavior:
      "Probes for material, care burden, and repeated-wear practicality before choosing.",
    zipcode: "60601",
  },
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

const buttonWithText = (container, text) =>
  Array.from(container.querySelectorAll("button")).find((button) =>
    button.textContent.includes(text)
  );

const click = (element) => {
  React.act(() => {
    element.click();
  });
};

describe("ShopperPicker", () => {
  let container;
  let root;

  beforeEach(() => {
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(() => {
    React.act(() => {
      root.unmount();
    });
    container.remove();
  });

  test("supports keyboard, hover, and tap-style preview before confirming", () => {
    const onChange = jest.fn();
    // This test uses React createRoot directly, so rendering must be inside act.
    // eslint-disable-next-line testing-library/no-unnecessary-act
    React.act(() => {
      root.render(
        <ShopperPicker
          profiles={profiles}
          profilesStatus="ready"
          selectedShopperProfileId={null}
          disabled={false}
          onChange={onChange}
        />
      );
    });

    const trigger = buttonWithText(container, "Shop as");
    click(trigger);

    expect(container.querySelector('[role="dialog"]')).not.toBeNull();
    expect(container.querySelectorAll(".shopper-picker__option")).toHaveLength(6);
    expect(document.activeElement).toBe(
      container.querySelector(".shopper-picker__option")
    );

    const morgan = buttonWithText(container, "Morgan");
    React.act(() => {
      morgan.dispatchEvent(new MouseEvent("mouseover", { bubbles: true }));
    });
    expect(container.querySelector(".shopper-picker__details").textContent).toContain(
      profiles[0].behavior
    );
    expect(container.querySelector(".shopper-picker__details").textContent).toContain(
      profiles[0].zipcode
    );

    const riley = buttonWithText(container, "Riley");
    React.act(() => {
      riley.focus();
    });
    expect(container.querySelector(".shopper-picker__details").textContent).toContain(
      profiles[4].shopper_type
    );

    const casey = buttonWithText(container, "Casey");
    click(casey);
    expect(container.querySelector(".shopper-picker__details").textContent).toContain(
      profiles[2].behavior
    );

    click(buttonWithText(container, "Shop as Casey"));
    expect(onChange).toHaveBeenCalledWith("shopper_casey");
    expect(container.querySelector('[role="dialog"]')).toBeNull();
    expect(document.activeElement).toBe(trigger);

    click(trigger);
    click(container.querySelector('[aria-label="Close shopper picker"]'));
    expect(container.querySelector('[role="dialog"]')).toBeNull();
    expect(document.activeElement).toBe(trigger);
  });

  test("blocks opening while streaming or while profiles load", () => {
    const onChange = jest.fn();
    // This test uses React createRoot directly, so rendering must be inside act.
    // eslint-disable-next-line testing-library/no-unnecessary-act
    React.act(() => {
      root.render(
        <ShopperPicker
          profiles={profiles}
          profilesStatus="ready"
          selectedShopperProfileId={null}
          disabled
          onChange={onChange}
        />
      );
    });

    let trigger = buttonWithText(container, "Shop as");
    expect(trigger.disabled).toBe(true);
    click(trigger);
    expect(container.querySelector('[role="dialog"]')).toBeNull();

    // eslint-disable-next-line testing-library/no-unnecessary-act
    React.act(() => {
      root.render(
        <ShopperPicker
          profiles={[]}
          profilesStatus="loading"
          selectedShopperProfileId="shopper_morgan"
          disabled={false}
          onChange={onChange}
        />
      );
    });

    trigger = buttonWithText(container, "Shop as");
    expect(trigger.disabled).toBe(true);
    expect(trigger.textContent).toContain("Loading…");
    expect(trigger.title).toBe("Representative shoppers are loading.");
    click(trigger);
    expect(container.querySelector('[role="dialog"]')).toBeNull();
    expect(onChange).not.toHaveBeenCalled();
  });
});
