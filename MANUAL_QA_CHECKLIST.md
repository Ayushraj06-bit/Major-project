# Manual QA checklist

Click-through checklist for the dashboard. Covers the judgements automation
cannot make: whether something is *visibly* distinguishable, whether a wait feels
broken, whether a caveat is actually readable.

**Setup**

```bash
pip install -e .
streamlit run dashboard/app.py
```

Do the whole pass **twice**: once with your OS in light mode, once in dark. The
widget-theme defect (QA-3) only appeared in dark, and everything looked correct
in light.

Mark each row: ✅ pass · ❌ fail · ⚠️ works but feels wrong.

---

## A. First load

| # | Action | Expected | ✓ |
|---|--------|----------|---|
| A1 | Open the page | Title, left rail, and eight panels. No red traceback box. | |
| A2 | Read the sidebar footer | Says the data is synthetic and describes a generator, not dengue. | |
| A3 | Read the subtitle under the title | Names the frozen model, its training date, and the interval level. | |
| A4 | **Look at every label** in the rail and the Scenario panel | All readable. No invisible or washed-out text. **This is the one that failed before.** | |
| A5 | Look at the sliders, radios and checkbox | All **teal**. Any red control is a regression. | |
| A6 | Open the browser console (F12) | No errors other than Streamlit's own `metrics` fetch. | |

## B. Area selection

| # | Action | Expected | ✓ |
|---|--------|----------|---|
| B1 | Pick a different state in **Area** | Detail panel retitles; summary, recommendation, thresholds, forecast, attribution all change together. | |
| B2 | Immediately after B1, scan every panel | **Nothing still shows the previous state.** Stale panels are the failure mode here. | |
| B3 | Click a coloured tile on the map | Same as B1, and the rail's Area box updates to match. | |
| B4 | Click a **hatched** tile | Nothing happens. Those states are not in the study. | |
| B5 | Hover any tile | Tooltip gives the state name and either a rate or "not in this study". | |
| B6 | Click 5–6 tiles rapidly | Page keeps up, settles on the last one, no traceback. | |

## C. Period

| # | Action | Expected | ✓ |
|---|--------|----------|---|
| C1 | Drag the **Period** slider | Map recolours; map and Watchlist titles both show the new month. | |
| C2 | Compare the two titles | They name the **same** month. A mismatch means a panel is stale. | |
| C3 | Press **Play** | Period advances by itself, roughly one step per second. No traceback. | |
| C4 | Press **Pause** | Stops and stays stopped. | |
| C5 | Drag the slider to the far left, then far right | Both ends work; no crash at the boundaries. | |

## D. Forward projection — the judgement calls

| # | Action | Expected | ✓ |
|---|--------|----------|---|
| D1 | Select **Project 3 periods** | A dotted line appears past the solid one on "Observed and predicted". | |
| D2 | **Look at the chart** | The dotted projection is *obviously* different from the solid fitted line at a glance, without reading the caption. | |
| D3 | Find the shaded region labelled "recursive" | Present, and its left edge is where the model starts reading its own output. | |
| D4 | **Look at the interval band** across the projection | It **visibly widens** left to right. If it looks like a constant-width ribbon, uncertainty is not propagating. | |
| D5 | Switch to **Project 6 periods** | Projection extends further; band widens further still. | |
| D6 | Read the caption under the chart | Explains recursion, climatological normals, and that the widening is not a coverage guarantee. | |
| D7 | Switch back to **Fitted only** | Dotted line and shading disappear cleanly. | |

## E. Display options

| # | Action | Expected | ✓ |
|---|--------|----------|---|
| E1 | Untick **Show prediction intervals** | Bands vanish; the lines stay exactly where they were. | |
| E2 | Re-tick it | Bands return identically. | |
| E3 | Add a state under **Compare with** | Second line on "Compare states"; selected state in teal, the other grey. | |
| E4 | Check both charts' x-axes | Same span. Different spans invite a false comparison. | |
| E5 | Open the **Compare with** list | The currently selected state is **not** offered. | |
| E6 | Add a state, then select that same state as Area | It moves to Area and leaves the compare list. No crash. | |

## F. Scenario

| # | Action | Expected | ✓ |
|---|--------|----------|---|
| F1 | Set rainfall, percent, +20, **Run scenario** | A spinner with a sentence, then two lines: grey baseline, teal scenario. | |
| F2 | **Watch the wait** | A spinner appears — not blank space, not a frozen page. | |
| F3 | Read the evidence list | Reports the scenario, mean change, and how many cells were clamped. | |
| F4 | Read the last caption | Says the model learns correlation, not causation. | |
| F5 | Set the amount to **+50**, run again | Clamped fraction rises; an out-of-distribution note appears. | |
| F6 | Set **0** and run | Scenario line lies exactly on the baseline. Any visible gap is a bug. | |
| F7 | Pick a variable the model does not use, run | Says so explicitly rather than silently showing no change. | |

## G. Watchlist and recommendation

| # | Action | Expected | ✓ |
|---|--------|----------|---|
| G1 | Look at the Watchlist | Either ranked rows, or a sentence saying nothing crossed. **Never blank.** | |
| G2 | Tab to a watchlist row and press Return | Selects that state — keyboard reachable, not pointer-only. | |
| G3 | Read the recommendation card | Tier badge, the trigger value, the threshold it was compared against, and its sample size. | |
| G4 | Read the action list | Complete sentences, not cut off at the panel edge. | |
| G5 | Note the action catalogue line | Still says PLACEHOLDER — an open item, not a defect. | |

## H. Export

| # | Action | Expected | ✓ |
|---|--------|----------|---|
| H1 | Click **PNG** | A file downloads. | |
| H2 | Open it | Matches what is on screen, projection included. | |
| H3 | Click **PDF**, open it, zoom to 400% | Text and lines stay sharp — it is vector, not a scaled bitmap. | |

## I. Degradation and edge cases

| # | Action | Expected | ✓ |
|---|--------|----------|---|
| I1 | Select a state with no cached SHAP | Attribution panel explains why and names the script to run. | |
| I2 | Move the period to the earliest available | All panels render; no crash at the boundary. | |
| I3 | Change Area while a 6-period projection is still computing | Ends on the state you picked last; no panel shows the abandoned one. | |
| I4 | Resize the window narrow, then wide | Layout reflows; nothing clipped; no horizontal scrollbar on the body. | |
| I5 | Reload mid-projection | Recovers to a valid state. | |

---

## What to do with a failure

Note the row number, the OS theme you were in, and whether a red traceback box
appeared. A traceback is a crash; a wrong number with no traceback is worse and
worth flagging louder.
