# STR 655 — Structural Stability Solver
## Spring 2026 | Assignment #1

**Excel VBA workbook for rigid-bar stability problems (Chapter 1, Chen & Lui)**

---

## 📁 Files

| File | Description |
|------|-------------|
| `STR655_StabilitySolver.xlsm` | Full Excel workbook (7 sheets, charts, pre-filled data) |
| `StabilitySolver_Macro.bas` | VBA macro code — paste into Excel Module |

---

## 🔬 Problems Covered

- **Problem 2a** — Two rigid bars (1.5L + L) with rotational spring k, load P at end
- **Problem 2b** — Two-bar symmetric system with two springs
- **Problem 2c** — Mirror of 2a (L + 1.5L)
- **Problem 2d** — Triangular truss, 45° geometry

---

## 📐 Analytical Results (Problem 2a)

| Method | Formula | Value (k=1) |
|--------|---------|-------------|
| Bifurcation | `Pcr = k(1/L1 + 1/L2)` | **5k/3 = 1.667 kN** |
| Energy (δ²V=0) | Same | **5k/3 = 1.667 kN** |
| Large deflection | `P/Pcr = θ/sinθ` | ≥ 1 (stable) |
| Equilibrium type | Stable symmetric bifurcation | Hardening post-buckling |

---

## 🚀 How to Use

1. Open `STR655_StabilitySolver.xlsm` → Enable Macros
2. Press `Alt+F11` → Insert → Module → Paste `StabilitySolver_Macro.bas`
3. Go to **Input** sheet → fill yellow cells
4. Right-click **▶ SOLVE** → Assign Macro → `SolveProblem` → Click
5. Results appear in Solution, PostBuckling, and Imperfection sheets

---

## 📊 Workbook Sheets

| Sheet | Contents |
|-------|----------|
| Instructions | Step-by-step guide |
| Input | Node coordinates, support types, spring constants, bar lengths |
| Solution | Pcr (bifurcation + energy), mode shape, stability classification |
| Drawing | ASCII structural sketch |
| PostBuckling | P/Pcr vs θ — 46 data points + chart |
| Imperfection | P/Pcr vs θ for θ₀ = 0, 0.5, 1, 2, 5, 10° + chart |
| VBA_Code | Full macro listing |

---

## 📚 Reference

> Chen, W.F. & Lui, E.M. (1987). *Structural Stability: Theory and Implementation*. Elsevier.

*PhD Candidate — Cairo University, Structural Engineering | STR 655 Spring 2026*