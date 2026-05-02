# Structural Stability Solver — Chapter 1 (STR 655)

**Author:** Dr. Yehia Abdelhamid Attia  
**Course:** STR 655 — Structural Stability, Spring 2026  
**Reference:** W.F. Chen & E.M. Lui — *Structural Stability: Theory and Implementation*

---

## What This Solver Does

An Excel VBA workbook that solves **any Chapter 1 problem** from the textbook, assignment, and lecture notes:

| System Type | Problems Covered | Methods |
|---|---|---|
| **A** — Rigid bars + springs | Assignment (a)(b)(c), textbook examples | Bifurcation + Energy |
| **B** — Elastic columns (ODE) | Q1, Q2 from Dr. solution, textbook | Numerical root-finding |
| **C** — Truss with springs | Assignment (d), 45° spring systems | Energy method |

### Outputs per problem
- Pcr (critical load)
- Buckled shape (eigenvector)
- Large deflection post-buckling path
- Stability check (fundamental + post-buckling + critical point)
- Geometric imperfection effect
- Auto-generated equilibrium path plot

---

## File Structure

```
Stability-Book/
├── README.md
├── VBA_Modules/
│   ├── Main.bas              ← Entry point: routes to correct solver
│   ├── TypeA_RigidBar.bas    ← 1-DOF & 2-DOF rigid bar systems
│   ├── TypeB_ElasticColumn.bas ← Elastic column ODE solver
│   ├── TypeC_Truss.bas       ← Truss / spring systems
│   └── Charts.bas            ← Auto-plot equilibrium paths
└── StabilityChapter1_Solver.xlsm  ← Main Excel file (import all .bas into)
```

---

## How to Use

### Step 1 — Open Excel
1. Open `StabilityChapter1_Solver.xlsm`
2. Enable macros

### Step 2 — Fill INPUT Sheet

| Cell | Field | Example |
|------|-------|---------|
| C4 | System Type | `A` or `B` or `C` |
| C7 | Sub-type / DOF | `2` (for 2-DOF) or `STEPPED` |
| C10 | Spring k (kN.m/rad) | `200` |
| C11 | Linear spring kLin (kN/m) | `0` or value |
| C14 | Length L1 (m) | `1.5` |
| C15 | Length L2 (m) | `1.0` |
| C18 | Imperfection? | `YES` or `NO` |
| C19 | θ₀₁ (rad) | `0.1` |
| C20 | θ₀₂ (rad) | `0` (auto-calc if 0) |
| C23 | Large deflection? | `YES` or `NO` |
| C25 | Spring angle (Type C) | `45` |
| C26 | BC left (Type B) | `HINGE` / `FIXED` / `FREE` |
| C27 | BC right (Type B) | `HINGE` / `FIXED` / `FREE` |

### Step 3 — Click SOLVE
Results appear in **RESULTS** sheet with chart.

---

## Assignment Problems — Quick Reference

| Problem | Type | L1 | L2 | k | kLin | DOF |
|---------|------|----|----|---|------|-----|
| (a) | A | 1.5L | L | k | k/L² | 2 |
| (b) | A | L | L | k | 0 | 2 |
| (c) | A | L | 1.5L | k | k/L² | 2 |
| (d) | C | L | — | k | 0 | 1 |
| Q1 (Dr) | B | L/2 | L/2 | EI/2EI | — | cont |
| Q2 (Dr) | B | L/2 | L/2 | EI/2EI | — | cont |

---

## Physics Behind the Code

### Type A — Bifurcation Approach
```
Build stiffness matrix [K(P)] from equilibrium equations
Set det[K] = 0  →  solve quadratic/polynomial for Pcr
Eigenvector = buckled shape
```

### Type A — Energy Approach
```
Π = U + V
U = Σ(½k·θᵢ²) + Σ(½kL·Δᵢ²)
V = -P·ΣΔhorizontal
dΠ/dθᵢ = 0  →  same Pcr as bifurcation ✓
```

### Large Deflection
```
Keep exact trig (sinθ, cosθ) — no linearization
Compatibility: L1·sinθ1 = L2·sinθ2
Solve: P(θ1) from dΠ/dθ1 = 0
```

### Stability Check
```
d²Π/dθ² > 0  →  STABLE
d²Π/dθ² < 0  →  UNSTABLE
d²Π/dθ² = 0  →  check d³Π, d⁴Π (Taylor series)
```
