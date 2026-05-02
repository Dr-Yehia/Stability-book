# How to Import VBA Modules into Excel

## Method 1 — Import .bas files (Recommended)

1. Open Excel → Press **Alt + F11** to open VBA Editor
2. In the Project tree (left panel), right-click your workbook name
3. Choose **Import File...**
4. Import each `.bas` file in this order:
   - `Main.bas`
   - `TypeA_RigidBar.bas`
   - `TypeB_ElasticColumn.bas`
   - `TypeC_Truss.bas`
   - `Charts.bas`
5. Close VBA Editor
6. Save as **`.xlsm`** (macro-enabled workbook)

## Method 2 — Create sheets manually

Create these 3 sheets in your workbook:
- **INPUT** — where you enter problem data
- **RESULTS** — where solution appears automatically
- **PLOT_DATA** — raw data for the chart

Then add a button on INPUT sheet:
1. Insert → Shapes → Rectangle → draw a button
2. Right-click → Assign Macro → select `SOLVE`
3. Label it **"SOLVE"**

## INPUT Sheet Layout

```
Row 3:  [System Title]         C3 = problem name (label only)
Row 4:  [System Type]          C4 = A or B or C
Row 7:  [DOF / Sub-type]       C7 = 1 or 2 (Type A)  |  UNIFORM/STEPPED (Type B)
Row 10: [k  (kN.m/rad)]        C10 = rotational spring stiffness
Row 11: [kLin  (kN/m)]         C11 = linear spring (0 if none)
Row 14: [L1  (m)]              C14 = bar/segment length 1
Row 15: [L2  (m)]              C15 = bar/segment length 2 (0 for 1-DOF)
Row 18: [Imperfection?]        C18 = YES or NO
Row 19: [theta01 (rad)]        C19 = initial imperfection angle 1
Row 20: [theta02 (rad)]        C20 = initial imperfection angle 2 (0=auto)
Row 23: [Large deflection?]    C23 = YES or NO
Row 25: [Spring angle (deg)]   C25 = 45 (Type C only)
Row 26: [BC left]              C26 = HINGE / FIXED / FREE (Type B)
Row 27: [BC right]             C27 = HINGE / FIXED / FREE (Type B)
```
