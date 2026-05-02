Attribute VB_Name = "TypeA_RigidBar"
Option Explicit

' ================================================================
'  TYPE A : Rigid-Bar + Springs  (Chapter 1 - STR 655)
'
'  PHYSICS (derived from scratch for each problem):
'
'  System: Two rigid bars pinned at base, connected at hinge,
'          loaded by P at top.
'
'  Springs:
'    - Rotational spring k1 at BASE hinge (resists theta1)
'    - Rotational spring k2 at MID  hinge (resists theta2-theta1)
'    - Linear spring kL at MID point (horizontal, stiffness kL)
'
'  Equilibrium equations (small deflection, moment about hinges):
'
'  Bar 1 (length L1):
'    k1*theta1 + k2*(theta1-theta2) - P*(L1*theta1 + L2*theta2) = 0
'    => (k1+k2 - P*L1)*theta1 - k2*theta2 - P*L2*theta2 = 0
'    => (k1+k2-P*L1)*theta1 - (k2+P*L2)*theta2 = 0         ...(1)
'
'  Bar 2 (length L2):
'    k2*(theta2-theta1) + kL*L2*sin(t2)*L2 - P*L2*theta2 = 0
'    => -k2*theta1 + (k2 + kL*L2^2 - P*L2)*theta2 = 0      ...(2)
'
'  Matrix form: [K]{theta} = 0
'  K = | k1+k2-P*L1      -(k2+P*L2)  |
'      | -k2             k2+kL*L2^2-P*L2 |
'
'  det(K) = 0  =>  quadratic in P  =>  Pcr1, Pcr2
'
'  Assignment mapping:
'   (a): k1=k, k2=k, kL=k/L^2, L1=1.5L, L2=L     (use L=1 symbolic)
'   (b): k1=k, k2=k, kL=0,     L1=L,    L2=L
'   (c): k1=k, k2=k, kL=k/L^2, L1=L,    L2=1.5L
'   (d): Type C (truss)
' ================================================================

Sub Solve_TypeA(ws As Worksheet)
    Dim nDOF    As Integer
    Dim k1      As Double   ' rotational spring at base
    Dim k2      As Double   ' rotational spring at mid (k2=k for all assignment problems)
    Dim kL      As Double   ' linear spring stiffness (kN/m)
    Dim L1      As Double
    Dim L2      As Double
    Dim hasImp  As Boolean
    Dim theta01 As Double
    Dim theta02 As Double
    Dim doLarge As Boolean

    nDOF    = CInt(ws.Range("C7").Value)
    k1      = CDbl(ws.Range("C10").Value)
    k2      = CDbl(ws.Range("C12").Value)   ' NEW cell: k2 (mid spring)
    kL      = CDbl(ws.Range("C11").Value)   ' linear spring kN/m
    L1      = CDbl(ws.Range("C14").Value)
    L2      = CDbl(ws.Range("C15").Value)
    hasImp  = (UCase(Trim(ws.Range("C18").Value)) = "YES")
    theta01 = CDbl(ws.Range("C19").Value)
    theta02 = CDbl(ws.Range("C20").Value)
    doLarge = (UCase(Trim(ws.Range("C23").Value)) = "YES")

    If nDOF = 1 Then
        Call TypeA_1DOF(k1, kL, L1, hasImp, theta01, doLarge)
    ElseIf nDOF = 2 Then
        Call TypeA_2DOF(k1, k2, kL, L1, L2, hasImp, theta01, theta02, doLarge)
    End If
End Sub

' ----------------------------------------------------------------
'  1-DOF : single rigid bar, base spring k1, optional linear kL
'  Pcr from: dPi/dtheta = 0 => (k1 + kL*L^2 - P*L)*theta = 0
'  => Pcr = (k1 + kL*L^2) / L
' ----------------------------------------------------------------
Sub TypeA_1DOF(k1 As Double, kL As Double, L As Double, _
               hasImp As Boolean, th0 As Double, doLarge As Boolean)
    Dim wsR As Worksheet, wsP As Worksheet
    Set wsR = ThisWorkbook.Sheets("RESULTS")
    Set wsP = ThisWorkbook.Sheets("PLOT_DATA")

    Dim Pcr As Double
    Pcr = (k1 + kL * L ^ 2) / L     ' general formula, degenerates correctly

    wsR.Range("B2").Value  = "STRUCTURAL STABILITY SOLVER  -  Chapter 1  (STR 655)"
    wsR.Range("B3").Value  = "System Type : A  |  DOF : 1"
    wsR.Range("B4").Value  = String(55, Chr(8212))
    wsR.Range("B6").Value  = "SMALL DEFLECTION  (Bifurcation = Energy - identical)"
    wsR.Range("B7").Value  = "Pcr = (k + kL*L^2) / L"
    wsR.Range("C7").Value  = Round(Pcr, 8)
    wsR.Range("D7").Value  = "kN   [or k/L units if k symbolic]"
    wsR.Range("B8").Value  = "Buckled Shape: theta <> 0  (single symmetric mode)"
    wsR.Range("B10").Value = "STABILITY CHECK  (d2Pi/dtheta2)"
    wsR.Range("B11").Value = "  P < Pcr  ->  STABLE   |   P > Pcr  ->  UNSTABLE"
    wsR.Range("B12").Value = "  At Pcr: d4Pi/dtheta4 > 0  ->  Stable post-buckling (symmetric bifurcation)"

    If doLarge Then
        wsR.Range("B14").Value = "LARGE DEFLECTION POST-BUCKLING  (exact)"
        wsR.Range("B15").Value = "  P(theta) = (k1*theta + kL*L^2*sin(theta)) / (L*sin(theta))"
        wsR.Range("B16").Value = "  As theta->0: P->Pcr  [check of formula]"
        wsR.Range("B17").Value = "  Post-buckling slope > 0  =>  STABLE  (not sensitive to imperfection)"

        wsP.Range("A1").Value = "theta_rad"
        wsP.Range("B1").Value = "P_fundamental"
        wsP.Range("C1").Value = "P_postbuckling_perfect"
        wsP.Range("D1").Value = "P_imperfect"

        Dim i As Integer
        For i = 1 To 30
            wsP.Cells(i + 1, 1).Value = 0
            wsP.Cells(i + 1, 2).Value = Round(Pcr * (i - 1) / 25, 8)
        Next i

        Dim th As Double, Pval As Double
        For i = 1 To 50
            th = i * 0.02
            If Abs(Sin(th)) > 0.00001 Then
                Pval = (k1 * th + kL * L ^ 2 * Sin(th)) / (L * Sin(th))
                wsP.Cells(i + 32, 1).Value = Round(th, 6)
                wsP.Cells(i + 32, 3).Value = Round(Pval, 8)
            End If
        Next i

        If hasImp And Abs(th0) > 0.0001 Then
            wsR.Range("B19").Value = "GEOMETRIC IMPERFECTION  theta0 = " & th0 & " rad"
            wsR.Range("B20").Value = "  P(theta) = (k1*(theta-th0) + kL*L^2*(sin(t)-sin(t0))) / (L*sin(theta))"
            wsR.Range("B21").Value = "  Imperfect curve approaches perfect from BELOW (stable)"
            Dim rw As Integer: rw = 84
            For i = 1 To 50
                th = th0 + i * 0.02
                If Abs(Sin(th)) > 0.00001 Then
                    Pval = (k1 * (th - th0) + kL * L ^ 2 * (Sin(th) - Sin(th0))) / (L * Sin(th))
                    wsP.Cells(rw, 1).Value = Round(th, 6)
                    wsP.Cells(rw, 4).Value = Round(Pval, 8)
                    rw = rw + 1
                End If
            Next i
        End If

        Call CreateEquilibriumChart(Pcr, "1-DOF Rigid Bar")
    End If

    wsR.Columns("B:D").AutoFit
    MsgBox "1-DOF DONE!  Pcr = " & Round(Pcr, 6) & Chr(13) & "Check RESULTS sheet.", vbInformation
End Sub

' ================================================================
'  2-DOF : Two rigid bars L1 + L2
'
'  STIFFNESS MATRIX  (derived from equilibrium, NOT assumed generic)
'
'  Taking moments about each hinge:
'
'  Eq.1 (hinge at base, bar 1):
'    Restoring: k1*theta1 + k2*(theta1-theta2)
'    Destabilizing: P * (horizontal displacement of top)
'      top disp = L1*theta1 + L2*theta2
'      moment arm about base = L1
'    => k1*theta1 + k2*(theta1-theta2) - P*(L1*theta1 + L2*theta2) = 0  ... NO
'
'  CORRECT approach - virtual work / direct equilibrium:
'
'  Let delta_top = L1*theta1 + L2*theta2  (small angle)
'
'  Sum moments about BASE for whole system:
'    k1*theta1 + k2*(theta1-theta2) - P*(L1*theta1 + L2*theta2) = 0   ...(I)
'
'  Sum moments about MID hinge for bar 2 only:
'    k2*(theta2-theta1) + kL*(L2*theta2)*L2 - P*L2*theta2 = 0
'    => -k2*theta1 + (k2 + kL*L2^2 - P*L2)*theta2 = 0                 ...(II)
'
'  From (I):  (k1+k2-P*L1)*theta1 - (k2+P*L2)*theta2 = 0
'  From (II): -k2*theta1 + (k2+kL*L2^2-P*L2)*theta2 = 0
'
'  Matrix [K]{q} = 0:
'  K11 = k1+k2-P*L1
'  K12 = -(k2+P*L2)
'  K21 = -k2
'  K22 = k2 + kL*L2^2 - P*L2
'
'  det(K) = K11*K22 - K12*K21 = 0
'  => (k1+k2-P*L1)*(k2+kL*L2^2-P*L2) - k2*(k2+P*L2) = 0
'  Expand and collect P^2, P^1, P^0 terms -> quadratic in P
' ================================================================
Sub TypeA_2DOF(k1 As Double, k2 As Double, kL As Double, _
               L1 As Double, L2 As Double, _
               hasImp As Boolean, th01 As Double, th02_in As Double, _
               doLarge As Boolean)

    Dim wsR As Worksheet, wsP As Worksheet
    Set wsR = ThisWorkbook.Sheets("RESULTS")
    Set wsP = ThisWorkbook.Sheets("PLOT_DATA")

    ' ── Quadratic coefficients from det(K)=0 ──────────────────
    ' det = (k1+k2-P*L1)*(k2+kL*L2^2-P*L2) - k2*(k2+P*L2) = 0
    '
    ' Let A = k1+k2,  B = k2+kL*L2^2
    ' det = (A-P*L1)*(B-P*L2) - k2*(k2+P*L2)
    '     = A*B - A*P*L2 - B*P*L1 + P^2*L1*L2 - k2^2 - k2*P*L2
    '     = P^2*(L1*L2)
    '       + P*(-A*L2 - B*L1 - k2*L2)
    '       + (A*B - k2^2)
    '
    Dim A As Double: A = k1 + k2
    Dim B As Double: B = k2 + kL * L2 ^ 2

    Dim qa As Double, qb As Double, qc As Double
    qa = L1 * L2
    qb = -(A * L2 + B * L1 + k2 * L2)
    qc = A * B - k2 ^ 2

    Dim disc As Double
    disc = qb ^ 2 - 4 * qa * qc

    If disc < 0 Then
        MsgBox "No real Pcr found - check input values.", vbCritical
        Exit Sub
    End If

    Dim Pcr1 As Double, Pcr2 As Double
    Pcr1 = (-qb - Sqr(disc)) / (2 * qa)   ' smaller = critical
    Pcr2 = (-qb + Sqr(disc)) / (2 * qa)

    ' ── Buckled shapes from K21*theta1 + K22*theta2 = 0 ────────
    ' theta2/theta1 = k2 / (k2 + kL*L2^2 - Pcr*L2)
    Dim m1 As Double, m2 As Double
    Dim denom1 As Double, denom2 As Double
    denom1 = k2 + kL * L2 ^ 2 - Pcr1 * L2
    denom2 = k2 + kL * L2 ^ 2 - Pcr2 * L2
    m1 = IIf(Abs(denom1) > 0.000001, k2 / denom1, 1E+30)
    m2 = IIf(Abs(denom2) > 0.000001, k2 / denom2, 1E+30)

    ' ── Write RESULTS ────────────────────────────────────────────
    wsR.Range("B2").Value  = "STRUCTURAL STABILITY SOLVER  -  Chapter 1  (STR 655)"
    wsR.Range("B3").Value  = "Type A | 2-DOF | L1=" & L1 & "  L2=" & L2 & _
                              "  k1=" & k1 & "  k2=" & k2 & "  kL=" & kL
    wsR.Range("B4").Value  = String(55, Chr(8212))
    wsR.Range("B5").Value  = "STIFFNESS MATRIX:"
    wsR.Range("B6").Value  = "  K11 = k1+k2-P*L1     K12 = -(k2+P*L2)"
    wsR.Range("B7").Value  = "  K21 = -k2             K22 = k2+kL*L2^2-P*L2"
    wsR.Range("B8").Value  = "  Characteristic eq:  " & Round(qa, 6) & "*P^2  +  (" & _
                              Round(qb, 6) & ")*P  +  " & Round(qc, 6) & "  = 0"
    wsR.Range("B9").Value  = String(55, Chr(8212))
    wsR.Range("B10").Value = "SMALL DEFLECTION  (Bifurcation + Energy - same Pcr)"
    wsR.Range("B11").Value = "Pcr1  (governs)  ="
    wsR.Range("C11").Value = Round(Pcr1, 8)
    wsR.Range("D11").Value = "kN"
    wsR.Range("B12").Value = "Pcr2  (2nd mode) ="
    wsR.Range("C12").Value = Round(Pcr2, 8)
    wsR.Range("D12").Value = "kN"
    wsR.Range("B14").Value = "BUCKLED SHAPES"
    wsR.Range("B15").Value = "  Mode 1:  theta2/theta1 ="
    wsR.Range("C15").Value = Round(m1, 6)
    wsR.Range("B16").Value = "  Mode 2:  theta2/theta1 ="
    wsR.Range("C16").Value = Round(m2, 6)
    wsR.Range("B18").Value = "STABILITY CHECK"
    wsR.Range("B19").Value = "  P < Pcr1  ->  STABLE"
    wsR.Range("B20").Value = "  P > Pcr1  ->  UNSTABLE (buckled in Mode 1)"

    If doLarge Then
        wsR.Range("B22").Value = "LARGE DEFLECTION  (exact trigonometry)"
        wsR.Range("B23").Value = "  Compatibility: L1*sin(t1) = L2*sin(t2)"
        wsR.Range("B24").Value = "  => t2 = arcsin(L1/L2 * sin(t1))"
        wsR.Range("B25").Value = "  P from dPi/dt1 = 0 (see post-buckling path below)"

        wsP.Range("A1").Value = "theta1_rad"
        wsP.Range("B1").Value = "P_fundamental"
        wsP.Range("C1").Value = "P_perfect_postbuckling"
        wsP.Range("D1").Value = "P_imperfect"
        wsP.Range("E1").Value = "d2Pi_sign"

        ' Fundamental path (theta=0, P varies 0 to 1.2*Pcr1)
        Dim j As Integer
        For j = 1 To 30
            wsP.Cells(j + 1, 1).Value = 0
            wsP.Cells(j + 1, 2).Value = Round(Pcr1 * (j - 1) / 25, 8)
        Next j

        ' Perfect post-buckling path (exact large deflection)
        Dim sinRatio As Double: sinRatio = L1 / L2
        Dim row As Integer:     row = 33
        Dim th1 As Double, th2 As Double, Ppb As Double, s2 As Double

        For j = 1 To 60
            th1 = j * 0.01
            s2 = sinRatio * Sin(th1)
            If Abs(s2) < 0.9999 Then
                th2 = Application.WorksheetFunction.Asin(s2)
                ' Large-deflection equilibrium (moment about base, exact):
                ' P = [k1*th1 + k2*(th1-th2) + kL*(L2*sin(th2))*L2*cos(th2)] /
                '     [L1*sin(th1) + L2*sin(th2)]
                Dim numP As Double, denP As Double
                numP = k1 * th1 + k2 * (th1 - th2) + kL * L2 ^ 2 * Sin(th2) * Cos(th2)
                denP = L1 * Sin(th1) + L2 * Sin(th2)
                If Abs(denP) > 0.000001 Then
                    Ppb = numP / denP
                    wsP.Cells(row, 1).Value = Round(th1, 6)
                    wsP.Cells(row, 3).Value = Round(Ppb, 8)
                    ' d2Pi/dth1^2 sign (stability on post-buckling path)
                    Dim d2Pi As Double
                    Dim dt2_dt1 As Double
                    dt2_dt1 = (L1 * Cos(th1)) / (L2 * Cos(th2))
                    d2Pi = k1 + k2 * (1 - dt2_dt1) ^ 2 + _
                           kL * L2 ^ 2 * (Cos(2 * th2) * dt2_dt1 ^ 2) - _
                           Ppb * (L1 * Cos(th1) + L2 * Cos(th2) * dt2_dt1)
                    wsP.Cells(row, 5).Value = IIf(d2Pi > 0, "STABLE", "UNSTABLE")
                    row = row + 1
                End If
            End If
        Next j

        ' Imperfect post-buckling path
        If hasImp And Abs(th01) > 0.0001 Then
            ' Auto-compute theta02 from compatibility
            Dim s02 As Double: s02 = sinRatio * Sin(th01)
            Dim th02 As Double
            If Abs(s02) < 0.9999 Then
                th02 = Application.WorksheetFunction.Asin(s02)
            Else
                th02 = th02_in
            End If

            wsR.Range("B27").Value = "GEOMETRIC IMPERFECTION"
            wsR.Range("B28").Value = "  th01 = " & th01 & " rad   th02 = " & Round(th02, 6) & " rad  (from compatibility)"
            wsR.Range("B29").Value = "  P from moment equilibrium with initial angles"

            Dim ri As Integer: ri = row + 2
            For j = 1 To 60
                th1 = th01 + j * 0.01
                s2 = sinRatio * Sin(th1)
                If Abs(s2) < 0.9999 Then
                    th2 = Application.WorksheetFunction.Asin(s2)
                    numP = k1 * (th1 - th01) + k2 * ((th1 - th2) - (th01 - th02)) + _
                           kL * L2 ^ 2 * (Sin(th2) * Cos(th2) - Sin(th02) * Cos(th02))
                    denP = L1 * Sin(th1) + L2 * Sin(th2)
                    If Abs(denP) > 0.000001 Then
                        Ppb = numP / denP
                        wsP.Cells(ri, 1).Value = Round(th1, 6)
                        wsP.Cells(ri, 4).Value = Round(Ppb, 8)
                        ri = ri + 1
                    End If
                End If
            Next j

            ' Post-buckling verdict
            wsR.Range("B31").Value = "POST-BUCKLING TYPE:"
            If wsP.Cells(33, 5).Value = "STABLE" Then
                wsR.Range("C31").Value = "STABLE -> NOT imperfection sensitive"
            Else
                wsR.Range("C31").Value = "UNSTABLE -> IMPERFECTION SENSITIVE (snap-through possible)"
            End If
        End If

        Call CreateEquilibriumChart(Pcr1, "2-DOF  L1=" & L1 & "  L2=" & L2)
    End If

    wsR.Columns("B:E").AutoFit
    MsgBox "2-DOF DONE!" & Chr(13) & _
           "Pcr1 = " & Round(Pcr1, 6) & Chr(13) & _
           "Pcr2 = " & Round(Pcr2, 6) & Chr(13) & _
           "Mode1 ratio theta2/theta1 = " & Round(m1, 4) & Chr(13) & _
           "Check RESULTS sheet.", vbInformation
End Sub
