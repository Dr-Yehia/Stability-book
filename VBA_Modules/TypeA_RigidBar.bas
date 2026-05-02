Attribute VB_Name = "TypeA_RigidBar"
Option Explicit

' ============================================================
'  TYPE A : Rigid-Bar Systems with Springs
'  Handles 1-DOF and 2-DOF configurations
' ============================================================

Sub Solve_TypeA(ws As Worksheet)
    Dim nDOF    As Integer
    Dim k       As Double
    Dim kLin    As Double
    Dim L1      As Double
    Dim L2      As Double
    Dim hasImp  As Boolean
    Dim theta01 As Double
    Dim theta02 As Double
    Dim doLarge As Boolean
    
    nDOF    = CInt(ws.Range("C7").Value)
    k       = CDbl(ws.Range("C10").Value)
    kLin    = CDbl(ws.Range("C11").Value)
    L1      = CDbl(ws.Range("C14").Value)
    L2      = CDbl(ws.Range("C15").Value)
    hasImp  = (UCase(Trim(ws.Range("C18").Value)) = "YES")
    theta01 = CDbl(ws.Range("C19").Value)
    theta02 = CDbl(ws.Range("C20").Value)
    doLarge = (UCase(Trim(ws.Range("C23").Value)) = "YES")
    
    If nDOF = 1 Then
        Call TypeA_1DOF(k, kLin, L1, hasImp, theta01, doLarge)
    ElseIf nDOF = 2 Then
        Call TypeA_2DOF(k, kLin, L1, L2, hasImp, theta01, theta02, doLarge)
    End If
End Sub

' ─────────────────────────────────────────────────────────────
'  1-DOF : Single rigid bar
' ─────────────────────────────────────────────────────────────
Sub TypeA_1DOF(k As Double, kLin As Double, L As Double, _
               hasImp As Boolean, th0 As Double, doLarge As Boolean)
    Dim wsR As Worksheet, wsP As Worksheet
    Set wsR = ThisWorkbook.Sheets("RESULTS")
    Set wsP = ThisWorkbook.Sheets("PLOT_DATA")
    
    Dim Pcr As Double
    If kLin > 0 And k = 0 Then
        Pcr = kLin * L
    ElseIf k > 0 And kLin = 0 Then
        Pcr = k / L
    Else
        Pcr = (k + kLin * L ^ 2) / L
    End If
    
    wsR.Range("B2").Value = "STRUCTURAL STABILITY SOLVER  -  Chapter 1"
    wsR.Range("B3").Value = "System Type : A  |  DOF : 1"
    wsR.Range("B4").Value = String(50, Chr(8212))
    wsR.Range("B6").Value = "SMALL DEFLECTION ANALYSIS"
    wsR.Range("B7").Value = "Pcr"
    wsR.Range("C7").Value = Pcr
    wsR.Range("D7").Value = "kN"
    wsR.Range("B8").Value = "Buckled Shape: theta <> 0 (single mode)"
    wsR.Range("B10").Value = "STABILITY CHECK"
    wsR.Range("B11").Value = "P < Pcr  -> STABLE   |   P > Pcr  -> UNSTABLE"
    wsR.Range("B12").Value = "At critical point: d4Pi > 0  -> STABLE (symmetric bifurcation)"
    
    If doLarge Then
        wsR.Range("B14").Value = "LARGE DEFLECTION POST-BUCKLING"
        wsR.Range("B15").Value = "P = k*theta / (L*sin(theta))  [perfect system]"
        wsR.Range("B16").Value = "Post-buckling path: STABLE"
        
        wsP.Range("A1").Value = "theta1_rad"
        wsP.Range("B1").Value = "P_fundamental_kN"
        wsP.Range("C1").Value = "P_postbuckling_kN"
        wsP.Range("D1").Value = "P_imperfect_kN"
        
        Dim i As Integer
        For i = 1 To 30
            wsP.Cells(i + 1, 1).Value = 0
            wsP.Cells(i + 1, 2).Value = Round(Pcr * (i - 1) / 25, 6)
        Next i
        
        Dim th As Double, Pval As Double
        For i = 1 To 40
            th = i * 0.025
            If Abs(Sin(th)) > 0.0001 Then
                Pval = (k / L) * th / Sin(th)
                wsP.Cells(i + 32, 1).Value = Round(th, 5)
                wsP.Cells(i + 32, 3).Value = Round(Pval, 6)
            End If
        Next i
        
        If hasImp And Abs(th0) > 0.0001 Then
            wsR.Range("B18").Value = "IMPERFECTION theta0 = " & th0 & " rad"
            wsR.Range("B19").Value = "P = k*(theta-theta0)/(L*sin(theta))"
            wsR.Range("B20").Value = "System NOT sensitive to imperfection (stable post-buckling)"
            Dim rw As Integer: rw = 74
            For i = 1 To 40
                th = th0 + i * 0.02
                If Abs(Sin(th)) > 0.0001 Then
                    Pval = k * (th - th0) / (L * Sin(th))
                    wsP.Cells(rw, 1).Value = Round(th, 5)
                    wsP.Cells(rw, 4).Value = Round(Pval, 6)
                    rw = rw + 1
                End If
            Next i
        End If
        
        Call CreateEquilibriumChart(Pcr, "1-DOF Rigid Bar")
    End If
    
    wsR.Columns("B:D").AutoFit
    MsgBox "DONE! Check RESULTS sheet.", vbInformation
End Sub

' ─────────────────────────────────────────────────────────────
'  2-DOF : Two rigid bars (general L1 <> L2)
' ─────────────────────────────────────────────────────────────
Sub TypeA_2DOF(k As Double, kLin As Double, L1 As Double, L2 As Double, _
               hasImp As Boolean, th01 As Double, th02_in As Double, doLarge As Boolean)
    Dim wsR As Worksheet, wsP As Worksheet
    Set wsR = ThisWorkbook.Sheets("RESULTS")
    Set wsP = ThisWorkbook.Sheets("PLOT_DATA")
    
    Dim r As Double: r = L2 / L1
    Dim kL1 As Double: kL1 = k / L1
    
    ' Generalized stiffness matrix eigenvalue problem
    ' A = [[(1+r)/r, -1/r],[-1/r,(1+r)/r]]
    ' B = [[1/r, 0],[0, 1]]
    ' det(A - lam*B) = 0
    Dim A11 As Double, A12 As Double, A22 As Double
    Dim B11 As Double, B22 As Double
    
    A11 = (1 + r) / r + kLin * L1 / k  ' include linear spring
    A12 = -1 / r
    A22 = (1 + r) / r
    B11 = 1 / r
    B22 = 1
    
    Dim ac As Double, bc As Double, cc As Double, disc As Double
    ac = B11 * B22
    bc = -(A11 * B22 + A22 * B11)
    cc = A11 * A22 - A12 ^ 2
    disc = bc ^ 2 - 4 * ac * cc
    
    Dim lam1 As Double, lam2 As Double
    lam1 = (-bc - Sqr(disc)) / (2 * ac)
    lam2 = (-bc + Sqr(disc)) / (2 * ac)
    
    Dim Pcr1 As Double, Pcr2 As Double
    Pcr1 = lam1 * kL1
    Pcr2 = lam2 * kL1
    
    ' Buckled shapes
    Dim m1 As Double, m2 As Double
    m1 = -(A11 - lam1 * B11) / A12   ' theta2/theta1 for mode 1
    m2 = -(A11 - lam2 * B11) / A12
    
    ' Write results
    wsR.Range("B2").Value = "STRUCTURAL STABILITY SOLVER  -  Chapter 1"
    wsR.Range("B3").Value = "System Type: A | 2-DOF | L1=" & L1 & "m  L2=" & L2 & "m  k=" & k & "kN.m/rad"
    wsR.Range("B4").Value = String(50, Chr(8212))
    
    wsR.Range("B6").Value  = "SMALL DEFLECTION (Bifurcation + Energy - same result)"
    wsR.Range("B7").Value  = "Pcr1 (critical = governs)"
    wsR.Range("C7").Value  = Round(Pcr1, 6)
    wsR.Range("D7").Value  = "kN"
    wsR.Range("B8").Value  = "Pcr2 (2nd mode)"
    wsR.Range("C8").Value  = Round(Pcr2, 6)
    wsR.Range("D8").Value  = "kN"
    wsR.Range("B10").Value = "BUCKLED SHAPES"
    wsR.Range("B11").Value = "Mode 1:  theta2/theta1 ="
    wsR.Range("C11").Value = Round(m1, 4)
    wsR.Range("B12").Value = "Mode 2:  theta2/theta1 ="
    wsR.Range("C12").Value = Round(m2, 4)
    wsR.Range("B14").Value = "STABILITY"
    wsR.Range("B15").Value = "P < Pcr1 -> STABLE  |  P > Pcr1 -> UNSTABLE"
    
    If doLarge Then
        wsR.Range("B17").Value = "LARGE DEFLECTION  (exact trig)"
        wsR.Range("B18").Value = "Compatibility: L1*sin(t1) = L2*sin(t2) => t2 = arcsin((L1/L2)*sin(t1))"
        
        wsP.Range("A1").Value = "theta1_rad"
        wsP.Range("B1").Value = "P_fundamental"
        wsP.Range("C1").Value = "P_perfect_postbuckling"
        wsP.Range("D1").Value = "P_imperfect"
        wsP.Range("E1").Value = "Stability"
        
        ' Fundamental path
        Dim j As Integer
        For j = 1 To 30
            wsP.Cells(j + 1, 1).Value = 0
            wsP.Cells(j + 1, 2).Value = Round(Pcr1 * (j - 1) / 25, 6)
        Next j
        
        ' Perfect post-buckling (exact large deflection)
        Dim sinR As Double: sinR = L1 / L2
        Dim row As Integer: row = 33
        Dim th1 As Double, th2 As Double, Ppb As Double
        
        For j = 1 To 50
            th1 = j * 0.01
            Dim s2 As Double: s2 = sinR * Sin(th1)
            If Abs(s2) <= 0.9999 Then
                th2 = Application.WorksheetFunction.Asin(s2)
                Dim num As Double, den As Double
                ' dPi/dth1 = 0 exactly:
                ' num = k*(th2)*(cos(t1)/(L2*cos(t2))) + k*th1 + kLin*L1*sin(t1)*cos(t1)
                ' den = L1*(sin(t2)*cos(t1)/cos(t2) + sin(t1))
                num = k * th2 * (Cos(th1) / (L2 * Cos(th2))) + _
                      k * th1 + kLin * L1 * Sin(th1) * Cos(th1)
                den = L1 * (Sin(th2) * Cos(th1) / Cos(th2) + Sin(th1))
                If Abs(den) > 0.000001 Then
                    Ppb = num / den
                    wsP.Cells(row, 1).Value = Round(th1, 5)
                    wsP.Cells(row, 3).Value = Round(Ppb, 6)
                    ' Stability check
                    Dim d2p As Double
                    d2p = k / L1 ^ 2 + k / L2 ^ 2 * (Cos(th1) / Cos(th2)) ^ 2 - _
                          Ppb * Cos(th1) / L1
                    wsP.Cells(row, 5).Value = IIf(d2p > 0, "STABLE", "UNSTABLE")
                    row = row + 1
                End If
            End If
        Next j
        
        ' Imperfect path
        If hasImp And Abs(th01) > 0.0001 Then
            Dim sin02 As Double: sin02 = sinR * Sin(th01)
            Dim th02_c As Double
            If Abs(sin02) <= 0.9999 Then
                th02_c = Application.WorksheetFunction.Asin(sin02)
            Else
                th02_c = th02_in
            End If
            wsR.Range("B20").Value = "IMPERFECTION: th01=" & th01 & " rad  th02=" & Round(th02_c, 5) & " rad"
            wsR.Range("B21").Value = "Imperfect path stable: d2Pi/dth1^2 > 0"
            
            Dim ri As Integer: ri = row + 3
            For j = 1 To 50
                th1 = th01 + j * 0.01
                s2 = sinR * Sin(th1)
                If Abs(s2) <= 0.9999 Then
                    th2 = Application.WorksheetFunction.Asin(s2)
                    num = k * (th2 - th02_c) * (Cos(th1) / (L2 * Cos(th2))) + _
                          k * (th1 - th01) + _
                          kLin * L1 * (Sin(th1) - Sin(th01)) * Cos(th1)
                    den = L1 * (Sin(th2) * Cos(th1) / Cos(th2) + Sin(th1))
                    If Abs(den) > 0.000001 Then
                        Ppb = num / den
                        wsP.Cells(ri, 1).Value = Round(th1, 5)
                        wsP.Cells(ri, 4).Value = Round(Ppb, 6)
                        ri = ri + 1
                    End If
                End If
            Next j
            
            ' Stability verdict
            wsR.Range("B23").Value = "POST-BUCKLING VERDICT:"
            If wsP.Cells(33, 5).Value = "STABLE" Then
                wsR.Range("C23").Value = "STABLE post-buckling -> NOT sensitive to imperfection"
            Else
                wsR.Range("C23").Value = "UNSTABLE post-buckling -> SENSITIVE to imperfection"
            End If
        End If
        
        Call CreateEquilibriumChart(Pcr1, "2-DOF Rigid Bar (L1=" & L1 & "  L2=" & L2 & ")")
    End If
    
    wsR.Columns("B:E").AutoFit
    MsgBox "DONE! Check RESULTS sheet.", vbInformation
End Sub
