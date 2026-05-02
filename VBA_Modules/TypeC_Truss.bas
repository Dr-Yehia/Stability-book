Attribute VB_Name = "TypeC_Truss"
Option Explicit

' ============================================================
'  TYPE C : Truss with Springs (e.g. 45-degree)
'  Energy method - full large deflection
' ============================================================

Sub Solve_TypeC(ws As Worksheet)
    Dim k      As Double: k      = CDbl(ws.Range("C10").Value)
    Dim L      As Double: L      = CDbl(ws.Range("C14").Value)
    Dim angle  As Double: angle  = CDbl(ws.Range("C25").Value)
    Dim hasImp As Boolean: hasImp = (UCase(Trim(ws.Range("C18").Value)) = "YES")
    Dim th0    As Double: th0    = CDbl(ws.Range("C19").Value)
    Dim doLarge As Boolean: doLarge = (UCase(Trim(ws.Range("C23").Value)) = "YES")
    
    ' For 45-deg springs: Pcr = k*L
    Dim Pcr As Double: Pcr = k * L
    
    Dim wsR As Worksheet: Set wsR = ThisWorkbook.Sheets("RESULTS")
    wsR.Range("B2").Value = "STRUCTURAL STABILITY SOLVER  -  Chapter 1"
    wsR.Range("B3").Value = "Type C  |  Truss Springs at " & angle & " deg"
    wsR.Range("B4").Value = String(50, Chr(8212))
    wsR.Range("B6").Value  = "Energy approach: U = kL2/2*(sin2t+(1-cost)2)  W=PL(1-cost)"
    wsR.Range("B7").Value  = "dPi/dtheta = 0  =>  P = kL (for theta<>0)"
    wsR.Range("B9").Value  = "Pcr"
    wsR.Range("C9").Value  = Round(Pcr, 6)
    wsR.Range("D9").Value  = "kN"
    wsR.Range("B11").Value = "Post-buckling path: P = kL = CONSTANT (neutral)"
    wsR.Range("B12").Value = "Fundamental path: STABLE if P < kL"
    wsR.Range("B13").Value = "Critical point: d4Pi > 0 -> STABLE"
    
    If hasImp And Abs(th0) > 0.0001 Then
        wsR.Range("B15").Value = "IMPERFECTION  theta0 = " & th0 & " rad"
        wsR.Range("B16").Value = "P = kL*cos(t)*(1 - sin(t0)/sin(t))"
        wsR.Range("B17").Value = "Pmax at sin(t0) = sin^3(t)"
        Dim smax As Double: smax = Sin(th0) ^ (1 / 3)
        If smax <= 1 Then
            Dim thmax As Double
            thmax = Application.WorksheetFunction.Asin(smax)
            wsR.Range("B18").Value = "Pmax (imperfect) ="
            wsR.Range("C18").Value = Round(k * L * Cos(thmax) ^ 3, 6)
            wsR.Range("D18").Value = "kN"
            wsR.Range("B19").Value = "System IS sensitive to imperfection (unstable post-buckling)"
        End If
    End If
    
    wsR.Columns("B:D").AutoFit
    MsgBox "Type C done! Check RESULTS.", vbInformation
End Sub
