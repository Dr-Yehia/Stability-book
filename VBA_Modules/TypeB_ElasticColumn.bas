Attribute VB_Name = "TypeB_ElasticColumn"
Option Explicit

' ============================================================
'  TYPE B : Elastic Column ODE Solver
'  Handles uniform, stepped, and symmetric columns
'  Numerical bisection for transcendental equations
' ============================================================

Sub Solve_TypeB(ws As Worksheet)
    Dim EI1    As Double
    Dim EI2    As Double
    Dim L      As Double
    Dim Lseg1  As Double
    Dim bcLeft As String
    Dim bcRight As String
    Dim colType As String
    
    EI1     = CDbl(ws.Range("C10").Value)
    EI2     = CDbl(ws.Range("C11").Value)
    L       = CDbl(ws.Range("C14").Value)
    Lseg1   = CDbl(ws.Range("C15").Value)
    bcLeft  = UCase(Trim(ws.Range("C26").Value))
    bcRight = UCase(Trim(ws.Range("C27").Value))
    colType = UCase(Trim(ws.Range("C7").Value))
    
    Dim Pcr As Double, Ke As Double
    Dim Pi As Double: Pi = Application.WorksheetFunction.Pi()
    
    Select Case colType
        Case "UNIFORM"
            Pcr = Solve_UniformColumn(EI1, L, bcLeft, bcRight)
            Ke  = Sqr(Pi ^ 2 * EI1 / L ^ 2 / Pcr)
        Case "STEPPED", "SYMMETRIC"
            Pcr = Solve_SteppedColumn(EI1, EI2, L, Lseg1)
            Ke  = Sqr(Pi ^ 2 * EI1 / L ^ 2 / Pcr)
        Case Else
            Pcr = Solve_UniformColumn(EI1, L, bcLeft, bcRight)
            Ke  = 1
    End Select
    
    Dim wsR As Worksheet: Set wsR = ThisWorkbook.Sheets("RESULTS")
    wsR.Range("B2").Value  = "STRUCTURAL STABILITY SOLVER  -  Chapter 1"
    wsR.Range("B3").Value  = "Type B  |  Elastic Column  |  " & colType
    wsR.Range("B4").Value  = String(50, Chr(8212))
    wsR.Range("B6").Value  = "Pcr"
    wsR.Range("C6").Value  = Round(Pcr, 6)
    wsR.Range("D6").Value  = "kN"
    wsR.Range("B7").Value  = "Effective length factor Ke"
    wsR.Range("C7").Value  = Round(Ke, 4)
    wsR.Range("B8").Value  = "Euler load Pe (K=1)"
    wsR.Range("C8").Value  = Round(Pi ^ 2 * EI1 / L ^ 2, 6)
    wsR.Range("D8").Value  = "kN"
    wsR.Range("B9").Value  = "KL (effective length)"
    wsR.Range("C9").Value  = Round(Ke * L, 4)
    wsR.Range("D9").Value  = "m"
    wsR.Range("B11").Value = "Buckled shape:  y(x) = A*sin(kx)  where k=sqrt(Pcr/EI)"
    wsR.Columns("B:D").AutoFit
    MsgBox "Type B done! Check RESULTS.", vbInformation
End Sub

Function Solve_UniformColumn(EI As Double, L As Double, bcL As String, bcR As String) As Double
    Dim Pi As Double: Pi = Application.WorksheetFunction.Pi()
    Dim Pcr As Double
    If bcL = "HINGE" And bcR = "HINGE" Then
        Pcr = Pi ^ 2 * EI / L ^ 2
    ElseIf bcL = "FIXED" And bcR = "FIXED" Then
        Pcr = 4 * Pi ^ 2 * EI / L ^ 2
    ElseIf (bcL = "FIXED" And bcR = "FREE") Or (bcL = "FREE" And bcR = "FIXED") Then
        Pcr = Pi ^ 2 * EI / (4 * L ^ 2)
    ElseIf (bcL = "FIXED" And bcR = "HINGE") Or (bcL = "HINGE" And bcR = "FIXED") Then
        Pcr = 2.0457 * Pi ^ 2 * EI / L ^ 2
    Else
        Pcr = Pi ^ 2 * EI / L ^ 2
    End If
    Solve_UniformColumn = Pcr
End Function

Function Solve_SteppedColumn(EI1 As Double, EI2 As Double, L As Double, a As Double) As Double
    ' Bisection on: tan(k1*a) + (k2/k1)*tan(k2*(L-a)) = 0
    Dim b As Double: b = L - a
    Dim Pi As Double: Pi = Application.WorksheetFunction.Pi()
    Dim Plo As Double, Phi2 As Double, Pm As Double
    Plo  = 0.01 * Pi ^ 2 * EI1 / L ^ 2
    Phi2 = 5   * Pi ^ 2 * EI1 / L ^ 2
    Dim k1 As Double, k2 As Double, flo As Double, fm As Double
    Dim it As Integer
    For it = 1 To 200
        Pm  = (Plo + Phi2) / 2
        k1  = Sqr(Pm  / EI1): k2  = Sqr(Pm  / EI2)
        fm  = CharEq(k1, k2, a, b)
        k1  = Sqr(Plo / EI1): k2  = Sqr(Plo / EI2)
        flo = CharEq(k1, k2, a, b)
        If flo * fm < 0 Then Phi2 = Pm Else Plo = Pm
        If (Phi2 - Plo) < 0.000001 * Pm Then Exit For
    Next it
    Solve_SteppedColumn = Pm
End Function

Function CharEq(k1 As Double, k2 As Double, a As Double, b As Double) As Double
    Dim c1 As Double, c2 As Double
    c1 = Cos(k1 * a): c2 = Cos(k2 * b)
    If Abs(c1) < 0.0001 Or Abs(c2) < 0.0001 Then
        CharEq = 1E+30: Exit Function
    End If
    CharEq = Tan(k1 * a) + (k2 / k1) * Tan(k2 * b)
End Function
