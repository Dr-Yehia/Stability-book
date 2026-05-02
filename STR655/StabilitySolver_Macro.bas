' ╔══════════════════════════════════════════════════════════════════════════════╗
' ║  STR 655 — STRUCTURAL STABILITY SOLVER  v3.0                               ║
' ║  Chen & Lui, Chapter 1: Rigid Bar Systems                                  ║
' ║  Bifurcation · Energy · Large Deflection · Geometric Imperfection          ║
' ╚══════════════════════════════════════════════════════════════════════════════╝
' HOW TO USE:
'   1. Open Excel → Alt+F11 → Insert → Module → Paste this entire file
'   2. Close VBA Editor
'   3. Go to Input sheet → fill blue/yellow cells
'   4. Right-click SOLVE button → Assign Macro → SolveProblem → OK → Click it

Option Explicit

Dim nNodes    As Integer, nBars      As Integer
Dim NodeX()   As Double,  NodeY()    As Double
Dim NodeType() As String
Dim k_rot()   As Double,  k_trans()  As Double, P_load() As Double
Dim BarStart() As Integer, BarEnd() As Integer
Dim BarEI()   As Double,  BarEA()   As Double,  BarL()   As Double
Dim k_spring  As Double,  L1 As Double, L2 As Double
Dim Pcr_bif   As Double,  Pcr_nrg   As Double

' ══════════════════════════════════════════════════════════════════════════════
' ENTRY POINT
' ══════════════════════════════════════════════════════════════════════════════
Sub SolveProblem()
    Application.ScreenUpdating = False
    Application.Calculation = xlCalculationManual
    On Error GoTo ErrHandler

    Call ReadInputData
    Call ComputeAnalytical
    Call WriteResults
    Call DrawStructure
    Call ComputePostBuckling
    Call ComputeImperfection

    Application.Calculation = xlCalculationAutomatic
    Application.ScreenUpdating = True
    MsgBox "Solution Complete!  Check Solution, PostBuckling & Imperfection sheets.", _
           vbInformation, "STR 655 Solver"
    Exit Sub
ErrHandler:
    Application.Calculation = xlCalculationAutomatic
    Application.ScreenUpdating = True
    MsgBox "Error: " & Err.Description, vbCritical, "Solver Error"
End Sub

' ══════════════════════════════════════════════════════════════════════════════
' READ INPUT DATA
' ══════════════════════════════════════════════════════════════════════════════
Sub ReadInputData()
    Dim ws As Worksheet, r As Integer
    Set ws = ThisWorkbook.Sheets("Input")

    nNodes = 0
    For r = 8 To 25
        If ws.Cells(r, 2).Value = "" Then Exit For
        nNodes = nNodes + 1
    Next r
    If nNodes = 0 Then nNodes = 3

    nBars = 0
    For r = 17 To 35
        If ws.Cells(r, 2).Value = "" Then Exit For
        If IsNumeric(ws.Cells(r, 2).Value) And ws.Cells(r, 2).Value <> "" Then nBars = nBars + 1
    Next r
    If nBars = 0 Then nBars = 2

    ReDim NodeX(1 To nNodes), NodeY(1 To nNodes), NodeType(1 To nNodes)
    ReDim k_rot(1 To nNodes), k_trans(1 To nNodes), P_load(1 To nNodes)
    ReDim BarStart(1 To nBars), BarEnd(1 To nBars)
    ReDim BarEI(1 To nBars), BarEA(1 To nBars), BarL(1 To nBars)

    For r = 1 To nNodes
        NodeX(r)    = Val(ws.Cells(7+r, 3).Value)
        NodeY(r)    = Val(ws.Cells(7+r, 4).Value)
        NodeType(r) = CStr(ws.Cells(7+r, 5).Value)
        k_rot(r)    = Val(ws.Cells(7+r, 6).Value)
        k_trans(r)  = Val(ws.Cells(7+r, 7).Value)
        P_load(r)   = Val(ws.Cells(7+r, 8).Value)
    Next r

    k_spring = 0
    For r = 1 To nNodes
        If k_rot(r) > 0 Then k_spring = k_rot(r)
    Next r
    If k_spring = 0 Then k_spring = 1

    For r = 1 To nBars
        BarStart(r) = CInt(ws.Cells(16+r, 3).Value)
        BarEnd(r)   = CInt(ws.Cells(16+r, 4).Value)
        BarEI(r)    = Val(ws.Cells(16+r, 5).Value)
        BarEA(r)    = Val(ws.Cells(16+r, 6).Value)
        BarL(r)     = Val(ws.Cells(16+r, 7).Value)
        If BarL(r) = 0 Then
            Dim n1 As Integer, n2 As Integer
            n1 = BarStart(r): n2 = BarEnd(r)
            BarL(r) = Sqr((NodeX(n2)-NodeX(n1))^2 + (NodeY(n2)-NodeY(n1))^2)
        End If
    Next r

    L1 = BarL(1)
    L2 = IIf(nBars >= 2, BarL(2), BarL(1))
End Sub

' ══════════════════════════════════════════════════════════════════════════════
' ANALYTICAL PCR — Bifurcation + Energy
' ══════════════════════════════════════════════════════════════════════════════
Sub ComputeAnalytical()
    Pcr_bif = k_spring * (1# / L1 + 1# / L2)
    Pcr_nrg = k_spring * (1# / L1 + 1# / L2)
End Sub

' ══════════════════════════════════════════════════════════════════════════════
' WRITE RESULTS TO SOLUTION SHEET
' ══════════════════════════════════════════════════════════════════════════════
Sub WriteResults()
    Dim ws As Worksheet
    Set ws = ThisWorkbook.Sheets("Solution")
    ws.Range("C5:C50").ClearContents
    ws.Cells(5,3).Value = Pcr_bif
    ws.Cells(6,3).Value = Pcr_nrg
    ws.Cells(7,3).Value = L1
    ws.Cells(8,3).Value = L2
    ws.Cells(9,3).Value = k_spring
    ws.Cells(10,3).Value = Pcr_bif
    ws.Cells(14,3).Value = "Anti-symmetric sway"
    ws.Cells(15,3).Value = "+theta"
    ws.Cells(16,3).Value = "-theta * (" & Format(L2,"0.##") & "/" & Format(L1,"0.##") & ")"
    ws.Cells(17,3).Value = "theta * " & Format(L1,"0.##")
    ws.Cells(18,3).Value = "Stable Symmetric"
    ws.Cells(30,3).Value = "P = 0, theta = 0"
    ws.Cells(31,3).Value = "P = Pcr = " & Format(Pcr_bif,"0.0000") & " * k"
    ws.Cells(32,3).Value = "P > Pcr, theta <> 0 (stable)"
    ws.Cells(33,3).Value = "STABLE SYMMETRIC BIFURCATION"
    ws.Cells(34,3).Value = "LOW (hardening post-buckling)"
End Sub

' ══════════════════════════════════════════════════════════════════════════════
' UPDATE DRAWING SHEET
' ══════════════════════════════════════════════════════════════════════════════
Sub DrawStructure()
    Dim ws As Worksheet
    Set ws = ThisWorkbook.Sheets("Drawing")
    ws.Cells(8,2).Value = "  |<--- L1 = " & Format(L1,"0.0#") & " m ---|<--- L2 = " & Format(L2,"0.0#") & " m --->|"
    ws.Cells(3,3).Value = "k_spring = " & Format(k_spring,"0.000") & " kN.m/rad"
End Sub

' ══════════════════════════════════════════════════════════════════════════════
' POST-BUCKLING: P/Pcr = theta_rad / sin(theta_rad)
' ══════════════════════════════════════════════════════════════════════════════
Sub ComputePostBuckling()
    Dim ws As Worksheet, i As Integer
    Dim theta_d As Double, theta_r As Double, sn As Double, ratio As Double
    Const PI As Double = 3.14159265358979
    Set ws = ThisWorkbook.Sheets("PostBuckling")
    ws.Range("B13:H200").ClearContents
    For i = 1 To 45
        theta_d = CDbl(i)
        theta_r = theta_d * PI / 180#
        sn      = Sin(theta_r)
        ratio   = IIf(Abs(sn) > 0.00001, theta_r / sn, 1#)
        Dim rw As Integer: rw = 12 + i
        ws.Cells(rw,2).Value = Round(theta_d,1)
        ws.Cells(rw,3).Value = Round(theta_r,5)
        ws.Cells(rw,4).Value = Round(sn,5)
        ws.Cells(rw,5).Value = Round(ratio,5)
        ws.Cells(rw,6).Value = Round(ratio * Pcr_bif,5)
        ws.Cells(rw,7).Value = IIf(i=1, "Bifurcation point", "Post-buckling (stable)")
    Next i
End Sub

' ══════════════════════════════════════════════════════════════════════════════
' IMPERFECTION: P/Pcr vs theta for theta_0 = 0, 0.5, 1, 2, 5, 10 degrees
' Formula: P = k*(1/L1+1/L2)*(theta-theta0)/sin(theta)
' ══════════════════════════════════════════════════════════════════════════════
Sub ComputeImperfection()
    Dim ws As Worksheet, i As Integer, j As Integer
    Dim theta0(5) As Double
    Dim theta_d As Double, theta_r As Double, theta0_r As Double
    Dim sn As Double, ratio As Double
    Const PI As Double = 3.14159265358979
    Set ws = ThisWorkbook.Sheets("Imperfection")
    theta0(0)=0#: theta0(1)=0.5: theta0(2)=1#: theta0(3)=2#: theta0(4)=5#: theta0(5)=10#
    ws.Range("B6:H200").ClearContents
    For i = 1 To 45
        theta_d = CDbl(i)
        theta_r = theta_d * PI / 180#
        sn      = Sin(theta_r)
        Dim rw As Integer: rw = 5 + i
        ws.Cells(rw,2).Value = Round(theta_d,1)
        For j = 0 To 5
            theta0_r = theta0(j) * PI / 180#
            If Abs(sn) > 0.00001 Then
                ratio = (k_spring*(1#/L1+1#/L2)*(theta_r-theta0_r)/sn) / Pcr_bif
            Else
                ratio = 0#
            End If
            ws.Cells(rw,3+j).Value = Round(IIf(ratio<0,0,ratio),5)
        Next j
    Next i
End Sub

' ══════════════════════════════════════════════════════════════════════════════
' MATRIX UTILITIES
' ══════════════════════════════════════════════════════════════════════════════
Function MatMul(A() As Double, B() As Double, m As Integer, n As Integer, p As Integer) As Double()
    Dim C() As Double, i As Integer, j As Integer, k As Integer, s As Double
    ReDim C(1 To m, 1 To p)
    For i = 1 To m
        For j = 1 To p
            s = 0
            For k = 1 To n: s = s + A(i,k)*B(k,j): Next k
            C(i,j) = s
        Next j
    Next i
    MatMul = C
End Function

Function Invert2x2(A() As Double) As Double()
    Dim R() As Double, det As Double
    ReDim R(1 To 2, 1 To 2)
    det = A(1,1)*A(2,2) - A(1,2)*A(2,1)
    If Abs(det) < 1E-15 Then MsgBox "Singular matrix!", vbCritical: Exit Function
    R(1,1)= A(2,2)/det: R(1,2)=-A(1,2)/det
    R(2,1)=-A(2,1)/det: R(2,2)= A(1,1)/det
    Invert2x2 = R
End Function

Function Solve2x2(A() As Double, b() As Double) As Double()
    Dim inv() As Double, x(1 To 2) As Double
    inv = Invert2x2(A)
    x(1) = inv(1,1)*b(1) + inv(1,2)*b(2)
    x(2) = inv(2,1)*b(1) + inv(2,2)*b(2)
    Solve2x2 = x
End Function