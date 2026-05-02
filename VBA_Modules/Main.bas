Attribute VB_Name = "Main"
Option Explicit

' ============================================================
'  STRUCTURAL STABILITY SOLVER - Chapter 1 (STR 655)
'  Author  : Built for Dr-Yehia / Stability-Book
'  Covers  : All Chapter-1 problem types
'             Type A : Rigid-bar + springs  (1-DOF & 2-DOF)
'             Type B : Elastic column ODE   (numerical root)
'             Type C : Truss / spring at 45-deg
'  Methods : Bifurcation + Energy approach
'            Small + Large deflection
'            Post-buckling path
'            Geometric imperfection
'            Stability check (d2Pi/dtheta2)
' ============================================================

Sub SOLVE()
    Dim ws As Worksheet
    Set ws = ThisWorkbook.Sheets("INPUT")
    
    Dim sType As String
    sType = UCase(Trim(ws.Range("C4").Value))
    
    Call ClearResults
    
    Select Case sType
        Case "A"
            Call Solve_TypeA(ws)
        Case "B"
            Call Solve_TypeB(ws)
        Case "C"
            Call Solve_TypeC(ws)
        Case Else
            MsgBox "Unknown system type. Enter A, B, or C in cell C4.", vbCritical
    End Select
End Sub

Sub ClearResults()
    Dim wsR As Worksheet, wsP As Worksheet
    Set wsR = ThisWorkbook.Sheets("RESULTS")
    Set wsP = ThisWorkbook.Sheets("PLOT_DATA")
    
    wsR.Cells.ClearContents
    wsP.Cells.ClearContents
    
    Dim cht As ChartObject
    For Each cht In wsR.ChartObjects
        cht.Delete
    Next cht
End Sub
