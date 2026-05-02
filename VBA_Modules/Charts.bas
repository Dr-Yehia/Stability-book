Attribute VB_Name = "Charts"
Option Explicit

' ============================================================
'  Auto-generate equilibrium path chart from PLOT_DATA
' ============================================================

Sub CreateEquilibriumChart(Pcr As Double, sTitle As String)
    Dim wsR As Worksheet, wsP As Worksheet
    Set wsR = ThisWorkbook.Sheets("RESULTS")
    Set wsP = ThisWorkbook.Sheets("PLOT_DATA")
    
    Dim lastRow As Long
    lastRow = wsP.Cells(wsP.Rows.Count, 1).End(xlUp).Row
    If lastRow < 3 Then Exit Sub
    
    Dim cht As ChartObject
    Set cht = wsR.ChartObjects.Add(Left:=320, Top:=20, Width:=520, Height:=380)
    
    With cht.Chart
        .ChartType = xlXYScatterSmoothNoMarkers
        .HasTitle = True
        .ChartTitle.Text = "Equilibrium Paths - " & sTitle
        .ChartTitle.Font.Size = 11
        .ChartTitle.Font.Bold = True
        
        ' Series 1: Fundamental
        .SeriesCollection.NewSeries
        With .SeriesCollection(1)
            .Name = "Fundamental Path"
            .XValues = wsP.Range("A2:A31")
            .Values  = wsP.Range("B2:B31")
            .Format.Line.ForeColor.RGB = RGB(0, 112, 192)
            .Format.Line.Weight = 2
        End With
        
        ' Series 2: Post-buckling perfect
        .SeriesCollection.NewSeries
        With .SeriesCollection(2)
            .Name = "Post-Buckling (Perfect)"
            .XValues = wsP.Range("A33:A72")
            .Values  = wsP.Range("C33:C72")
            .Format.Line.ForeColor.RGB = RGB(192, 0, 0)
            .Format.Line.Weight = 2
        End With
        
        ' Series 3: Imperfect path
        .SeriesCollection.NewSeries
        With .SeriesCollection(3)
            .Name = "Imperfect Path"
            .XValues = wsP.Range("A74:A120")
            .Values  = wsP.Range("D74:D120")
            .Format.Line.ForeColor.RGB = RGB(0, 176, 80)
            .Format.Line.DashStyle = msoLineDash
            .Format.Line.Weight = 1.5
        End With
        
        ' Pcr horizontal reference line
        .SeriesCollection.NewSeries
        With .SeriesCollection(4)
            .Name = "Pcr = " & Round(Pcr, 3) & " kN"
            .XValues = Array(-0.6, 0, 0.6)
            .Values  = Array(Pcr, Pcr, Pcr)
            .Format.Line.ForeColor.RGB = RGB(100, 100, 100)
            .Format.Line.DashStyle = msoLineDot
            .Format.Line.Weight = 1
        End With
        
        With .Axes(xlCategory)
            .HasTitle = True
            .AxisTitle.Text = "theta1 (rad)"
            .CrossesAt = 0
        End With
        With .Axes(xlValue)
            .HasTitle = True
            .AxisTitle.Text = "P (kN)"
        End With
        
        .Legend.Position = xlLegendPositionBottom
        .Legend.Font.Size = 9
        .PlotArea.Interior.Color = RGB(255, 255, 255)
    End With
End Sub
