param(
    [string]$SourceDeck = "",
    [string]$OutputDeck = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptRoot

if ([string]::IsNullOrWhiteSpace($SourceDeck)) {
    $SourceDeck = Join-Path $repoRoot "End_Documentation\Panel Review 2.pptx"
}
if ([string]::IsNullOrWhiteSpace($OutputDeck)) {
    $OutputDeck = Join-Path $scriptRoot "Review_2_Updated_Project_Presentation.pptx"
}

$SourceDeck = [System.IO.Path]::GetFullPath($SourceDeck)
$OutputDeck = [System.IO.Path]::GetFullPath($OutputDeck)
$OutputPdf = [System.IO.Path]::ChangeExtension($OutputDeck, ".pdf")
$PreviewDir = Join-Path $scriptRoot "presentation_preview"

if (-not (Test-Path -LiteralPath $SourceDeck)) {
    throw "Source presentation not found: $SourceDeck"
}

[System.IO.Directory]::CreateDirectory((Split-Path -Parent $OutputDeck)) | Out-Null
[System.IO.Directory]::CreateDirectory($PreviewDir) | Out-Null
[System.IO.File]::Copy($SourceDeck, $OutputDeck, $true)

function Get-Rgb([int]$R, [int]$G, [int]$B) {
    return $R + (256 * $G) + (65536 * $B)
}

$Blue = Get-Rgb 64 111 183
$DarkBlue = Get-Rgb 31 78 121
$LightBlue = Get-Rgb 232 240 252
$VeryLightBlue = Get-Rgb 246 249 253
$Green = Get-Rgb 57 128 73
$LightGreen = Get-Rgb 233 246 236
$Orange = Get-Rgb 202 110 25
$LightOrange = Get-Rgb 253 241 227
$Gray = Get-Rgb 85 85 85
$LightGray = Get-Rgb 243 245 247
$White = Get-Rgb 255 255 255
$Red = Get-Rgb 192 45 45

function Get-TitleShape($Slide) {
    $best = $null
    for ($i = 1; $i -le $Slide.Shapes.Count; $i++) {
        $shape = $Slide.Shapes.Item($i)
        try {
            if ($shape.HasTextFrame -and $shape.Top -lt 100 -and $shape.Width -gt 200) {
                if ($null -eq $best -or $shape.Width -gt $best.Width) {
                    $best = $shape
                }
            }
        } catch {}
    }
    return $best
}

function Get-SlideTitle($Slide) {
    $shape = Get-TitleShape $Slide
    if ($null -eq $shape) { return "" }
    try { return $shape.TextFrame.TextRange.Text.Trim() } catch { return "" }
}

function Set-SlideTitle($Slide, [string]$Text) {
    $shape = Get-TitleShape $Slide
    if ($null -eq $shape) { throw "No title shape found on slide $($Slide.SlideIndex)" }
    $shape.TextFrame.TextRange.Text = $Text
    $shape.TextFrame.TextRange.Font.Name = "Arial"
    $shape.TextFrame.TextRange.Font.Bold = -1
}

function Clear-SlideBody($Slide) {
    for ($i = $Slide.Shapes.Count; $i -ge 1; $i--) {
        $shape = $Slide.Shapes.Item($i)
        if ($shape.Top -ge 90 -and $shape.Top -lt 495) {
            $shape.Delete()
        }
    }
}

function Add-TextBox(
    $Slide,
    [float]$Left,
    [float]$Top,
    [float]$Width,
    [float]$Height,
    [string]$Text,
    [float]$FontSize = 16,
    [int]$Color = 0,
    [bool]$Bold = $false,
    [int]$Alignment = 1
) {
    $shape = $Slide.Shapes.AddTextbox(1, $Left, $Top, $Width, $Height)
    $shape.TextFrame.MarginLeft = 5
    $shape.TextFrame.MarginRight = 5
    $shape.TextFrame.MarginTop = 3
    $shape.TextFrame.MarginBottom = 3
    $shape.TextFrame.TextRange.Text = $Text
    $shape.TextFrame.TextRange.Font.Name = "Arial"
    $shape.TextFrame.TextRange.Font.Size = $FontSize
    $shape.TextFrame.TextRange.Font.Bold = $(if ($Bold) { -1 } else { 0 })
    $shape.TextFrame.TextRange.Font.Color.RGB = $Color
    $shape.TextFrame.TextRange.ParagraphFormat.Alignment = $Alignment
    return $shape
}

function Add-Card(
    $Slide,
    [float]$Left,
    [float]$Top,
    [float]$Width,
    [float]$Height,
    [string]$Heading,
    [string]$Body,
    [int]$Accent,
    [int]$Fill
) {
    $card = $Slide.Shapes.AddShape(5, $Left, $Top, $Width, $Height)
    $card.Fill.ForeColor.RGB = $Fill
    $card.Fill.Transparency = 0
    $card.Line.ForeColor.RGB = $Accent
    $card.Line.Weight = 1.5
    $null = Add-TextBox $Slide ($Left + 12) ($Top + 9) ($Width - 24) 30 $Heading 18 $Accent $true 1
    $null = Add-TextBox $Slide ($Left + 12) ($Top + 44) ($Width - 24) ($Height - 53) $Body 14 $Gray $false 1
    return $card
}

function New-ThemedSlide($Presentation, [string]$Title, [int]$InsertAt) {
    $duplicateRange = $Presentation.Slides.Item(4).Duplicate()
    $slide = $duplicateRange.Item(1)
    $slide.MoveTo($InsertAt)
    Set-SlideTitle $slide $Title
    Clear-SlideBody $slide
    return $slide
}

function Set-CellText($Cell, [string]$Text, [float]$Size, [int]$Color, [bool]$Bold, [int]$Alignment) {
    $shape = $Cell.Shape
    $shape.TextFrame.MarginLeft = 5
    $shape.TextFrame.MarginRight = 5
    $shape.TextFrame.MarginTop = 4
    $shape.TextFrame.MarginBottom = 3
    $shape.TextFrame.TextRange.Text = $Text
    $shape.TextFrame.TextRange.Font.Name = "Arial"
    $shape.TextFrame.TextRange.Font.Size = $Size
    $shape.TextFrame.TextRange.Font.Color.RGB = $Color
    $shape.TextFrame.TextRange.Font.Bold = $(if ($Bold) { -1 } else { 0 })
    $shape.TextFrame.TextRange.ParagraphFormat.Alignment = $Alignment
    $shape.TextFrame.VerticalAnchor = 3
}

function Add-LiteratureTable($Slide) {
    Clear-SlideBody $Slide
    Set-SlideTitle $Slide "3. Literature Survey and Research Gap"

    $tableShape = $Slide.Shapes.AddTable(4, 4, 22, 103, 916, 323)
    $table = $tableShape.Table
    $widths = @(145, 225, 245, 301)
    for ($c = 1; $c -le 4; $c++) { $table.Columns.Item($c).Width = $widths[$c - 1] }
    $table.Rows.Item(1).Height = 43
    for ($r = 2; $r -le 4; $r++) { $table.Rows.Item($r).Height = 93 }

    $headers = @("Source", "What the paper does", "Limitation for our problem", "What our project improves")
    for ($c = 1; $c -le 4; $c++) {
        $cell = $table.Cell(1, $c)
        $cell.Shape.Fill.ForeColor.RGB = $DarkBlue
        Set-CellText $cell $headers[$c - 1] 13 $White $true 2
    }

    $rows = @(
        @(
            "Freq-Pret",
            "Retrains the classifier using frequency-perturbed images.",
            "Changes model weights and requires a clean subset plus retraining.",
            "Keeps the classifier frozen and produces a corrected image from one input."
        ),
        @(
            "BAM",
            "Learns an input purifier from simulated backdoor behavior.",
            "General image-space purification; spectral support is less explicit.",
            "Uses amplitude, phase, residual and frequency position, with causal support tests."
        ),
        @(
            "Semantic-aware",
            "Uses a trusted model to detect suspicious behavior, then performs unlearning.",
            "Needs a trusted comparison model and modifies the suspicious classifier.",
            "Uses no second model at inference and leaves the classifier unchanged."
        )
    )

    for ($r = 2; $r -le 4; $r++) {
        for ($c = 1; $c -le 4; $c++) {
            $cell = $table.Cell($r, $c)
            $cell.Shape.Fill.ForeColor.RGB = $(if (($r % 2) -eq 0) { $VeryLightBlue } else { $LightGray })
            Set-CellText $cell $rows[$r - 2][$c - 1] 11.5 $Gray ($c -eq 1) 1
        }
    }

    $note = $Slide.Shapes.AddShape(5, 56, 439, 848, 47)
    $note.Fill.ForeColor.RGB = $LightBlue
    $note.Line.ForeColor.RGB = $Blue
    $note.Line.Weight = 1.25
    $note.TextFrame.MarginLeft = 12
    $note.TextFrame.MarginRight = 12
    $note.TextFrame.MarginTop = 7
    $note.TextFrame.TextRange.Text = "Research gap: protect a frozen third-party classifier by correcting one image in frequency space, without requiring its matching clean image at inference."
    $note.TextFrame.TextRange.Font.Name = "Arial"
    $note.TextFrame.TextRange.Font.Size = 14
    $note.TextFrame.TextRange.Font.Bold = -1
    $note.TextFrame.TextRange.Font.Color.RGB = $DarkBlue
    $note.TextFrame.TextRange.ParagraphFormat.Alignment = 2
}

function Add-MetricsSlide($Presentation, [int]$InsertAt) {
    $slide = New-ThemedSlide $Presentation "7. Standard Algorithms and Evaluation Metrics" $InsertAt

    $algorithms = "FFT / IFFT`r`n- Decompose image into amplitude and phase.`r`n`r`nResNet-18`r`n- ImageNet-pretrained 47-class texture classifier.`r`n`r`nCorrection generator`r`n- Predicts a gate and signed log-amplitude change.`r`n`r`nOptimization`r`n- Classification, image, spectral, identity, sparsity and smoothness losses."
    $metrics = "Clean accuracy`r`n- Normal classification utility.`r`n`r`nAttack Success Rate (ASR)`r`n- Fraction forced into the attacker's target class.`r`n`r`nTrue-label recovery`r`n- Correct classification after correction.`r`n`r`nTarget-rate lift and conditional ASR`r`n- Verify that the trigger caused new failures.`r`n`r`nPixel L1 and TNR`r`n- Measure image change and trigger/noise stress."

    $null = Add-Card $slide 46 108 412 359 "Core algorithms" $algorithms $Blue $VeryLightBlue
    $null = Add-Card $slide 502 108 412 359 "Evaluation metrics" $metrics $Green $LightGreen
    return $slide
}

function Add-ToolsSlide($Presentation, [int]$InsertAt) {
    $slide = New-ThemedSlide $Presentation "8. Tools and Implementation Expertise" $InsertAt

    $training = "Python and PyTorch`r`n- Dataset pipelines, poisoning, model training and generator optimization.`r`n`r`nKaggle GPU`r`n- NVIDIA T4 sessions for attack and generator experiments.`r`n`r`nPretrained learning`r`n- ImageNet ResNet-18 adapted to 47 DTD classes."
    $analysis = "Signal processing`r`n- FFT/IFFT, log amplitude, phase preservation and block-DCT trigger injection.`r`n`r`nReproducibility`r`n- Saved checkpoints, JSON summaries, manifests and fixed evaluation splits.`r`n`r`nVisualization`r`n- Correction maps, spectral panels, ablation charts and a CPU demonstration notebook."

    $null = Add-Card $slide 46 108 412 359 "Training and models" $training $Blue $VeryLightBlue
    $null = Add-Card $slide 502 108 412 359 "Analysis and evidence" $analysis $Orange $LightOrange
    return $slide
}

function Add-ProjectPlanSlide($Presentation, [int]$InsertAt) {
    $slide = New-ThemedSlide $Presentation "12. Project Plan for Semester 3" $InsertAt

    $tableShape = $slide.Shapes.AddTable(5, 4, 50, 113, 860, 320)
    $table = $tableShape.Table
    $widths = @(128, 305, 120, 307)
    for ($c = 1; $c -le 4; $c++) { $table.Columns.Item($c).Width = $widths[$c - 1] }
    $table.Rows.Item(1).Height = 42
    for ($r = 2; $r -le 5; $r++) { $table.Rows.Item($r).Height = 69.5 }

    $headers = @("Period", "Activity", "Status", "Expected output")
    for ($c = 1; $c -le 4; $c++) {
        $cell = $table.Cell(1, $c)
        $cell.Shape.Fill.ForeColor.RGB = $DarkBlue
        Set-CellText $cell $headers[$c - 1] 13 $White $true 2
    }

    $rows = @(
        @("Completed", "Attack calibration, generator training, held-out tests, baselines and TNR study", "Done", "Frozen results, checkpoints and evidence figures"),
        @("Oct-Nov 2026", "Organize literature comparison and write IEEE manuscript sections", "In progress", "Complete first paper draft"),
        @("Nov 2026", "Run one locked repeat and multi-seed summary where compute permits", "Planned", "Stronger reproducibility statement"),
        @("Nov-Dec 2026", "Guide review, revisions, formatting and final submission preparation", "Planned", "Submission-ready paper and dissertation material")
    )

    for ($r = 2; $r -le 5; $r++) {
        for ($c = 1; $c -le 4; $c++) {
            $cell = $table.Cell($r, $c)
            $cell.Shape.Fill.ForeColor.RGB = $(if (($r % 2) -eq 0) { $VeryLightBlue } else { $LightGray })
            $color = $(if ($c -eq 3 -and $rows[$r - 2][$c - 1] -eq "Done") { $Green } else { $Gray })
            Set-CellText $cell $rows[$r - 2][$c - 1] 12 $color ($c -eq 1 -or $c -eq 3) 1
        }
    }

    $null = Add-TextBox $slide 62 445 836 37 "Primary Phase 3 goal: convert the completed experimental evidence into a defensible, reproducible IEEE-format paper." 14 $DarkBlue $true 2
    return $slide
}

function Find-SlideIndexByTitle($Presentation, [string]$Pattern) {
    for ($i = 1; $i -le $Presentation.Slides.Count; $i++) {
        if ((Get-SlideTitle $Presentation.Slides.Item($i)) -match $Pattern) { return $i }
    }
    return 0
}

function Update-SectionTitles($Presentation) {
    for ($i = 1; $i -le $Presentation.Slides.Count; $i++) {
        $slide = $Presentation.Slides.Item($i)
        $title = Get-SlideTitle $slide
        if ([string]::IsNullOrWhiteSpace($title)) { continue }

        switch -Regex ($title) {
            '^3\. Where the First Panel' { Set-SlideTitle $slide "3.1 Where the First Panel Left the Project"; break }
            '^2\. Questions Raised' { Set-SlideTitle $slide "3.2 Questions Raised After the First Panel"; break }
            '^6\. Implementation' { Set-SlideTitle $slide "6. Implementation Progress and Discussion"; break }
            '^7\. Results.*Trigger' { Set-SlideTitle $slide "9. Results - Trigger Versus Noise"; break }
            '^7\. Results$' {
                Set-SlideTitle $slide "9. Results"
                (Get-TitleShape $slide).TextFrame.TextRange.Font.Size = 32
                break
            }
            '^7\. Results' { Set-SlideTitle $slide "9. Results"; break }
            '^8\. Demo' { Set-SlideTitle $slide "10. Demonstration"; break }
            '^9\. SDG Mapping' { Set-SlideTitle $slide "11. SDG Mapping"; break }
            '^10\. Domain Expertise' { Set-SlideTitle $slide "13. Domain Expertise Course"; break }
            '^11\. Technical Enrichment' { Set-SlideTitle $slide "14. Technical Enrichment Activities"; break }
            '^12\. Conclusion' { Set-SlideTitle $slide "15. Conclusion"; break }
            '^13\. References' { Set-SlideTitle $slide "16. References"; break }
        }
    }
}

function Update-ReferencesSlide($Presentation) {
    $index = Find-SlideIndexByTitle $Presentation '^16\. References$'
    if ($index -eq 0) { return }
    $slide = $Presentation.Slides.Item($index)
    Clear-SlideBody $slide

    $refs = @(
        '[1] M. Cimpoi et al., "Describing Textures in the Wild," CVPR, 2014.',
        '[2] K. He et al., "Deep Residual Learning for Image Recognition," CVPR, 2016.',
        '[3] T. Wang et al., "An Invisible Black-box Backdoor Attack through Frequency Domain," ECCV, 2022.',
        '[4] P. Zhang, H. Yue, and C. Yuen, "Backdoor Defense Strategy for Image Classification Tasks Based on Frequency Domain Perturbation," IEEE IoT Journal, 2026.',
        '[5] R. Lu et al., "BAM: A Backdoor-Agnostic Input Purification Method," Pattern Recognition, 2026.',
        '[6] K. Fu and J. Dai, "Semantic-Aware Framework for Backdoor Detection in AI Models," IJSWIS, 2025.',
        '[7] T. Gu, B. Dolan-Gavitt, and S. Garg, "BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain," 2017.'
    )
    $text = $refs -join "`r`n`r`n"
    $null = Add-TextBox $slide 66 108 832 365 $text 13.5 $Gray $false 1
}

$powerPoint = $null
$presentation = $null
try {
    $powerPoint = New-Object -ComObject PowerPoint.Application
    $powerPoint.Visible = -1
    $powerPoint.DisplayAlerts = 1
    $presentation = $powerPoint.Presentations.Open($OutputDeck, $false, $false, $false)

    Add-LiteratureTable $presentation.Slides.Item(5)

    $null = Add-MetricsSlide $presentation 17
    $null = Add-ToolsSlide $presentation 18

    $domainIndex = Find-SlideIndexByTitle $presentation '^10\. Domain Expertise'
    if ($domainIndex -eq 0) { throw "Could not locate Domain Expertise slide." }
    $null = Add-ProjectPlanSlide $presentation $domainIndex

    Update-SectionTitles $presentation
    Update-ReferencesSlide $presentation

    $presentation.Save()

    $previewPatterns = @(
        '^3\. Literature Survey',
        '^7\. Standard Algorithms',
        '^8\. Tools',
        '^9\. Results$',
        '^12\. Project Plan',
        '^16\. References'
    )
    foreach ($pattern in $previewPatterns) {
        $index = Find-SlideIndexByTitle $presentation $pattern
        if ($index -gt 0) {
            $safeName = ($pattern -replace '[^A-Za-z0-9]+', '_').Trim('_')
            $path = Join-Path $PreviewDir ("slide_{0:D2}_{1}.png" -f $index, $safeName)
            $presentation.Slides.Item($index).Export($path, "PNG", 1600, 900)
        }
    }

    try {
        $presentation.SaveCopyAs($OutputPdf, 32)
    }
    catch {
        Write-Warning "PPTX was created, but PDF export was unavailable: $($_.Exception.Message)"
    }

    Write-Output "Created: $OutputDeck"
    if (Test-Path -LiteralPath $OutputPdf) { Write-Output "Created: $OutputPdf" }
    Write-Output "Slides: $($presentation.Slides.Count)"
    Write-Output "Previews: $PreviewDir"
}
finally {
    if ($null -ne $presentation) {
        try { $presentation.Close() } catch {}
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($presentation)
    }
    if ($null -ne $powerPoint) {
        try { $powerPoint.Quit() } catch {}
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($powerPoint)
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
