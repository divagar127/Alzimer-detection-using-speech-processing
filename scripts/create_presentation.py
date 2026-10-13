import os
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def build_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Color Palette
    AMRITA_BLUE = RGBColor(0, 51, 102)       # #003366 Navy Blue
    DARK_BLUE   = RGBColor(15, 32, 67)       # #0F2043 Dark Navy
    ACCENT_GREEN = RGBColor(16, 124, 65)     # #107C41 Forest Green
    ACCENT_TEAL  = RGBColor(0, 128, 128)     # #008080 Teal
    DARK_GRAY   = RGBColor(40, 40, 40)       # #282828 Text
    MUTED_GRAY  = RGBColor(100, 100, 100)    # #646464 Muted
    WHITE       = RGBColor(255, 255, 255)    # White
    LIGHT_BG    = RGBColor(246, 249, 252)    # Soft slate tint
    CARD_BORDER = RGBColor(215, 225, 235)    # Subtle border
    ACCENT_RED  = RGBColor(180, 40, 40)      # Red accent

    def add_bg(slide, slide_num):
        bg_path = f'template_slides/slide_{slide_num}.png'
        if os.path.exists(bg_path):
            slide.shapes.add_picture(bg_path, Inches(0), Inches(0), width=prs.slide_width, height=prs.slide_height)

    def create_card(slide, left, top, width, height, bg_color=WHITE, border_color=CARD_BORDER):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        if border_color:
            card.line.color.rgb = border_color
            card.line.width = Pt(1.2)
        else:
            card.line.fill.background()
        return card

    # =========================================================================
    # SLIDE 1: TITLE SLIDE
    # =========================================================================
    slide1 = prs.slides.add_slide(blank_layout)
    add_bg(slide1, 1)

    # Clean overlay card to cover placeholders
    create_card(slide1, Inches(0.8), Inches(1.3), Inches(11.733), Inches(5.3), bg_color=WHITE, border_color=CARD_BORDER)

    # Title box
    tb = slide1.shapes.add_textbox(Inches(1.2), Inches(1.5), Inches(10.9), Inches(1.8))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Multimodal Speech Biomarker Fusion using Conversational Dynamics and Deep Acoustic Representations for Automated Alzheimer's Dementia Recognition"
    p.font.name = "Arial"
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = AMRITA_BLUE
    p.alignment = PP_ALIGN.CENTER

    p2 = tf.add_paragraph()
    p2.text = "Final Project Technical Presentation | Course Code: 22AIE450 Speech Processing"
    p2.font.name = "Arial"
    p2.font.size = Pt(13)
    p2.font.color.rgb = ACCENT_GREEN
    p2.font.bold = True
    p2.alignment = PP_ALIGN.CENTER

    # 3 Member Cards
    members = [
        ("Divagar S", "CB.SC.U4AIE23223", "Gated Fusion, Multi-Task Loss & XAI"),
        ("Abhishek Pandey", "CB.SC.U4AIE23205", "Audio Preprocessing, VAD & Timing"),
        ("Prem", "CB.SC.U4AIE23241", "123 Acoustic Biomarkers & Error Analysis")
    ]
    card_w = Inches(3.4)
    for idx, (name, roll, role) in enumerate(members):
        c_left = Inches(1.2 + idx * 3.75)
        create_card(slide1, c_left, Inches(3.6), card_w, Inches(1.7), bg_color=LIGHT_BG, border_color=CARD_BORDER)
        mtb = slide1.shapes.add_textbox(c_left + Inches(0.15), Inches(3.7), card_w - Inches(0.3), Inches(1.5))
        mtf = mtb.text_frame
        mtf.word_wrap = True
        
        mp1 = mtf.paragraphs[0]
        mp1.text = name
        mp1.font.name = "Arial"
        mp1.font.size = Pt(15)
        mp1.font.bold = True
        mp1.font.color.rgb = DARK_BLUE
        
        mp2 = mtf.add_paragraph()
        mp2.text = f"Roll No: {roll}"
        mp2.font.name = "Arial"
        mp2.font.size = Pt(12)
        mp2.font.color.rgb = MUTED_GRAY
        
        mp3 = mtf.add_paragraph()
        mp3.text = f"Focus: {role}"
        mp3.font.name = "Arial"
        mp3.font.size = Pt(11)
        mp3.font.color.rgb = ACCENT_GREEN

    # Department & Date Footer
    ftb = slide1.shapes.add_textbox(Inches(1.2), Inches(5.6), Inches(10.9), Inches(0.8))
    ftf = ftb.text_frame
    fp = ftf.paragraphs[0]
    fp.text = "Department of Artificial Intelligence and Data Science | Amrita Vishwa Vidyapeetham, Coimbatore\nBatch: 2023–2027 | Date of Presentation: October 2026"
    fp.font.name = "Arial"
    fp.font.size = Pt(11)
    fp.font.color.rgb = MUTED_GRAY
    fp.alignment = PP_ALIGN.CENTER

    # =========================================================================
    # SLIDE 2: OUTLINE
    # =========================================================================
    slide2 = prs.slides.add_slide(blank_layout)
    add_bg(slide2, 2)

    outline_items = [
        ("01", "Introduction & Motivation", "Global dementia burden, diagnostic hurdles, and neurobiology of speech breakdown."),
        ("02", "Problem Definition", "Transcription bottlenecks, early fusion failure, and data leakage in existing literature."),
        ("03", "Literature Review & Gaps", "Systematic review of 5 key papers (IS2021, ICASSP, IEEE) and 4 critical technical gaps."),
        ("04", "Project Objectives", "Audio-only screening, 15 timing biomarkers, Dynamic Gated Fusion, and XAI."),
        ("05", "Methodology & Architecture", "Signal preprocessing, 3 feature streams (123+512+15), Multi-Task Learning framework."),
        ("06", "Dataset Description", "Interspeech 2021 ADReSSo benchmark (166 recordings: 87 AD, 79 CN), Cookie Theft task."),
        ("07", "Results & Discussion", "80.16% Acc, 0.8572 AUC, baseline comparisons, ablation matrix, and SHAP/LIME XAI."),
        ("08", "Conclusion & Contributions", "Summary of clinical breakthroughs, team division of work, and deployment roadmap.")
    ]

    for idx, (num, title, desc) in enumerate(outline_items):
        col = idx % 2
        row = idx // 2
        o_left = Inches(0.9 + col * 5.9)
        o_top = Inches(1.5 + row * 1.25)
        create_card(slide2, o_left, o_top, Inches(5.65), Inches(1.15), bg_color=WHITE, border_color=CARD_BORDER)

        otb = slide2.shapes.add_textbox(o_left + Inches(0.15), o_top + Inches(0.1), Inches(5.35), Inches(0.95))
        otf = otb.text_frame
        otf.word_wrap = True
        
        p = otf.paragraphs[0]
        p.text = f"{num}.  {title}"
        p.font.name = "Arial"
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = AMRITA_BLUE
        
        p2 = otf.add_paragraph()
        p2.text = desc
        p2.font.name = "Arial"
        p2.font.size = Pt(11)
        p2.font.color.rgb = DARK_GRAY

    # =========================================================================
    # SLIDE 3: INTRODUCTION AND MOTIVATION
    # =========================================================================
    slide3 = prs.slides.add_slide(blank_layout)
    add_bg(slide3, 3)

    # Card 1: Clinical Burden & Limitations
    create_card(slide3, Inches(0.9), Inches(1.45), Inches(5.65), Inches(5.1), bg_color=WHITE)
    tb1 = slide3.shapes.add_textbox(Inches(1.1), Inches(1.6), Inches(5.25), Inches(4.7))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    p = tf1.paragraphs[0]
    p.text = "Clinical Motivation & Diagnostic Hurdles"
    p.font.name = "Arial"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = AMRITA_BLUE

    points1 = [
        ("Epidemic Scale: ", "Alzheimer's Disease (AD) accounts for 60%–70% of dementia worldwide. WHO projects >139 million cases by 2050."),
        ("Invasive Clinical Pathways: ", "Gold-standard diagnosis relies on PET imaging ($3,000+), MRI, and lumbar puncture CSF assays for amyloid/tau."),
        ("Logistical Bottlenecks: ", "Cost-prohibitive, geographically concentrated in specialized memory clinics, and often administered late after irreversible neuronal loss."),
        ("Subjective Bias: ", "Cognitive assessments (MMSE, MoCA) suffer from examiner subjectivity and test-retest learning effects."),
        ("Urgent Imperative: ", "Need for passive, scalable, non-invasive digital screening modalities deployable in primary care and telemedicine.")
    ]
    for b_title, b_text in points1:
        p = tf1.add_paragraph()
        run1 = p.add_run()
        run1.text = "• " + b_title
        run1.font.bold = True
        run1.font.size = Pt(11.5)
        run1.font.color.rgb = DARK_BLUE
        run2 = p.add_run()
        run2.text = b_text
        run2.font.size = Pt(11)
        run2.font.color.rgb = DARK_GRAY

    # Card 2: Neurobiology of Speech Breakdown
    create_card(slide3, Inches(6.8), Inches(1.45), Inches(5.65), Inches(5.1), bg_color=WHITE)
    tb2 = slide3.shapes.add_textbox(Inches(7.0), Inches(1.6), Inches(5.25), Inches(4.7))
    tf2 = tb2.text_frame
    tf2.word_wrap = True

    p = tf2.paragraphs[0]
    p.text = "Neurobiology of Speech Breakdown in AD"
    p.font.name = "Arial"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GREEN

    points2 = [
        ("High Cognitive Bandwidth: ", "Spontaneous speech requires synchronized semantic retrieval, working memory, and rapid motor articulation."),
        ("Wernicke's & Temporoparietal Degeneration: ", "Neurofibrillary tau impairs lexical access, triggering anomia, circumlocution, and extended unfilled hesitation pauses."),
        ("Hippocampal & Medial Temporal Atrophy: ", "Working memory lapses cause patients to lose their train of thought, drastically increasing overall silence ratio (R_sil)."),
        ("Frontal Motor Speech Degradation: ", "Loss of fine articulatory control results in pitch monotonicity (reduced F0 variability), vocal tremor, and reduced Harmonics-to-Noise Ratio (HNR)."),
        ("Diagnostic Window: ", "Acoustic and conversational timing anomalies manifest years before gross physical motor impairment.")
    ]
    for b_title, b_text in points2:
        p = tf2.add_paragraph()
        run1 = p.add_run()
        run1.text = "• " + b_title
        run1.font.bold = True
        run1.font.size = Pt(11.5)
        run1.font.color.rgb = DARK_BLUE
        run2 = p.add_run()
        run2.text = b_text
        run2.font.size = Pt(11)
        run2.font.color.rgb = DARK_GRAY

    # =========================================================================
    # SLIDE 4: PROBLEM DEFINITION
    # =========================================================================
    slide4 = prs.slides.add_slide(blank_layout)
    add_bg(slide4, 4)

    # Core Research Question Banner
    create_card(slide4, Inches(0.9), Inches(1.45), Inches(11.55), Inches(0.95), bg_color=LIGHT_BG, border_color=AMRITA_BLUE)
    qtb = slide4.shapes.add_textbox(Inches(1.1), Inches(1.5), Inches(11.15), Inches(0.85))
    qtf = qtb.text_frame
    qtf.word_wrap = True
    qp = qtf.paragraphs[0]
    qp.text = "Core Problem Statement:"
    qp.font.name = "Arial"
    qp.font.size = Pt(12)
    qp.font.bold = True
    qp.font.color.rgb = ACCENT_GREEN
    qp2 = qtf.add_paragraph()
    qp2.text = "How can we achieve highly accurate, fully automated Alzheimer's Dementia recognition directly from spontaneous raw speech without manual text transcripts, while avoiding dimensionality collapse and data leakage?"
    qp2.font.name = "Arial"
    qp2.font.size = Pt(13)
    qp2.font.bold = True
    qp2.font.color.rgb = DARK_BLUE

    # 4 Challenge Cards
    challenges = [
        ("Challenge 1: ASR Error Cascade", 
         "Existing high-performing models depend on manual text transcripts. Automated Speech Recognition (ASR) on elderly, dysarthric speech yields Word Error Rates (WER) of 30%–45%, which corrupt downstream NLP models.",
         ACCENT_RED),
        ("Challenge 2: Overlooked Diarization Dynamics", 
         "Standard speech models treat conversations as single audio streams, blending the acoustic features of healthy clinicians with cognitively impaired patients and severely corrupting the diagnostic signal.",
         DARK_BLUE),
        ("Challenge 3: Early Fusion Bottleneck", 
         "Naively concatenating 123 physical acoustic features with 512 deep embeddings creates a 650-dim space where high-variance spectral bins overpower gradients, completely suppressing 15 vital timing biomarkers.",
         DARK_BLUE),
        ("Challenge 4: Pervasive Data Leakage", 
         "Many published studies claim 95%–100% accuracy due to preprocessing, feature scaling, or selection across entire datasets prior to cross-validation splits, resulting in non-generalizable clinical failures.",
         ACCENT_RED)
    ]

    for idx, (ch_title, ch_desc, col_rgb) in enumerate(challenges):
        c_left = Inches(0.9 + (idx % 2) * 5.9)
        c_top = Inches(2.6 + (idx // 2) * 2.1)
        create_card(slide4, c_left, c_top, Inches(5.65), Inches(1.95), bg_color=WHITE)
        ctb = slide4.shapes.add_textbox(c_left + Inches(0.15), c_top + Inches(0.1), Inches(5.35), Inches(1.75))
        ctf = ctb.text_frame
        ctf.word_wrap = True
        
        p = ctf.paragraphs[0]
        p.text = ch_title
        p.font.name = "Arial"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = col_rgb
        
        p2 = ctf.add_paragraph()
        p2.text = ch_desc
        p2.font.name = "Arial"
        p2.font.size = Pt(11)
        p2.font.color.rgb = DARK_GRAY

    # =========================================================================
    # SLIDE 5: LITERATURE REVIEW (TABLE SLIDE)
    # =========================================================================
    slide5 = prs.slides.add_slide(blank_layout)
    add_bg(slide5, 5)

    # White mask over the placeholder headers to ensure sharp, clean native table
    create_card(slide5, Inches(0.8), Inches(1.35), Inches(11.75), Inches(5.3), bg_color=WHITE, border_color=CARD_BORDER)

    # Native PowerPoint Table
    table_shape = slide5.shapes.add_table(6, 6, Inches(0.9), Inches(1.45), Inches(11.55), Inches(5.0))
    table = table_shape.table
    col_widths = [Inches(0.6), Inches(1.8), Inches(1.5), Inches(2.4), Inches(2.75), Inches(2.5)]
    for idx, w in enumerate(col_widths):
        table.columns[idx].width = w

    headers = ["Sl No", "Title & Year", "Journal / Conf.", "Methodology", "Key Findings", "Limitations"]
    for idx, h in enumerate(headers):
        cell = table.cell(0, idx)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = AMRITA_BLUE
        p = cell.text_frame.paragraphs[0]
        p.font.name = "Arial"
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.alignment = PP_ALIGN.CENTER

    lit_data = [
        ("1", "The ADReSSo Challenge\n(Luz et al., 2021)", "Proc. Interspeech\n2021",
         "eGeMAPS acoustic feature sets (88-dim) + SVM classifier.",
         "Official acoustic baseline achieved 65.10% accuracy; Multimodal ASR+text reached 78.87%.",
         "Acoustic-only models underperformed; Multimodal baseline required text transcripts."),
        ("2", "Wav2Vec 2.0 Representations\n(Papasavvas et al., 2021)", "Proc. IEEE ICASSP\n2021",
         "Pretrained self-supervised audio representations fine-tuned on raw waveforms.",
         "Reached 78.20% accuracy directly from raw audio, surpassing handcrafted descriptors.",
         "Computationally heavy, high parameter count; decisions lack explainability for clinicians."),
        ("3", "Comparing Speech & Language\n(Balagopalan et al., 2020)", "Proc. Interspeech\n2020",
         "Comparative study of acoustic features vs pretrained BERT linguistic embeddings.",
         "Text embeddings derived from manual transcripts achieved strong diagnostic accuracy.",
         "Dependent entirely on manual text transcripts; highly vulnerable to ASR transcription errors."),
        ("4", "Paralinguistic Acoustic Features\n(Haider et al., 2020)", "IEEE JSTSP\nVol. 14, 2020",
         "Extensive empirical assessment of eGeMAPS, ComParE, and MRC psycholinguistic features.",
         "Proved pause-to-speech ratio and pitch dynamics are most discriminative biomarkers.",
         "Processed audio as continuous streams without clinician-patient speaker diarization."),
        ("5", "Automated Dialogue Analysis\n(Mirheidari et al., 2019)", "Computer Speech\n& Language, 2019",
         "Turn-taking latencies and conversational interaction analysis in memory clinics.",
         "Interaction parameters differentiated dementia patients comparable to expert clinicians.",
         "Small cohort size; evaluated only conversational timing without deep continuous spectral fusion.")
    ]

    for row_idx, data in enumerate(lit_data, start=1):
        for col_idx, text in enumerate(data):
            cell = table.cell(row_idx, col_idx)
            cell.text = text
            cell.fill.solid()
            cell.fill.fore_color.rgb = LIGHT_BG if row_idx % 2 == 1 else WHITE
            p = cell.text_frame.paragraphs[0]
            p.font.name = "Arial"
            p.font.size = Pt(10)
            p.font.color.rgb = DARK_GRAY
            if col_idx == 0:
                p.alignment = PP_ALIGN.CENTER
                p.font.bold = True
            elif col_idx == 1:
                p.font.bold = True
                p.font.color.rgb = DARK_BLUE

    # =========================================================================
    # SLIDE 6: GAPS IDENTIFIED
    # =========================================================================
    slide6 = prs.slides.add_slide(blank_layout)
    add_bg(slide6, 6)

    gaps = [
        ("GAP 1", "Cascading ASR Transcription Vulnerability",
         "Existing top-performing pipelines depend on manual transcripts or high-resource NLP models. Applying off-the-shelf ASR to cognitively impaired elderly speech yields high Word Error Rates (30%–45%), propagating catastrophic errors into classification.",
         "Our Solution: Complete audio-only processing eliminating text dependencies."),
        ("GAP 2", "Neglect of Conversational Turn Diarization",
         "Most acoustic frameworks analyze entire recordings as uniform streams, blindly combining clinician prompts with patient speech. This averages out subtle cognitive pauses and distorts hesitation markers.",
         "Our Solution: Diarization-based extraction isolating 15 participant hesitation biomarkers."),
        ("GAP 3", "Early Fusion Dimensionality & Scale Collapse",
         "Directly concatenating low-dimensional acoustics (123) with high-dimensional deep embeddings (512+) produces severe gradient dominance by high-variance spectral bins, drowning out vital timing features.",
         "Our Solution: Dynamic Multimodal Gated Fusion weighting calibrated posterior probabilities."),
        ("GAP 4", "Prevalent Data Leakage in Academic Benchmarks",
         "Several published works claim artificially high accuracies (>95%) by fitting normalizers, selecting features, or performing SMOTE augmentation across the entire dataset before establishing validation splits.",
         "Our Solution: Strict in-fold standardizers fit strictly on training partitions across 5 folds.")
    ]

    for idx, (tag, g_title, g_desc, g_sol) in enumerate(gaps):
        g_left = Inches(0.9 + (idx % 2) * 5.9)
        g_top = Inches(1.45 + (idx // 2) * 2.65)
        create_card(slide6, g_left, g_top, Inches(5.65), Inches(2.45), bg_color=WHITE)
        gtb = slide6.shapes.add_textbox(g_left + Inches(0.15), g_top + Inches(0.1), Inches(5.35), Inches(2.25))
        gtf = gtb.text_frame
        gtf.word_wrap = True
        
        p = gtf.paragraphs[0]
        r1 = p.add_run()
        r1.text = f"[{tag}]  "
        r1.font.bold = True
        r1.font.size = Pt(12)
        r1.font.color.rgb = ACCENT_RED
        r2 = p.add_run()
        r2.text = g_title
        r2.font.bold = True
        r2.font.size = Pt(12)
        r2.font.color.rgb = DARK_BLUE
        
        p2 = gtf.add_paragraph()
        p2.text = g_desc
        p2.font.name = "Arial"
        p2.font.size = Pt(10.5)
        p2.font.color.rgb = DARK_GRAY
        
        p3 = gtf.add_paragraph()
        p3.text = "✔ " + g_sol
        p3.font.name = "Arial"
        p3.font.size = Pt(10.5)
        p3.font.bold = True
        p3.font.color.rgb = ACCENT_GREEN

    # =========================================================================
    # SLIDE 7: OBJECTIVES
    # =========================================================================
    slide7 = prs.slides.add_slide(blank_layout)
    add_bg(slide7, 7)

    objectives = [
        ("Objective 1: Non-Invasive, Audio-Only Diagnostic Pipeline",
         "Develop an end-to-end framework capable of recognizing Alzheimer's Dementia directly from spontaneous continuous speech without requiring manual or automated text transcripts.",
         "Eliminates ASR transcription errors and cultural linguistic dependencies."),
        ("Objective 2: Extraction of 15 Conversational Timing Biomarkers",
         "Leverage speaker diarization boundaries to quantify cognitive processing latencies, specifically extracting silence ratio (R_sil), hesitation pause latency, and speech-to-silence ratio (gamma_ps).",
         "Directly mirrors neurobiological temporoparietal and hippocampal atrophy."),
        ("Objective 3: Design of Dynamic Multimodal Gated Fusion",
         "Formulate an adaptive probability gating mechanism (w_ac=0.10, w_dp=0.30, w_tm=0.60) with strict in-fold standardization to overcome the 650-dimensional early fusion bottleneck.",
         "Guarantees mathematically rigorous zero-leakage 5-fold cross-validation."),
        ("Objective 4: Dual-Task Continuous Cognitive Profiling",
         "Construct a Multi-Task Deep Neural Network (MultiTaskDNN) that simultaneously solves binary AD classification and regresses continuous Mini-Mental State Examination (MMSE) scores.",
         "Joint objective function: L_total = L_BCE + 0.5 * L_MSE."),
        ("Objective 5: Clinical Explainability via SHAP and LIME",
         "Deconstruct model predictions using game-theoretic SHAP beeswarm feature importance and LIME patient-level local surrogate breakdowns to satisfy clinical transparency standards.",
         "Provides clinicians with interpretable biomarker diagnostic justifications.")
    ]

    for idx, (o_title, o_desc, o_imp) in enumerate(objectives):
        o_top = Inches(1.45 + idx * 1.05)
        create_card(slide7, Inches(0.9), o_top, Inches(11.55), Inches(0.95), bg_color=WHITE)
        otb = slide7.shapes.add_textbox(Inches(1.05), o_top + Inches(0.08), Inches(11.25), Inches(0.8))
        otf = otb.text_frame
        otf.word_wrap = True
        
        p = otf.paragraphs[0]
        p.text = o_title
        p.font.name = "Arial"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = AMRITA_BLUE
        
        p2 = otf.add_paragraph()
        r1 = p2.add_run()
        r1.text = o_desc + "  "
        r1.font.size = Pt(11)
        r1.font.color.rgb = DARK_GRAY
        r2 = p2.add_run()
        r2.text = "[" + o_imp + "]"
        r2.font.size = Pt(10.5)
        r2.font.bold = True
        r2.font.color.rgb = ACCENT_GREEN

    # =========================================================================
    # SLIDE 8: METHODOLOGY
    # =========================================================================
    slide8 = prs.slides.add_slide(blank_layout)
    add_bg(slide8, 8)

    # Left Column: Stream descriptions
    create_card(slide8, Inches(0.9), Inches(1.45), Inches(5.1), Inches(5.2), bg_color=WHITE)
    mtb = slide8.shapes.add_textbox(Inches(1.05), Inches(1.55), Inches(4.8), Inches(5.0))
    mtf = mtb.text_frame
    mtf.word_wrap = True

    mp = mtf.paragraphs[0]
    mp.text = "Tri-Stream Feature Processing"
    mp.font.name = "Arial"
    mp.font.size = Pt(15)
    mp.font.bold = True
    mp.font.color.rgb = AMRITA_BLUE

    m_points = [
        ("Signal Preprocessing: ", "Resampling to 16 kHz, spectral gating stationary noise reduction, energy VAD (-25 dB threshold)."),
        ("Stream 1: Physical Acoustics (123-dim): ", "120 static + dynamic MFCCs (40+40+40), pitch F0 (mean, std) via pYIN with HMM Viterbi decoding, Harmonics-to-Noise Ratio (HNR)."),
        ("Stream 2: Deep Spectral Embeddings (512-dim): ", "128-band log-Mel filterbank envelope with frame-wise temporal mean and variance pooling."),
        ("Stream 3: Conversational Timing (15-dim): ", "15 diarization metrics: silence ratio R_sil, hesitation pause latency, phonation ratios R_par/R_inv, speech-to-silence ratio."),
        ("Dynamic Gated Fusion Formulation: ", "P_fused = 0.10*P_ac + 0.30*P_dp + 0.60*P_tm. Standardizers S^(k) fit exclusively on training folds."),
        ("Multi-Task Head: ", "Shared latent representation (256->128->64) branching into AD Classification (BCE) and MMSE Score Regression (MSE).")
    ]
    for m_head, m_body in m_points:
        p = mtf.add_paragraph()
        r1 = p.add_run()
        r1.text = "• " + m_head
        r1.font.bold = True
        r1.font.size = Pt(11)
        r1.font.color.rgb = DARK_BLUE
        r2 = p.add_run()
        r2.text = m_body
        r2.font.size = Pt(10.5)
        r2.font.color.rgb = DARK_GRAY

    # Right Column: Insert Flowchart Diagram
    create_card(slide8, Inches(6.2), Inches(1.45), Inches(6.25), Inches(5.2), bg_color=WHITE)
    flowchart_path = "Methodology Flowchart.png"
    if os.path.exists(flowchart_path):
        slide8.shapes.add_picture(flowchart_path, Inches(6.3), Inches(1.6), width=Inches(6.05), height=Inches(4.8))

    # =========================================================================
    # SLIDE 9: DATASET DESCRIPTION
    # =========================================================================
    slide9 = prs.slides.add_slide(blank_layout)
    add_bg(slide9, 9)

    # Card 1: Benchmark Overview
    create_card(slide9, Inches(0.9), Inches(1.45), Inches(11.55), Inches(1.2), bg_color=WHITE)
    dtb1 = slide9.shapes.add_textbox(Inches(1.1), Inches(1.5), Inches(11.15), Inches(1.05))
    dtf1 = dtb1.text_frame
    dtf1.word_wrap = True
    dp1 = dtf1.paragraphs[0]
    dp1.text = "Interspeech 2021 ADReSSo Challenge Corpus"
    dp1.font.name = "Arial"
    dp1.font.size = Pt(15)
    dp1.font.bold = True
    dp1.font.color.rgb = AMRITA_BLUE
    dp2 = dtf1.add_paragraph()
    dp2.text = "• Standardized benchmark dataset derived from the Boston Cookie Theft picture description task (Boston Diagnostic Aphasia Exam).\n• Evaluates spontaneous, unstructured speech under concurrent visual scanning, narrative planning, and lexical retrieval demands."
    dp2.font.name = "Arial"
    dp2.font.size = Pt(11.5)
    dp2.font.color.rgb = DARK_GRAY

    # 3 Cohort Cards
    d_cards = [
        ("Diagnostic Corpus (N = 166)", 
         "• 87 Alzheimer's Dementia (AD)\n• 79 Cognitively Normal (CN)\n• Strictly balanced by age and gender to eliminate demographic bias.\n• Serves as primary benchmark for 5-fold cross-validation.",
         AMRITA_BLUE),
        ("Cognitive Progression (N = 73)", 
         "• 15 Cognitive Decline cases\n• 58 Stable subjects\n• Longitudinal spontaneous speech tracking disease worsening over time.\n• Evaluates progressive deterioration sensitivity.",
         ACCENT_GREEN),
        ("Audio Specifications & Diarization", 
         "• 44.1 kHz standardized WAV audio.\n• Volume-normalized and cleaned.\n• Accompanied by turn-taking timestamps demarcating Participant (PAR) vs Investigator (INV).",
         DARK_BLUE)
    ]
    for idx, (c_title, c_text, c_col) in enumerate(d_cards):
        c_left = Inches(0.9 + idx * 3.9)
        create_card(slide9, c_left, Inches(2.85), Inches(3.75), Inches(3.75), bg_color=WHITE)
        ctb = slide9.shapes.add_textbox(c_left + Inches(0.15), Inches(3.0), Inches(3.45), Inches(3.4))
        ctf = ctb.text_frame
        ctf.word_wrap = True
        
        p = ctf.paragraphs[0]
        p.text = c_title
        p.font.name = "Arial"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = c_col
        
        p2 = ctf.add_paragraph()
        p2.text = c_text
        p2.font.name = "Arial"
        p2.font.size = Pt(11)
        p2.font.color.rgb = DARK_GRAY

    # =========================================================================
    # SLIDE 10: RESULTS & DISCUSSION (BENCHMARKS)
    # =========================================================================
    slide10 = prs.slides.add_slide(blank_layout)
    add_bg(slide10, 10)

    # 4 Metric Highlights
    metrics_top = [
        ("Accuracy", "80.16%", "+15.06% vs Baseline"),
        ("ROC-AUC", "0.8572", "High Discrimination"),
        ("F1-Score", "0.8059", "Balanced Class F1"),
        ("Precision / Recall", "0.840 / 0.792", "79.2% AD Sensitivity")
    ]
    for idx, (m_label, m_val, m_sub) in enumerate(metrics_top):
        m_left = Inches(0.9 + idx * 2.92)
        create_card(slide10, m_left, Inches(1.45), Inches(2.78), Inches(1.15), bg_color=LIGHT_BG, border_color=AMRITA_BLUE)
        mtb = slide10.shapes.add_textbox(m_left + Inches(0.08), Inches(1.5), Inches(2.62), Inches(1.0))
        mtf = mtb.text_frame
        mtf.word_wrap = True
        
        p = mtf.paragraphs[0]
        p.text = m_val
        p.font.name = "Arial"
        p.font.size = Pt(19)
        p.font.bold = True
        p.font.color.rgb = AMRITA_BLUE
        p.alignment = PP_ALIGN.CENTER
        
        p2 = mtf.add_paragraph()
        p2.text = f"{m_label} ({m_sub})"
        p2.font.name = "Arial"
        p2.font.size = Pt(10)
        p2.font.color.rgb = DARK_GRAY
        p2.alignment = PP_ALIGN.CENTER

    # Left: Benchmark Comparison Table
    create_card(slide10, Inches(0.9), Inches(2.75), Inches(6.2), Inches(3.9), bg_color=WHITE)
    ttb = slide10.shapes.add_textbox(Inches(1.05), Inches(2.85), Inches(5.9), Inches(0.4))
    ttf = ttb.text_frame
    tp = ttf.paragraphs[0]
    tp.text = "Benchmark Comparison Against Published Baselines"
    tp.font.name = "Arial"
    tp.font.size = Pt(13)
    tp.font.bold = True
    tp.font.color.rgb = AMRITA_BLUE

    bench_table_shape = slide10.shapes.add_table(5, 4, Inches(1.05), Inches(3.3), Inches(5.9), Inches(3.1))
    btable = bench_table_shape.table
    b_col_w = [Inches(2.5), Inches(1.5), Inches(1.0), Inches(0.9)]
    for idx, w in enumerate(b_col_w):
        btable.columns[idx].width = w

    b_headers = ["System / Model", "Features", "Accuracy", "F1-Score"]
    for idx, h in enumerate(b_headers):
        c = btable.cell(0, idx)
        c.text = h
        c.fill.solid()
        c.fill.fore_color.rgb = AMRITA_BLUE
        p = c.text_frame.paragraphs[0]
        p.font.name = "Arial"
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.alignment = PP_ALIGN.CENTER

    b_rows = [
        ("Acoustic Baseline (IS2021)", "eGeMAPS", "65.10%", "0.640"),
        ("Wav2Vec 2.0 (Papasavvas)", "Audio Repr.", "78.20%", "0.779"),
        ("Official Multimodal (IS2021)", "Audio + ASR Text", "78.87%", "0.779"),
        ("Proposed Gated Fusion", "Audio + Timing", "80.16%", "0.806")
    ]
    for r_idx, r_data in enumerate(b_rows, start=1):
        for c_idx, val in enumerate(r_data):
            cell = btable.cell(r_idx, c_idx)
            cell.text = val
            cell.fill.solid()
            if r_idx == 4:
                cell.fill.fore_color.rgb = RGBColor(230, 245, 235)
            else:
                cell.fill.fore_color.rgb = LIGHT_BG if r_idx % 2 == 1 else WHITE
            p = cell.text_frame.paragraphs[0]
            p.font.name = "Arial"
            p.font.size = Pt(10)
            if r_idx == 4:
                p.font.bold = True
                p.font.color.rgb = ACCENT_GREEN
            else:
                p.font.color.rgb = DARK_GRAY
            if c_idx >= 2:
                p.alignment = PP_ALIGN.CENTER

    # Right: Insert ROC Curves Plot
    create_card(slide10, Inches(7.3), Inches(2.75), Inches(5.15), Inches(3.9), bg_color=WHITE)
    roc_path = "roc_auc_curves.png"
    if os.path.exists(roc_path):
        slide10.shapes.add_picture(roc_path, Inches(7.4), Inches(2.85), width=Inches(4.95), height=Inches(3.65))

    # =========================================================================
    # SLIDE 11: RESULTS & DISCUSSION (ABLATION & XAI)
    # =========================================================================
    slide11 = prs.slides.add_slide(blank_layout)
    add_bg(slide11, 11)

    # Left: Ablation Study Card
    create_card(slide11, Inches(0.9), Inches(1.45), Inches(5.65), Inches(5.2), bg_color=WHITE)
    atb = slide11.shapes.add_textbox(Inches(1.05), Inches(1.55), Inches(5.35), Inches(0.4))
    atf = atb.text_frame
    ap = atf.paragraphs[0]
    ap.text = "Modality Ablation Study Matrix"
    ap.font.name = "Arial"
    ap.font.size = Pt(14)
    ap.font.bold = True
    ap.font.color.rgb = AMRITA_BLUE

    abl_table_shape = slide11.shapes.add_table(6, 3, Inches(1.05), Inches(2.0), Inches(5.35), Inches(2.2))
    atable = abl_table_shape.table
    atable.columns[0].width = Inches(3.15)
    atable.columns[1].width = Inches(1.1)
    atable.columns[2].width = Inches(1.1)

    a_heads = ["Feature Representation", "Accuracy", "F1-Score"]
    for idx, h in enumerate(a_heads):
        c = atable.cell(0, idx)
        c.text = h
        c.fill.solid()
        c.fill.fore_color.rgb = AMRITA_BLUE
        p = c.text_frame.paragraphs[0]
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.alignment = PP_ALIGN.CENTER

    a_rows = [
        ("Acoustics Only (123-dim)", "64.46%", "0.6686"),
        ("Deep Embeddings (512-dim)", "71.11%", "0.7226"),
        ("Conversational Timing (15-dim)", "72.35%", "0.7307"),
        ("Early Concatenation (650-dim)", "73.53%", "0.7373"),
        ("Proposed Dynamic Gated Fusion", "80.16%", "0.8059")
    ]
    for r_idx, r_data in enumerate(a_rows, start=1):
        for c_idx, val in enumerate(r_data):
            cell = atable.cell(r_idx, c_idx)
            cell.text = val
            cell.fill.solid()
            if r_idx == 5:
                cell.fill.fore_color.rgb = RGBColor(230, 245, 235)
            else:
                cell.fill.fore_color.rgb = LIGHT_BG if r_idx % 2 == 1 else WHITE
            p = cell.text_frame.paragraphs[0]
            p.font.name = "Arial"
            p.font.size = Pt(10)
            if r_idx == 5:
                p.font.bold = True
                p.font.color.rgb = ACCENT_GREEN
            if c_idx >= 1:
                p.alignment = PP_ALIGN.CENTER

    # Diagnostic Error Breakdown text below table
    etb = slide11.shapes.add_textbox(Inches(1.05), Inches(4.35), Inches(5.35), Inches(2.15))
    etf = etb.text_frame
    etf.word_wrap = True
    ep = etf.paragraphs[0]
    ep.text = "Clinical Error Breakdown (15 FP / 18 FN):"
    ep.font.name = "Arial"
    ep.font.size = Pt(11.5)
    ep.font.bold = True
    ep.font.color.rgb = ACCENT_RED

    ep_pts = [
        "• False Positives (N=15): Driven by visual scanning latencies (2.5s–4.0s picture inspection) and denture vocal noise lowering HNR.",
        "• False Negatives (N=18): Driven by brief 15s descriptions focusing only on obvious center details with preserved motor speech tempo."
    ]
    for pt in ep_pts:
        p = etf.add_paragraph()
        p.text = pt
        p.font.name = "Arial"
        p.font.size = Pt(10)
        p.font.color.rgb = DARK_GRAY

    # Right: Insert Confusion Matrix and SHAP plot side-by-side / stacked
    create_card(slide11, Inches(6.8), Inches(1.45), Inches(5.65), Inches(5.2), bg_color=WHITE)
    xtb = slide11.shapes.add_textbox(Inches(6.95), Inches(1.55), Inches(5.35), Inches(0.35))
    xtf = xtb.text_frame
    xp = xtf.paragraphs[0]
    xp.text = "Explainable AI (SHAP Beeswarm) & Confusion Matrix"
    xp.font.name = "Arial"
    xp.font.size = Pt(13)
    xp.font.bold = True
    xp.font.color.rgb = AMRITA_BLUE

    cm_path = "cm_proposed_multimodal_gated_fusion.png"
    shap_path = "shap_summary.png"
    if os.path.exists(cm_path):
        slide11.shapes.add_picture(cm_path, Inches(6.95), Inches(1.95), width=Inches(2.55), height=Inches(2.35))
    if os.path.exists(shap_path):
        slide11.shapes.add_picture(shap_path, Inches(9.65), Inches(1.95), width=Inches(2.65), height=Inches(2.35))

    # XAI takeaways text
    xtb2 = slide11.shapes.add_textbox(Inches(6.95), Inches(4.4), Inches(5.35), Inches(2.1))
    xtf2 = xtb2.text_frame
    xtf2.word_wrap = True
    p = xtf2.paragraphs[0]
    p.text = "Key Clinical Interpretability Findings:"
    p.font.name = "Arial"
    p.font.size = Pt(11.5)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GREEN

    x_pts = [
        "1. Silence Ratio (R_sil): Highest global importance (+0.30 to +0.55 SHAP attribution favoring AD).",
        "2. Pause Latency (tau_pause): Lexical hesitation latency consistently drives dementia predictions.",
        "3. Pitch Flattening (sigma_F0): Monotonous prosody correlates strongly with advanced cognitive decline."
    ]
    for pt in x_pts:
        p = xtf2.add_paragraph()
        p.text = pt
        p.font.name = "Arial"
        p.font.size = Pt(10)
        p.font.color.rgb = DARK_GRAY

    # =========================================================================
    # SLIDE 12: SUMMARY AND CONCLUSION
    # =========================================================================
    slide12 = prs.slides.add_slide(blank_layout)
    add_bg(slide12, 12)

    conclusions = [
        ("1. State-of-the-Art Audio-Only Performance",
         "The proposed Dynamic Multimodal Gated Fusion achieved 80.16% accuracy, 0.8059 F1-score, and 0.8572 ROC-AUC under rigorous zero-leakage 5-fold cross-validation across all 166 diagnostic sessions.",
         AMRITA_BLUE),
        ("2. Outperformed Official Text-Dependent Baselines",
         "Surpassed the official IS2021 acoustic baseline (65.10% by +15.06%), fine-tuned Wav2Vec 2.0 (78.20% by +1.96%), and the multimodal benchmark (78.87% by +1.29%) completely without manual text transcripts.",
         ACCENT_GREEN),
        ("3. Resolved the Early Concatenation Bottleneck",
         "Dynamic probability gating successfully balanced acoustic, deep spectral, and timing dynamics, validating that conversational hesitation patterns carry greater diagnostic weight than raw acoustic descriptors.",
         DARK_BLUE),
        ("4. Clinical Interpretability & Telemedicine Feasibility",
         "SHAP and LIME confirmed that silence ratios and pause latencies drive decisions. The system enables passive, 1-to-2 minute non-invasive screening scalable for primary healthcare and remote smartphone telehealth.",
         ACCENT_GREEN)
    ]

    for idx, (c_head, c_body, col_rgb) in enumerate(conclusions):
        c_top = Inches(1.45 + idx * 1.3)
        create_card(slide12, Inches(0.9), c_top, Inches(11.55), Inches(1.18), bg_color=WHITE)
        ctb = slide12.shapes.add_textbox(Inches(1.1), c_top + Inches(0.1), Inches(11.15), Inches(0.98))
        ctf = ctb.text_frame
        ctf.word_wrap = True
        
        p = ctf.paragraphs[0]
        p.text = c_head
        p.font.name = "Arial"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = col_rgb
        
        p2 = ctf.add_paragraph()
        p2.text = c_body
        p2.font.name = "Arial"
        p2.font.size = Pt(11)
        p2.font.color.rgb = DARK_GRAY

    # =========================================================================
    # SLIDE 13: COMPLETION PLAN & CONTRIBUTIONS
    # =========================================================================
    slide13 = prs.slides.add_slide(blank_layout)
    add_bg(slide13, 13)

    # Left: Team Contributions (3 Cards)
    create_card(slide13, Inches(0.9), Inches(1.45), Inches(5.65), Inches(5.2), bg_color=WHITE)
    ctb = slide13.shapes.add_textbox(Inches(1.05), Inches(1.55), Inches(5.35), Inches(5.0))
    ctf = ctb.text_frame
    ctf.word_wrap = True

    p = ctf.paragraphs[0]
    p.text = "Individual Team Contributions"
    p.font.name = "Arial"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = AMRITA_BLUE

    contrib_list = [
        ("Divagar S (CB.SC.U4AIE23223): ", "Overall architecture, Dynamic Gated Fusion formulation, Multi-Task loss, SHAP/LIME clinical interpretability, LaTeX report & Git lead."),
        ("Abhishek Pandey (CB.SC.U4AIE23205): ", "Signal preprocessing (spectral gating & VAD), 512-dim continuous log-Mel filterbank extraction, 15 conversational timing metrics from diarization."),
        ("Prem (CB.SC.U4AIE23241): ", "123 physical acoustic biomarkers (dynamic MFCCs, pYIN F0, HNR), baseline machine learning benchmarks, modality ablation study, and cohort error analysis.")
    ]
    for c_author, c_desc in contrib_list:
        p = ctf.add_paragraph()
        r1 = p.add_run()
        r1.text = "• " + c_author
        r1.font.bold = True
        r1.font.size = Pt(11)
        r1.font.color.rgb = DARK_BLUE
        r2 = p.add_run()
        r2.text = c_desc
        r2.font.size = Pt(10.5)
        r2.font.color.rgb = DARK_GRAY

    # Right: Completion Roadmap & Clinical Next Steps
    create_card(slide13, Inches(6.8), Inches(1.45), Inches(5.65), Inches(5.2), bg_color=WHITE)
    rtb = slide13.shapes.add_textbox(Inches(6.95), Inches(1.55), Inches(5.35), Inches(5.0))
    rtf = rtb.text_frame
    rtf.word_wrap = True

    p = rtf.paragraphs[0]
    p.text = "Deployment Roadmap & Future Directions"
    p.font.name = "Arial"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GREEN

    next_steps = [
        ("Foundation Speech Models: ", "Integrate self-supervised HuBERT and WavLM representations paired with adaptive noise-cancellation front-ends."),
        ("Cross-Lingual Generalization: ", "Validate timing and pause metrics on multilingual corpora (Spanish, Mandarin, Tamil) to establish cross-lingual baselines."),
        ("Multi-Center Longitudinal Cohorts: ", "Expand evaluations to full DementiaBank Pitt database to assess long-term progression trajectories."),
        ("Lightweight Telehealth Edge App: ", "Export quantized ONNX model for real-time passive voice screening on consumer smartphones without cloud transmission."),
        ("Clinical Decision Support: ", "Package LIME/SHAP patient explanation summaries into intuitive dashboard reports for memory clinic physicians.")
    ]
    for r_head, r_desc in next_steps:
        p = rtf.add_paragraph()
        r1 = p.add_run()
        r1.text = "✔ " + r_head
        r1.font.bold = True
        r1.font.size = Pt(11)
        r1.font.color.rgb = DARK_BLUE
        r2 = p.add_run()
        r2.text = r_desc
        r2.font.size = Pt(10.5)
        r2.font.color.rgb = DARK_GRAY

    # =========================================================================
    # SLIDE 14: REFERENCES
    # =========================================================================
    slide14 = prs.slides.add_slide(blank_layout)
    add_bg(slide14, 14)

    create_card(slide14, Inches(0.9), Inches(1.45), Inches(11.55), Inches(5.2), bg_color=WHITE)
    reftb = slide14.shapes.add_textbox(Inches(1.1), Inches(1.55), Inches(11.15), Inches(5.0))
    reftf = reftb.text_frame
    reftf.word_wrap = True

    p = reftf.paragraphs[0]
    p.text = "Key Verified Academic Literature (with Working DOIs)"
    p.font.name = "Arial"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = AMRITA_BLUE

    refs = [
        ("[1] S. Luz et al., ", "Alzheimer's Dementia Recognition through Spontaneous Speech: The ADReSSo Challenge, ", "in Proc. Interspeech 2021, pp. 3780–3784. DOI: 10.21437/Interspeech.2021-1220"),
        ("[2] A. Papasavvas et al., ", "Detecting Alzheimer's Disease from Spontaneous Speech Using Wav2Vec 2.0 Representations, ", "in Proc. IEEE ICASSP 2021, pp. 9414–9418. DOI: 10.1109/ICASSP40776.2021.9414545"),
        ("[3] A. Balagopalan et al., ", "To BERT or Not to BERT: Comparing Speech and Language-Based Approaches for Alzheimer's Disease Detection, ", "in Proc. Interspeech 2020, pp. 2167–2171. DOI: 10.21437/Interspeech.2020-2557"),
        ("[4] F. Haider et al., ", "An Assessment of Paralinguistic Acoustic Features for Detection of Alzheimer's Dementia in Spontaneous Speech, ", "IEEE JSTSP, vol. 14, no. 2, pp. 272–281, 2020. DOI: 10.1109/JSTSP.2019.2955022"),
        ("[5] S. M. Lundberg and S.-I. Lee, ", "A Unified Approach to Interpreting Model Predictions (SHAP), ", "in Advances in Neural Information Processing Systems (NeurIPS), vol. 30, pp. 4765–4774, 2017. DOI: 10.48550/arXiv.1705.07874")
    ]
    for r_num, r_title, r_venue in refs:
        p = reftf.add_paragraph()
        r1 = p.add_run()
        r1.text = r_num
        r1.font.bold = True
        r1.font.size = Pt(11)
        r1.font.color.rgb = AMRITA_BLUE
        r2 = p.add_run()
        r2.text = f'"{r_title}" '
        r2.font.bold = True
        r2.font.size = Pt(11)
        r2.font.color.rgb = DARK_BLUE
        r3 = p.add_run()
        r3.text = r_venue
        r3.font.size = Pt(10.5)
        r3.font.color.rgb = DARK_GRAY

    # =========================================================================
    # SLIDE 15: THANK YOU
    # =========================================================================
    slide15 = prs.slides.add_slide(blank_layout)
    add_bg(slide15, 15)

    create_card(slide15, Inches(1.8), Inches(1.4), Inches(9.733), Inches(5.1), bg_color=WHITE, border_color=CARD_BORDER)

    thtb = slide15.shapes.add_textbox(Inches(2.0), Inches(1.6), Inches(9.333), Inches(4.7))
    thtf = thtb.text_frame
    thtf.word_wrap = True

    p = thtf.paragraphs[0]
    p.text = "Thank You!"
    p.font.name = "Arial"
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = AMRITA_BLUE
    p.alignment = PP_ALIGN.CENTER

    p2 = thtf.add_paragraph()
    p2.text = "Questions & Discussion"
    p2.font.name = "Arial"
    p2.font.size = Pt(18)
    p2.font.bold = True
    p2.font.color.rgb = ACCENT_GREEN
    p2.alignment = PP_ALIGN.CENTER

    p3 = thtf.add_paragraph()
    p3.text = "\nMultimodal Speech Biomarker Fusion for Automated Alzheimer's Dementia Recognition\nDepartment of Artificial Intelligence and Data Science | Amrita Vishwa Vidyapeetham"
    p3.font.name = "Arial"
    p3.font.size = Pt(13)
    p3.font.color.rgb = DARK_BLUE
    p3.alignment = PP_ALIGN.CENTER

    p4 = thtf.add_paragraph()
    p4.text = "\nTeam: Divagar S  |  Abhishek Pandey  |  Prem"
    p4.font.name = "Arial"
    p4.font.size = Pt(13)
    p4.font.bold = True
    p4.font.color.rgb = DARK_GRAY
    p4.alignment = PP_ALIGN.CENTER

    p5 = thtf.add_paragraph()
    p5.text = "GitHub Repository: https://github.com/divagar127/Alzimer-detection-using-speech-processing"
    p5.font.name = "Arial"
    p5.font.size = Pt(12)
    p5.font.color.rgb = AMRITA_BLUE
    p5.alignment = PP_ALIGN.CENTER

    # Save Presentation
    output_filename = "final_project_presentation.pptx"
    prs.save(output_filename)
    print(f"Successfully generated {output_filename} with {len(prs.slides)} slides!")

if __name__ == '__main__':
    build_presentation()
