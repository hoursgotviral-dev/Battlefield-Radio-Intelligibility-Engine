"""
Generate the official pitch deck presentation (.pptx) for the
Battlefield Radio Intelligibility Engine — Snapdragon AI Lab Challenge.
"""

import os
import sys
from pathlib import Path
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE


def create_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)  # 16:9 widescreen
    prs.slide_height = Inches(7.5)

    # Color Palette: Tactical Dark Theme
    BG_COLOR = RGBColor(11, 15, 25)        # #0b0f19 Dark tactical
    CARD_BG = RGBColor(17, 24, 39)         # #111827 Dark card
    CYAN = RGBColor(0, 240, 255)           # #00f0ff Neon cyan
    AMBER = RGBColor(251, 191, 36)         # #fbbf24 Warm amber
    WHITE = RGBColor(243, 244, 246)        # #f3f4f6 Off-white
    GRAY = RGBColor(156, 163, 175)         # #9ca3af Cool gray
    RED = RGBColor(248, 113, 113)          # #f87171 Tactical red

    def apply_background(slide):
        background = slide.background
        fill = background.fill
        fill.solid()
        fill.fore_color.rgb = BG_COLOR

    def add_header(slide, title_text, subtitle_text=""):
        # Header box
        header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(1.1))
        tf = header_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        
        p = tf.paragraphs[0]
        p.text = title_text
        p.font.name = "Arial"
        p.font.size = Pt(26)
        p.font.bold = True
        p.font.color.rgb = WHITE
        
        if subtitle_text:
            p2 = tf.add_paragraph()
            p2.text = subtitle_text
            p2.font.name = "Arial"
            p2.font.size = Pt(13)
            p2.font.color.rgb = CYAN

    # ==========================================
    # SLIDE 1: Title Slide
    # ==========================================
    blank_layout = prs.slide_layouts[6]
    s1 = prs.slides.add_slide(blank_layout)
    apply_background(s1)

    tb = s1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.3), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "BATTLEFIELD RADIO INTELLIGIBILITY ENGINE"
    p.font.name = "Arial"
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = CYAN

    p = tf.add_paragraph()
    p.text = "Real-Time On-Device Speech Enhancement Deployed on Qualcomm Snapdragon Hexagon NPU"
    p.font.name = "Arial"
    p.font.size = Pt(18)
    p.font.color.rgb = WHITE
    p.space_after = Pt(24)

    p = tf.add_paragraph()
    p.text = "• Target Hardware: Snapdragon X Elite (Qualcomm Hexagon NPU — 411 µs / 100% On-NPU)\n" \
             "• Dual Tactical Modes: Intelligibility Mode (0.409 WER) vs. Quality Mode (2.398 PESQ)\n" \
             "• Full Deployment: QNN Context Binary (Job jg9zozmlp) + Profile (Job jp1nonj2g) + INT8 Quantized"
    p.font.name = "Arial"
    p.font.size = Pt(14)
    p.font.color.rgb = GRAY

    p = tf.add_paragraph()
    p.text = "Qualcomm Snapdragon AI Lab Build & Present Challenge | September 2026"
    p.font.name = "Arial"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = AMBER
    p.space_before = Pt(28)

    # ==========================================
    # SLIDE 2: The Tactical Problem
    # ==========================================
    s2 = prs.slides.add_slide(blank_layout)
    apply_background(s2)
    add_header(s2, "The Tactical Problem: Why Standard Denoising Fails in Combat", "High-SPL acoustic shockwaves, RF bandpass restrictions, and low-latency constraints")

    # 3 Cards
    card_w = Inches(3.64)
    card_h = Inches(4.8)
    gap = Inches(0.4)
    
    cards_data = [
        ("1. Extreme Acoustic Cascade", RED, [
            "• High-SPL weapon impulse transients (gunfire/artillery).",
            "• Heavy low-frequency mechanical rumble (tanks, diesel, rotor wash).",
            "• Signal-to-Noise Ratios plummeting below -5 dB to +2 dB.",
        ]),
        ("2. Severe RF Bottlenecks", AMBER, [
            "• RF channel bandpass filtering (300 Hz - 3400 Hz).",
            "• Narrowband vocoder quantization (Codec2/MELP 1200-2400 bps).",
            "• Non-linear PTT preamp clipping and saturation.",
        ]),
        ("3. Edge Hardware Barriers", CYAN, [
            "• Heavy models (DeepFilterNet3, Demucs) fail NPU export.",
            "• Complex STFT ops cause 100% CPU fallback and latency blowup.",
            "• Standard small models hallucinate and collapse at low SNR.",
        ])
    ]

    for idx, (ctitle, ccolor, bullet_list) in enumerate(cards_data):
        left = Inches(0.8) + idx * (card_w + gap)
        top = Inches(1.8)
        shape = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, card_w, card_h)
        shape.fill.solid()
        shape.fill.fore_color.rgb = CARD_BG
        shape.line.color.rgb = ccolor
        shape.line.width = Pt(1.5)

        tb = s2.shapes.add_textbox(left + Inches(0.2), top + Inches(0.2), card_w - Inches(0.4), card_h - Inches(0.4))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = ctitle
        p.font.name = "Arial"
        p.font.size = Pt(16)
        p.font.bold = True
        p.font.color.rgb = ccolor
        p.space_after = Pt(12)

        for b in bullet_list:
            p = tf.add_paragraph()
            p.text = b
            p.font.name = "Arial"
            p.font.size = Pt(12)
            p.font.color.rgb = WHITE
            p.space_after = Pt(6)

    # ==========================================
    # SLIDE 3: System Architecture
    # ==========================================
    s3 = prs.slides.add_slide(blank_layout)
    apply_background(s3)
    add_header(s3, "System Architecture: 3-Stage Co-Designed Pipeline", "Host DSP Causal STFT -> 100% Hexagon NPU Neural Core -> Guardrail -> Post-Filter")

    # Left: Text breakdown, Right: Embed Image
    left_tb = s3.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(5.8), Inches(5.0))
    tf = left_tb.text_frame
    tf.word_wrap = True

    sections = [
        ("1. Host Causal STFT (32 ms Chunk)", CYAN, "Causal Hanning window (N_FFT=512, Hop=128, 4 frames). Feeds magnitude chunk [1, 1, 257, 4] into NPU."),
        ("2. Branch A Neural Core (Snapdragon NPU)", CYAN, "CausalConv2d + 2-layer stateful GRU bottleneck. Computes real spectral mask with zero CPU fallback (411 µs)."),
        ("3. Deterministic Reliability Guardrail", AMBER, "Monitors energy ratio, spectral flatness, and envelope correlation. Soft-blends input to prevent speech collapse."),
        ("4. DSP Spectral Post-Filter", WHITE, "Wiener-style stationary noise reduction and temporal smoothing for crystal-clear tactical audio."),
    ]
    for i, (stitle, scolor, sdesc) in enumerate(sections):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = stitle
        p.font.name = "Arial"
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = scolor
        p = tf.add_paragraph()
        p.text = sdesc
        p.font.name = "Arial"
        p.font.size = Pt(11)
        p.font.color.rgb = GRAY
        p.space_after = Pt(10)

    # Right side: Add image if available
    img_path = Path("demo/assets/architecture_diagram.png")
    if img_path.exists():
        s3.shapes.add_picture(str(img_path), Inches(6.8), Inches(1.8), width=Inches(5.7))

    # ==========================================
    # SLIDE 4: Physical Hardware Profiling Data
    # ==========================================
    s4 = prs.slides.add_slide(blank_layout)
    apply_background(s4)
    add_header(s4, "Qualcomm AI Hub: Physical Snapdragon Silicon Validation", "Verified cloud device profile on Snapdragon X Elite Hexagon NPU")

    stats = [
        ("100.0%", "NPU Operator Residency", "240/240 Ops on NPU (0 CPU Fallback)", CYAN),
        ("411.0 µs", "Median Chunk Latency", "32ms Chunk Budget (1.28% NPU Load)", CYAN),
        ("77.86×", "Real-Time Throughput", "RTF = 0.01284 (Host CPU is 28× RTF)", AMBER),
        ("13.77 MB", "Peak Memory Footprint", "Ultra-lean RAM footprint for tactical SDR", WHITE),
    ]

    for idx, (val, label, sub, color) in enumerate(stats):
        col = idx % 2
        row = idx // 2
        l = Inches(0.8) + col * Inches(5.9)
        t = Inches(1.8) + row * Inches(2.5)
        shape = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, l, t, Inches(5.6), Inches(2.2))
        shape.fill.solid()
        shape.fill.fore_color.rgb = CARD_BG
        shape.line.color.rgb = color
        shape.line.width = Pt(1.5)

        tb = s4.shapes.add_textbox(l + Inches(0.3), t + Inches(0.2), Inches(5.0), Inches(1.8))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = val
        p.font.name = "Arial"
        p.font.size = Pt(32)
        p.font.bold = True
        p.font.color.rgb = color

        p = tf.add_paragraph()
        p.text = label
        p.font.name = "Arial"
        p.font.size = Pt(15)
        p.font.bold = True
        p.font.color.rgb = WHITE

        p = tf.add_paragraph()
        p.text = sub
        p.font.name = "Arial"
        p.font.size = Pt(11)
        p.font.color.rgb = GRAY

    # Add Job IDs note
    note_tb = s4.shapes.add_textbox(Inches(0.8), Inches(6.6), Inches(11.7), Inches(0.5))
    tf = note_tb.text_frame
    p = tf.paragraphs[0]
    p.text = "Qualcomm AI Hub Verified Job IDs: Compile: jg9zozmlp (Success) | Profile: jp1nonj2g (Success) | Target: Snapdragon X Elite"
    p.font.name = "Arial"
    p.font.size = Pt(11)
    p.font.color.rgb = GRAY

    # ==========================================
    # SLIDE 5: Pipeline Optimization Journey
    # ==========================================
    s5 = prs.slides.add_slide(blank_layout)
    apply_background(s5)
    add_header(s5, "Engineering Iteration: Pipeline Optimization Journey", "Systematic post-processing and guard tuning on validation set without model retraining")

    journey_tb = s5.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(4.8))
    tf = journey_tb.text_frame
    tf.word_wrap = True

    steps = [
        ("Step 1: Baseline Deployment (v1)", "Branch A Causal ConvGRU + Default Guard. Baseline deployed on Snapdragon NPU achieving 411 µs latency (PESQ: 1.232, STOI: 0.758)."),
        ("Step 2: Reliability Guard Hyperparameter Tuning (Phase 8A — ASR Mode)", "Systematic grid search on validation set (energy ratios, spectral flatness, envelope correlation). Yields optimal Whisper WER of 0.409 (median: 0.353 vs 0.364 baseline)."),
        ("Step 3: Host-Side DSP Spectral Post-Filter (Phase 8B — Quality Mode)", "Classical Wiener post-filter applied after iSTFT synthesis. Squelches stationary radio hiss, boosting test PESQ from 1.879 to 2.398 (+0.519 delta; Val: 2.413) with STOI 0.944."),
        ("Step 4: INT8 Dynamic Quantization (Phase 8C)", "ONNX Runtime dynamic quantization generated INT8 model (artifacts/branch_a_denoiser_int8.onnx, 6.51 MB) for memory-constrained edge SDR."),
    ]
    for i, (stitle, sdesc) in enumerate(steps):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"• {stitle}"
        p.font.name = "Arial"
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = CYAN
        p = tf.add_paragraph()
        p.text = f"  {sdesc}"
        p.font.name = "Arial"
        p.font.size = Pt(12)
        p.font.color.rgb = WHITE
        p.space_after = Pt(12)

    # ==========================================
    # SLIDE 6: Comparative Benchmark Analysis
    # ==========================================
    s6 = prs.slides.add_slide(blank_layout)
    apply_background(s6)
    add_header(s6, "Benchmark Results: Tactical Intelligibility & Deployability", "Direct comparison against raw degraded baseline and heavy non-deployable baselines")

    tb = s6.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf = tb.text_frame
    tf.word_wrap = True

    comp_text = [
        ("Tactical Mission Flexibility: Quality Mode vs. Intelligibility Mode", AMBER),
        ("In military communications, automated ASR command-and-control requires preserving consonant transients, while human operators in noisy armored cockpits require background hiss squelch.", WHITE),
        ("Key Comparative Insights & Canonical Test Set Results:", CYAN),
        ("1. Intelligibility Mode (Branch A + Guard): Minimizes Word Error Rate (0.409 mean / 0.353 median WER vs 0.443 / 0.364 raw degraded) for automated Whisper ingestion.", WHITE),
        ("2. Quality Mode (+ Spectral Post-Filter): Squelches residual static for human ears, achieving a massive +0.52 PESQ jump (1.879 -> 2.398, STOI: 0.944). Toggleable in real time.", WHITE),
        ("3. 100% NPU Residency: 240/240 ops on Hexagon NPU (0 CPU fallback). DeepFilterNet3 cannot be compiled due to complex STFT ops.", WHITE),
        ("4. 77× Real-Time Throughput: 411 µs latency per 32 ms chunk leaves 98.7% of NPU compute budget free for concurrent ASR & crypto.", WHITE),
    ]
    for i, (title, color) in enumerate(comp_text):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = title
        p.font.name = "Arial"
        p.font.size = Pt(13) if "Insights" not in title and "Tactical" not in title else Pt(15)
        p.font.bold = "Insights" in title or "Tactical" in title
        p.font.color.rgb = color
        p.space_after = Pt(6)

    # ==========================================
    # SLIDE 7: Out-of-Distribution Tactical Scenarios
    # ==========================================
    s7 = prs.slides.add_slide(blank_layout)
    apply_background(s7)
    add_header(s7, "Out-of-Distribution Robustness: Real Combat Scenarios", "Validated on extreme operational acoustic conditions")

    scenarios = [
        ("Tank Heavy Rumble (-5 dB SNR)", "Armored column diesel track rumble. Guardrail preserves formant energy and intelligibility."),
        ("Urban Firefight (0 dB SNR)", "High-crest gunfire impulses and PTT clipping. Soft mask suppresses blast tails."),
        ("Rotorcraft Wash (+3 dB SNR)", "Cyclic helicopter blade turbulence. Recurrent GRU tracks non-stationary envelope."),
        ("Radio Codec2 Drop (+8 dB SNR)", "2400 bps vocoder packet loss & RF static. Spectral post-filter cleans noise floor."),
    ]
    for idx, (stitle, sdesc) in enumerate(scenarios):
        col = idx % 2
        row = idx // 2
        l = Inches(0.8) + col * Inches(5.9)
        t = Inches(1.8) + row * Inches(2.5)
        shape = s7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, l, t, Inches(5.6), Inches(2.2))
        shape.fill.solid()
        shape.fill.fore_color.rgb = CARD_BG
        shape.line.color.rgb = CYAN
        shape.line.width = Pt(1.5)

        tb = s7.shapes.add_textbox(l + Inches(0.3), t + Inches(0.2), Inches(5.0), Inches(1.8))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = stitle
        p.font.name = "Arial"
        p.font.size = Pt(15)
        p.font.bold = True
        p.font.color.rgb = AMBER
        p.space_after = Pt(6)

        p = tf.add_paragraph()
        p.text = sdesc
        p.font.name = "Arial"
        p.font.size = Pt(12)
        p.font.color.rgb = WHITE

    # ==========================================
    # SLIDE 8: Live Demo & Deliverables
    # ==========================================
    s8 = prs.slides.add_slide(blank_layout)
    apply_background(s8)
    add_header(s8, "Interactive Demo & Open-Source Artifacts", "Complete ecosystem ready for deployment and evaluation")

    deliv_tb = s8.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf = deliv_tb.text_frame
    tf.word_wrap = True

    delivs = [
        ("Interactive Gradio Web Demo (demo/gradio_app.py)", "Live A/B audio player, 3-panel waveform & spectrogram inspection, metrics diffs, and real-time streaming ONNX processing."),
        ("Command-Line Engine (demo/app.py)", "Batch and single-utterance streaming inference with latency profiling and transcript evaluation."),
        ("Deployment Toolchain (deploy/)", "ONNX exporter with static tensors, ONNX Runtime numeric parity validator, and INT8 dynamic quantization script."),
        ("Full Test & Benchmark Suite (tests/, eval/)", "21 unit tests covering all DSP modules, causal convs, STFT, PESQ, STOI, and Whisper WER evaluation."),
        ("Qualcomm AI Hub Verified Assets (results/)", "Real hardware compile/profile jobs jg9zozmlp and jp1nonj2g targeting Snapdragon X Elite."),
    ]
    for i, (dtitle, ddesc) in enumerate(delivs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"✔ {dtitle}"
        p.font.name = "Arial"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = CYAN
        p = tf.add_paragraph()
        p.text = f"   {ddesc}"
        p.font.name = "Arial"
        p.font.size = Pt(11)
        p.font.color.rgb = WHITE
        p.space_after = Pt(6)

    out_pptx = Path("docs/Battlefield_Radio_Intelligibility_Engine_Pitch_Deck.pptx")
    prs.save(str(out_pptx))
    print(f"[+] Pitch deck created successfully at {out_pptx}")


if __name__ == "__main__":
    create_deck()
