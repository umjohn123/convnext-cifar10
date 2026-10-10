/**
 * Build the AIassign2 (AI Assignment #1) report as a Word document.
 *
 * Report requirements covered (01_requirements/report_requirements.txt):
 *   (1) how the AI tools were asked to find the algorithm
 *   (2) algorithm description
 *   (3) how the AI implemented the algorithm
 *   (4) experiment settings and results
 *   (5) what has been learnt from the assignment
 *   (6) a webpage link of the source codes
 */

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, Header, Footer, AlignmentType, HeadingLevel, BorderStyle,
  WidthType, ShadingType, TableOfContents, PageNumber, PageBreak,
  LevelFormat, VerticalAlign,
} = require("docx");

// --------------------------------------------------------------------------- //
// Paths
// --------------------------------------------------------------------------- //
const ROOT = "d:/Document/workbuddy-ai/CISC3024PRAI/Assignment2";
const FIG = path.join(ROOT, "05_results/full_convnext_tiny_cifar10/figures");
const OUT = path.join(ROOT, "06_report/AIassign2_Report_ConvNeXt_CIFAR10.docx");

// --------------------------------------------------------------------------- //
// Layout constants (US Letter, 1 inch margins -> content width 9360 DXA)
// --------------------------------------------------------------------------- //
const CONTENT_W = 9360;
const FONT = "Arial";
const MONO = "Consolas";
const ACCENT = "1F4E79";     // dark blue for headings
const HDR_FILL = "D9E2F3";   // light blue for table headers
const ALT_FILL = "F2F6FC";   // light row shading

const BORDER = { style: BorderStyle.SINGLE, size: 1, color: "BFBFBF" };
const BORDERS = { top: BORDER, bottom: BORDER, left: BORDER, right: BORDER };
const CELL_MARGINS = { top: 60, bottom: 60, left: 110, right: 110 };

// --------------------------------------------------------------------------- //
// Small helpers
// --------------------------------------------------------------------------- //

/** Convert "some **bold** text" into docx TextRun[] (supports **bold** segments). */
function runs(text, base = {}) {
  const out = [];
  const parts = String(text).split("**");
  parts.forEach((part, i) => {
    if (part === "") return;
    out.push(new TextRun({ text: part, bold: i % 2 === 1, ...base }));
  });
  if (out.length === 0) out.push(new TextRun({ text: "", ...base }));
  return out;
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    pageBreakBefore: true,
    children: [new TextRun(text)],
  });
}

function h1NoBreak(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] });
}

function h2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] });
}

function h3(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(text)] });
}

function para(text, opts = {}) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 120, line: 264 },
    children: runs(text),
    ...opts,
  });
}

function bullet(text) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 60, line: 264 },
    children: runs(text),
  });
}

function numbered(text) {
  return new Paragraph({
    numbering: { reference: "numbers", level: 0 },
    alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 60, line: 264 },
    children: runs(text),
  });
}

function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 60, after: 200 },
    children: runs(text, { italics: true, size: 18, color: "595959" }),
  });
}

function tableCaption(text) {
  return new Paragraph({
    alignment: AlignmentType.LEFT,
    spacing: { before: 160, after: 80 },
    children: runs(text, { bold: true, size: 18, color: "333333" }),
  });
}

function code(lines) {
  const arr = Array.isArray(lines) ? lines : [lines];
  return arr.map((line, i) => new Paragraph({
    spacing: { after: i === arr.length - 1 ? 160 : 0, line: 240 },
    shading: { fill: "F5F5F5", type: ShadingType.CLEAR },
    children: [new TextRun({ text: line, font: MONO, size: 17 })],
  }));
}

function cell(text, width, opts = {}) {
  return new TableCell({
    borders: BORDERS,
    width: { size: width, type: WidthType.DXA },
    margins: CELL_MARGINS,
    shading: opts.fill ? { fill: opts.fill, type: ShadingType.CLEAR } : undefined,
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      alignment: opts.align || AlignmentType.LEFT,
      spacing: { after: 0, line: 240 },
      children: runs(text, { size: opts.size || 18, bold: !!opts.bold }),
    })],
  });
}

/** Build a full-width table. `rows[0]` is the header row. */
function table(rows, widths, opts = {}) {
  const bodyFill = opts.zebra === false ? null : ALT_FILL;
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: widths,
    rows: rows.map((row, r) => new TableRow({
      tableHeader: r === 0,
      children: row.map((c, i) => cell(
        c, widths[i],
        r === 0
          ? { fill: HDR_FILL, bold: true, size: 18 }
          : { align: opts.centerData ? AlignmentType.CENTER : AlignmentType.LEFT,
              fill: (r % 2 === 1 && bodyFill) ? bodyFill : undefined },
      )),
    })),
  });
}

function figure(file, widthPx, heightPx, capText) {
  const data = fs.readFileSync(path.join(FIG, file));
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 120, after: 0 },
      children: [new ImageRun({
        type: "png",
        data,
        transformation: { width: widthPx, height: heightPx },
        altText: { title: capText, description: capText, name: file },
      })],
    }),
    caption(capText),
  ];
}

function spacer() {
  return new Paragraph({ spacing: { after: 120 }, children: [new TextRun("")] });
}

// --------------------------------------------------------------------------- //
// Document content
// --------------------------------------------------------------------------- //
const CONTENT = [];

// ============================== TITLE PAGE ================================= //
CONTENT.push(
  new Paragraph({ spacing: { before: 1200, after: 0 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Reimplementing ConvNeXt for CIFAR-10 Image Classification", bold: true, size: 40, color: ACCENT })] }),
  new Paragraph({ spacing: { before: 120, after: 0 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "An AI-Driven and Reproducible Pipeline: Algorithm Search, Implementation, Experiments and Reporting", size: 24, color: "404040" })] }),
  new Paragraph({ spacing: { before: 480, after: 240 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "CISC3024 Pattern Recognition — AI Assignment #1 (AIassign2)", size: 22 })] }),
);

CONTENT.push(table([
  ["Item", "Details"],
  ["Algorithm", "ConvNeXt — *A ConvNet for the 2020s*, **CVPR 2022**, arXiv:2201.03545"],
  ["Algorithm category", "**Deep CNN** (pure convolution; no attention mechanism)"],
  ["Application domain", "Computer vision / pattern recognition — **image classification**"],
  ["Task and dataset", "CIFAR-10 (32x32 RGB, 10 classes, 60,000 images)"],
  ["Final result", "Test **Top-1 = 92.66 %** (99.56 % Top-5); **92.87 %** with test-time augmentation"],
  ["Implementation", "Python 3.14 + PyTorch 2.14.1 (CUDA 13.0); ~2,600 lines of code in 24 source files"],
  ["Compute used", "NVIDIA GeForce RTX 4060 Laptop GPU (8 GB) and CPU (20 logical cores)"],
  ["Report date", "10 October 2026"],
  ["Author", "<student name> (<student ID>) — **to be filled in before submission**"],
], [2200, 7160]));

CONTENT.push(
  new Paragraph({ spacing: { before: 320, after: 80 }, children: [new TextRun({ text: "Declaration", bold: true, size: 22 })] }),
  para("Every line of code, every experiment and every section of this report was produced by AI tools (an AI coding agent and large language models) in accordance with the assignment rules. No code was written by hand. The complete dialogue records that document this process, the archived run artefacts and the reproduction commands are included in the submitted project archive; their locations are listed in Appendix A."),
  new Paragraph({ children: [new PageBreak()] }),
  new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Table of Contents")] }),
  new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-3" }),
);

// ==================== 1. HOW THE ALGORITHM WAS FOUND ======================= //
CONTENT.push(
  h1("1. How I Asked AI Tools to Find the Algorithm"),
  para("The assignment fixes three hard constraints before any modelling decision can be made: the algorithm must belong to the **Deep CNN** or **Deep Autoencoder** family, it must have been published between **2022 and 2026**, and it must be applied to a **computer-vision / pattern-recognition** task. Therefore the AI tools were not asked an open question such as \"which CNN is good?\". They were first asked to convert the assignment text into a checkable constraint list, and only then asked to search for candidates that satisfy every constraint."),

  h2("1.1 Step 1 — turning the assignment into checkable constraints"),
  para("The first prompt asked the AI to extract the mandatory requirements verbatim and to restate them as a list that could later be verified item by item. Working with an explicit list made it possible to discard attractive but non-compliant options early (for example, pure vision transformers, or archives that only provide pre-trained weights without a paper)."),
  tableCaption("Table 1. Constraints extracted from the assignment and used as search filters."),
  table([
    ["#", "Constraint", "How it was checked"],
    ["C1", "Category must be Deep CNN or Deep Autoencoder", "Model is built only from standard convolution / linear / normalisation operators — no self-attention anywhere in the code"],
    ["C2", "Proposed in 2022–2026 with a verifiable source", "Formal publication (CVPR 2022) plus arXiv identifier and open-access link"],
    ["C3", "Applied to CV / pattern recognition", "Image classification on CIFAR-10"],
    ["C4", "All code must be generated by AI", "Every source file is embedded in the AI-generated project and in the generated Colab notebook; dialogue records kept"],
    ["C5", "Report must be in English, Word or PDF", "This document"],
    ["C6", "Report must cover the six required items", "Sections 1–6 of this report"],
    ["C7", "An accessible web link to the source code", "Section 6 (link to be finalised by the author before submission)"],
  ], [700, 3860, 4800]),

  h2("1.2 Step 2 — divergent search for candidates"),
  para("The second prompt asked for a shortlist of Deep-CNN or autoencoder algorithms published between 2022 and 2026, with the paper title, venue, a one-sentence contribution and — most importantly for this project — an explicit assessment of **how feasible each option is on limited hardware** (a CPU-only laptop and a free-tier GPU). Requesting the feasibility column from the start prevented the selection of algorithms whose cost is acceptable only on large clusters."),
  tableCaption("Table 2. Candidate pool returned by the AI and the criteria used for the decision."),
  table([
    ["Candidate", "Year", "Category", "Feasibility on limited hardware", "Decision"],
    ["**ConvNeXt** (*A ConvNet for the 2020s*)", "2022", "Deep CNN", "High — pure PyTorch, no custom CUDA kernels; module boundaries are clean and each design element can be verified independently", "**Selected**"],
    ["ConvNeXt V2 (FCMAE)", "2023", "Deep CNN + self-supervised pretraining", "Medium — an extra masked-autoencoder pretraining stage roughly doubles the training cost", "Rejected (cost)"],
    ["Masked Autoencoders (MAE)", "2022", "Deep Autoencoder", "Low to medium — compliant with the category rule but the pretraining budget is high for a first deliverable", "Rejected (cost)"],
  ], [2100, 620, 1500, 3740, 1400]),

  h2("1.3 Step 3 — converging on ConvNeXt"),
  para("The third prompt explicitly asked the AI to rank the candidates along three axes — **category compliance, implementation clarity and compute affordability** — and to state the adaptation cost for a small-resolution dataset. The reasoning that decided the selection was:"),
  bullet("**Category compliance** — ConvNeXt contains no attention operator at all; it is a pure convolutional network, which is the most literal reading of the \u201CDeep CNN\u201D constraint."),
  bullet("**Implementation clarity** — the architecture decomposes into a stem, four stages and a classifier head, and every design element (inverted bottleneck, 7x7 depthwise convolution, LayerNorm in place of BatchNorm, separate downsampling layers, layer scale) can be implemented and checked one by one, which is exactly what Section 2 of this report requires."),
  bullet("**Compute affordability** — the project defines five model presets (atto, femto, pico, nano, tiny). The smallest one is small enough to run the complete pipeline end-to-end on a CPU in about 19 minutes, while the tiny preset (27.8 M parameters) matches the paper's ConvNeXt-T and can be trained on a single 8 GB laptop GPU."),
  bullet("**Small-resolution adaptation** — the paper's stem is a 4x4 convolution with stride 4 because ImageNet images are 224x224. Applying it to 32x32 CIFAR-10 images would shrink the feature maps to 8 -> 4 -> 2 -> 1 and destroy most of the spatial information. The AI therefore proposed replacing the stem with a 3x3 convolution of stride 1, giving feature-map sizes of 32 -> 16 -> 8 -> 4 while keeping every other design element of the paper intact (Section 2.5)."),

  h2("1.4 Step 4 — verifying the provenance of the algorithm"),
  para("Because the assignment demands a verifiable source, a dedicated prompt asked for the full bibliographic record, the licence of the official implementation, and an explicit statement of how the algorithm satisfies the category and year constraints."),
  tableCaption("Table 3. Verifiable provenance of the selected algorithm."),
  table([
    ["Item", "Value"],
    ["Paper", "*A ConvNet for the 2020s* — Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie"],
    ["Venue / year", "**CVPR 2022**, pp. 11976–11986 (IEEE/CVF Conference on Computer Vision and Pattern Recognition)"],
    ["Preprint", "arXiv:2201.03545 (first released 10 January 2022), https://arxiv.org/abs/2201.03545"],
    ["Open access", "https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html"],
    ["Official code", "https://github.com/facebookresearch/ConvNeXt (MIT License)"],
    ["Category check", "Built exclusively from convolutions, linears and normalisation layers -> Deep CNN (satisfies C1)"],
    ["Year check", "Published 2022, inside the required 2022–2026 window (satisfies C2)"],
  ], [1900, 7460]),

  h2("1.5 How the prompts were structured"),
  para("Every search round used the same three-part structure, which later made the dialogue records directly quotable in this report: **(i) ask a constrained question, (ii) ask the AI to justify its answer with checkable evidence, and (iii) ask for the result to be written to a file in the project archive.** The five rounds are archived in the project folder `02_ai_chats/` (algorithm search dialogue, candidate list, algorithm profile, iteration summary). Two habits proved decisive: asking the AI to argue *against* its own proposal (which is how the 4x4 stride-4 stem problem surfaced), and requiring every claim about the literature to come with a link or an identifier that can be opened and verified."),
);

// ======================== 2. ALGORITHM DESCRIPTION ========================= //
CONTENT.push(
  h1("2. Algorithm Description"),
  para("This section describes ConvNeXt as published, then states precisely how it was adapted to 32x32 inputs and which training recipe was used."),

  h2("2.1 Motivation and central idea"),
  para("Vision transformers replaced convolutional networks as the dominant backbone after 2020, and much of that shift was attributed to the self-attention operator. ConvNeXt starts from the opposite hypothesis: the performance gap between transformers and ConvNets is caused mainly by **training techniques and architectural design choices that convolutional networks had not adopted**, not by attention itself. The authors therefore take a standard ResNet-50 and modernise it step by step, borrowing design elements from transformers (and from Swin Transformer in particular), evaluating the effect of every single change. The result is a family of pure convolutional networks whose accuracy is on par with — and in detection and segmentation slightly better than — comparable Swin transformers."),

  h2("2.2 The modernisation roadmap"),
  para("The roadmap is the core contribution of the paper: eight design decisions, each introduced separately and each measurable in ImageNet top-1 accuracy (ResNet-50 baseline = 76.1 %)."),
  tableCaption("Table 4. Design elements of the roadmap and what ConvNeXt adopts."),
  table([
    ["#", "Design element", "ResNet-50 baseline", "ConvNeXt choice", "Rationale"],
    ["1", "Macro design (stage ratio)", "(3, 4, 6, 3)", "**(3, 3, 9, 3)**", "Match the 1:1:3:1 stage ratio of Swin-T; more capacity in the third stage"],
    ["2", "Stem", "7x7 conv, stride 2 + max-pool", "**4x4 conv, stride 4 (patchify)**", "Non-overlapping patches, as in ViT (adapted to 3x3 stride 1 for CIFAR-10, Section 2.5)"],
    ["3", "ResNeXt-ify", "3x3 conv, width 64", "**Grouped / depthwise convolution**", "Depthwise convolution with a larger width achieves comparable accuracy at lower cost"],
    ["4", "Inverted bottleneck", "Bottleneck: wide -> narrow -> wide", "**1x1 expand x4 -> depthwise -> 1x1 project**", "Follows the transformer block, where the MLP is four times wider than the input"],
    ["5", "Large kernel", "3x3 convolutions", "**7x7 depthwise convolution**", "Enlarges the effective receptive field, analogous to self-attention's global mixing"],
    ["6", "Micro design — activation", "ReLU", "**GELU**, and only one activation per block", "Matches the transformer MLP; fewer activations reduce latency and improve accuracy"],
    ["7", "Micro design — normalisation", "BatchNorm", "**LayerNorm**", "BatchNorm is unnecessary for training stability in the modernised design and harms transfer"],
    ["8", "Downsampling", "stride-2 in the first block of each stage", "**Separate 2x2 stride-2 convolution between stages**", "Decouples spatial reduction from the residual path"],
  ], [420, 1900, 1800, 2240, 3000]),

  h2("2.3 The ConvNeXt block"),
  para("Every stage is a stack of identical blocks. In the final design a block is: a 7x7 depthwise convolution, a LayerNorm, a 1x1 convolution that expands the channel width by a factor of four, a GELU activation, a 1x1 convolution that projects back to the original width, a learnable per-channel rescaling (layer scale), a stochastic-depth (drop-path) regulariser, and a residual connection. Formally, with input x, channel dimension C, expansion factor 4 and layer-scale parameter gamma in R^C:"),
  ...code([
    "y = Linear_2( GELU( Linear_1( LayerNorm( DWConv7x7(x) ) ) ) )      # C -> 4C -> C",
    "z = gamma * y                                                        # layer scale, gamma initialised to 1e-6",
    "out = x + DropPath(z)                                                # residual connection",
  ]),
  para("Two details matter in practice. First, the depthwise convolution is the only operator with a large receptive field, which is why the 7x7 kernel (rather than 3x3) is essential. Second, **layer scale** multiplies the residual branch by a learnable vector initialised to 1e-6, so that at the beginning of training every block behaves approximately as the identity mapping; this stabilises the optimisation of deep, normalisation-light networks. The second detail becomes important again in Section 4.6, where it is shown to interact badly with 16-bit floating point."),

  h2("2.4 Stage layout, downsampling and classifier head"),
  para("The network contains four stages. Between stage i and stage i+1 a dedicated downsampling layer applies a LayerNorm followed by a 2x2 stride-2 convolution. The final feature map is normalised with a GroupNorm that has a single group (equivalent to a per-sample, channel-first LayerNorm), globally average-pooled, and classified by one linear layer. The five presets used in this project (the tiny preset is the paper's ConvNeXt-T) are:"),
  tableCaption("Table 5. Model presets defined in the project (parameter counts measured on the implemented model)."),
  table([
    ["Preset", "Depths", "Dims", "Parameters", "Role in this project"],
    ["atto", "(2, 2, 6, 2)", "(40, 80, 160, 320)", "3.38 M", "CPU end-to-end MVP run"],
    ["femto", "(2, 2, 6, 2)", "(48, 96, 192, 384)", "5.2 M", "Reserved for ablations"],
    ["pico", "(2, 2, 6, 2)", "(64, 128, 256, 512)", "9.0 M", "Fast GPU experiments"],
    ["nano", "(2, 2, 8, 2)", "(80, 160, 320, 640)", "15.6 M", "Reserved for ablations"],
    ["**tiny**", "**(3, 3, 9, 3)**", "**(96, 192, 384, 768)**", "**27.83 M**", "**Main experiment (paper's ConvNeXt-T)**"],
  ], [900, 1500, 2000, 1200, 3760]),

  h2("2.5 Adaptation to 32x32 inputs (the only structural change)"),
  para("ImageNet images are 224x224, so the paper's 4x4 stride-4 stem is appropriate: the feature maps evolve as 56 -> 28 -> 14 -> 7. CIFAR-10 images are 32x32, and the same stem would produce 8 -> 4 -> 2 -> 1 feature maps, leaving 1x1 spatial maps in the last stage and discarding most of the spatial structure. The implementation therefore uses a **3x3 convolution with stride 1** as the stem, which yields 32 -> 16 -> 8 -> 4 in the four stages. All other elements of the paper are preserved unchanged: the (3, 3, 9, 3) stage ratio, the 4x expansion inverted bottleneck, the 7x7 depthwise convolution, GELU, LayerNorm, the separate downsampling layers, layer scale and stochastic depth. The stem is switchable at run time (`--stem-mode conv|patchify`), so the original paper configuration remains available."),

  h2("2.6 Training recipe and regularisation"),
  para("The recipe follows the paper's training practice, adapted to a small dataset: AdamW with a base learning rate of 1e-3 and weight decay 0.05, where weights of one-dimensional parameters (normalisation scales and offsets, all biases) are excluded from weight decay; a linear warm-up over five epochs followed by cosine decay to 1e-5; label smoothing 0.1; Mixup and CutMix applied to 50 % of the batches with alpha values 0.8 and 1.0; RandomResizedCrop-free augmentation suited to 32x32 images (random crop with 4-pixel reflect padding, horizontal flip, random erasing with probability 0.25); stochastic depth with a maximum rate of 0.1, growing linearly with block index; and layer scale initialised to 1e-6."),
);

// ==================== 3. HOW THE AI IMPLEMENTED IT ========================= //
CONTENT.push(
  h1("3. How the AI Implemented the Algorithm"),
  para("The implementation was produced by an AI coding agent under a simple protocol: **the agent writes all code, the human only states requirements, runs the produced commands and inspects the output.** Nothing in the project is hand-written. This section summarises what was generated, the engineering decisions taken, and the verification steps that were used to decide whether the generated code could be trusted."),

  h2("3.1 Working protocol"),
  numbered("Describe the deliverable and its acceptance criteria in plain language, including the hardware constraints."),
  numbered("Let the AI generate the complete project (not a single script), with a documented command-line interface and deterministic output paths."),
  numbered("Run the generated commands; keep the raw logs, metrics and figures as evidence."),
  numbered("Ask the AI to analyse the produced artefacts and to justify any follow-up change with measurements rather than opinion."),
  numbered("Record the dialogue, so that every change remains traceable to a request."),

  h2("3.2 Generated project structure"),
  tableCaption("Table 6. Structure of the generated code base (24 Python files plus a self-contained Colab notebook)."),
  table([
    ["Path", "Responsibility"],
    ["`configs.py`", "Single source of truth for data / model / training hyper-parameters, dataclass-based, fully overridable from the command line, serialised to JSON for reproducibility"],
    ["`train.py`, `evaluate.py`", "Entry points: train-validation-test pipeline with artefact export; standalone evaluation of a saved checkpoint (with optional test-time augmentation)"],
    ["`src/models/convnext.py`", "ConvNeXt: layer-scale block, depthwise convolution, LayerNorm (both layouts), separate downsampling, five presets, parameter summary"],
    ["`src/engine/trainer.py`", "Training loop: AdamW with parameter grouping, warm-up + cosine schedule, gradient accumulation, gradient clipping, AMP with an automatic safety net, checkpointing, resume, per-epoch metric export"],
    ["`src/engine/evaluator.py`", "Top-1 / Top-5 accuracy, per-class accuracy, confusion matrix, test-time augmentation with identical protocol for all artefacts"],
    ["`src/data/`", "CIFAR-10 loading with a single base dataset per split, deterministic split protocol, synthetic dataset for zero-download smoke tests"],
    ["`src/utils/`", "Device selection and memory-adaptive batch-size probe, AMP compatibility layer, mixup / cutmix, CSV export, logging, seeding, visualisation, TTA"],
    ["`notebooks/`", "Generator that writes a self-contained Colab notebook from the sources, plus a verbatim self-check that fails if notebook and sources ever diverge"],
    ["`scripts/fetch_cifar10.py`", "Mirror probing, multi-connection chunked download, MD5 verification and extraction for the 170 MB CIFAR-10 archive"],
  ], [2200, 7160]),

  h2("3.3 Engineering decisions worth highlighting"),
  bullet("**One base dataset per split.** `torchvision.datasets.CIFAR10` re-validates the MD5 checksums of the 186 MB archive every time it is constructed. Building the dataset once per split and layering the different augmentation pipelines through a `TransformSubset` wrapper reduced the hot path from 4.5 s to 1.16 s while keeping the index semantics identical."),
  bullet("**Memory-adaptive batch size.** Before training, the code runs a real forward plus backward plus optimiser step on the target device and adjusts the batch size until it fits, logging the measured peak. If the batch size is reduced, gradient accumulation compensates so that the effective batch stays close to the requested one. Section 4.6 describes the additional safety margin that proved necessary on Windows."),
  bullet("**A compatibility and safety layer for mixed precision.** The project supports both the pre-2.4 and current PyTorch AMP APIs, prefers bfloat16 whenever the device supports it natively, records the dynamic loss-scale of the gradient scaler for every epoch, and can fall back to full precision automatically."),
  bullet("**Experiment isolation.** Every run is identified by a tag (`--run-tag`); all artefacts land in `05_results/<tag>/{figures,logs,models,metrics}`, so parallel or repeated experiments cannot overwrite or cross-reference each other."),
  bullet("**Incremental persistence.** Training history is written to disk after every epoch, and both the best and the last checkpoint contain model, optimiser, scheduler, scaler state, history and the full configuration. This is what made the mid-training interruption described in Section 4.4 recoverable without losing a single epoch."),

  h2("3.4 Verification performed on the generated code"),
  tableCaption("Table 7. Checks applied to the AI-generated implementation."),
  table([
    ["Check", "How it was performed", "Result"],
    ["Smoke test without any download", "Train the smallest preset for 2 epochs on locally generated synthetic data", "Pipeline runs end to end in ~20 s"],
    ["End-to-end run on real data (CPU)", "atto preset, 4,500 training images, 10 epochs", "Test Top-1 45.05 % — clearly above the 10 % chance level, so the pipeline genuinely learns"],
    ["Determinism of the split protocol", "Re-computed the training / validation / test index sets and compared them with the archived `04_data/splits/*.pt` files", "Identical index sets"],
    ["Notebook-versus-source consistency", "The notebook generator re-reads the sources and compares them verbatim with the embedded copies", "Passes (24/24 files); the check exits with a non-zero code otherwise"],
    ["Numerical-health monitoring", "Every epoch logs the gradient-scaler loss scale; checkpoint files contain optimiser and scaler states that can be inspected afterwards", "This is what exposed the failure analysed in Section 4.6"],
  ], [2400, 4460, 2500]),

  h2("3.5 Three engineering problems that only appeared on real hardware"),
  para("The pipeline was first validated on a CPU and then moved to a GPU, where three problems appeared that no amount of code reading had revealed. All three were diagnosed from the artefacts of the runs themselves and then fixed in the code, and they are reported here because they are part of the work the assignment asks to document."),
  tableCaption("Table 8. Problems found by running the pipeline, and the fixes applied to the generated code."),
  table([
    ["Problem", "Symptom", "Diagnosis", "Fix"],
    ["**16-bit overflow of the loss scale** (T4 GPU, fp16)", "60 epochs completed without error, but the loss never moved and the test accuracy stayed at 25.73 %", "The checkpoint contained `scaler_state.scale = 0.0156` (default 65536) and Adam's second-moment buffers of the early blocks were exactly zero, i.e. those weights had never been updated", "Prefer bfloat16 (same exponent range as fp32); log the loss scale every epoch; abort to fp32 automatically if the scale collapses"],
    ["**Over-committed GPU memory** (Windows/WDDM)", "The batch-size probe reported a peak of 10.8 GB on an 8 GB card, after which the real training failed immediately with an out-of-memory or a `CUDA driver error`", "Windows allows CUDA to allocate beyond physical video memory, so the probe can \u201Cpass\u201D a batch size that does not really fit", "Require the probe peak to stay below 70 % of the free memory, reset peak statistics before every probe, and treat driver-side memory errors as \u201Cdoes not fit\u201D (batch 512 -> 128 on the 8 GB laptop)"],
    ["**Commit-charge exhaustion in the multi-process data loader** (Windows)", "The process died after about two hours with `Couldn't open shared file mapping ... error code: 1455`", "PyTorch's multi-process loader moves every batch through a shared file mapping; on Windows this slowly consumes system commit charge", "Run long jobs with a single-process loader, and resume from the last checkpoint when a run is interrupted"],
  ], [2000, 2300, 2600, 2460]),

  h2("3.6 Reproducibility measures"),
  bullet("Fixed seeds for Python, NumPy, PyTorch and the loader workers; the training / validation split is derived from a fixed-seed permutation and archived as index files."),
  bullet("Every run writes its complete hyper-parameter set to `metrics/config.json`, which is also embedded in the checkpoints."),
  bullet("Per-epoch metrics are exported both as JSON and as a CSV with a fixed column order, so results of different experiments can be stacked into a single comparison table."),
  bullet("The Colab notebook is generated from the sources and self-checked, which guarantees that the notebook and the repository cannot drift apart."),
);

// ================== 4. EXPERIMENT SETTINGS AND RESULTS ===================== //
CONTENT.push(
  h1("4. Experiment Settings and Results"),
  para("Three runs are reported: a CPU end-to-end MVP that validates the pipeline, the main full-scale GPU run, and the failed fp16 run that is used as a controlled comparison for the effect of numerical precision."),

  h2("4.1 Data"),
  para("CIFAR-10 consists of 60,000 32x32 RGB images in ten classes (6,000 per class). The official split was respected exactly: the 50,000 training images were shuffled with a fixed seed (42) and divided into 45,000 training and 5,000 validation images, while the official 10,000-image test set was never used for training or for model selection. Augmentation was applied only to the training split; validation and test images received a deterministic transform (tensor conversion and normalisation with the CIFAR-10 channel statistics (0.4914, 0.4822, 0.4465) and (0.2470, 0.2435, 0.2616))."),

  h2("4.2 Settings of the main experiment"),
  tableCaption("Table 9. Configuration of the full-scale run (`full_convnext_tiny_cifar10`); the same values are archived in `metrics/config.json`."),
  table([
    ["Item", "Value"],
    ["Model", "ConvNeXt-Tiny — depths (3, 3, 9, 3), widths (96, 192, 384, 768), 27,825,802 parameters"],
    ["Stem", "3x3 convolution, stride 1 (32x32 adaptation); stage resolutions 32 -> 16 -> 8 -> 4"],
    ["Input / classes", "32x32 RGB / 10"],
    ["Data split", "45,000 train / 5,000 validation / 10,000 test (official test set held out)"],
    ["Optimiser", "AdamW, learning rate 1e-3, weight decay 0.05, one-dimensional parameters exempt from decay, betas (0.9, 0.999)"],
    ["Schedule", "5 epochs linear warm-up, then cosine decay to 1e-5, 60 epochs total"],
    ["Batch size", "128 (effective batch 128, no gradient accumulation)"],
    ["Augmentation", "Random crop 32 with 4-pixel reflect padding; horizontal flip; random erasing p = 0.25; Mixup / CutMix on 50 % of batches (alpha 0.8 / 1.0)"],
    ["Regularisation", "Label smoothing 0.1; stochastic depth 0 -> 0.1 over depth; layer-scale initialised at 1e-6; gradient clipping at norm 1.0"],
    ["Mixed precision", "bfloat16 autocast with gradient scaling; minimum observed loss scale 2.10e6; no fallback to fp32 was triggered"],
    ["Hardware / software", "NVIDIA RTX 4060 Laptop GPU (8 GB), PyTorch 2.14.1+cu130, CUDA 13.0, Python 3.14; data loading with a single process (see Section 3.5)"],
    ["Seed / wall-clock time", "42 / 1 h 55 min for the resumed segment (about 3 h 05 min including the interrupted segment)"],
  ], [2100, 7260]),

  h2("4.3 Main results"),
  para("The checkpoint with the best validation accuracy (epoch 59, 93.50 %) was evaluated on the held-out test set twice: once with a single view and once with test-time augmentation (the average of the identity and horizontally flipped views)."),
  tableCaption("Table 10. Test-set results of the full-scale run."),
  table([
    ["Metric", "Single view", "With test-time augmentation"],
    ["Test cross-entropy loss (label smoothing 0.1)", "0.6842", "0.6842 (loss is computed without TTA)"],
    ["**Top-1 accuracy**", "**92.66 %**", "**92.87 %**"],
    ["**Top-5 accuracy**", "**99.56 %**", "**99.61 %**"],
    ["Macro-average accuracy", "92.67 %", "92.87 %"],
    ["Correct / total", "9,266 / 10,000", "9,287 / 10,000"],
    ["Best validation accuracy (model selection)", "**93.50 %** (epoch 59)", "—"],
  ], [3760, 2800, 2800]),
  para("For reference, random guessing on ten classes gives 10 % Top-1 and 50 % Top-5, so the trained model is 82.9 and 49.6 percentage points better than chance. Test-time augmentation adds **+0.21 percentage points**, which matches the +0.2 to +0.5 range documented for this technique in the project's own design notes."),

  h2("4.4 Training behaviour"),
  para("Training followed the expected two-phase behaviour: a fast phase in which the validation accuracy rose from 43.0 % (epoch 1) to 90.7 % (epoch 30), and a slow refinement phase that continued to 93.50 % at epoch 59. The validation loss decreased monotonically to 0.6697, and it never diverged from the training loss, which indicates that no overfitting occurred — a consequence of Mixup/CutMix, label smoothing, stochastic depth and strong augmentation. The learning rate followed the warm-up and cosine schedule exactly (1e-3 -> 1.08e-5)."),
  para("One operational incident is worth recording because it is visible in the logs: after epoch 32 the process was terminated by a Windows-specific memory error in the multi-process data loader (Section 3.5, third row). The run was resumed from the last checkpoint with `--resume auto`, which restored the model, the optimiser, the learning-rate schedule and the full history; training continued from epoch 33 and the accuracy curve crossed the interruption smoothly (90.70 % -> 91.44 %). The only visible trace is a change of the per-epoch time from about 120 s to about 230 s, caused by the change of the data-loader worker count. No epoch was lost."),

  h2("4.5 Per-class results and error analysis"),
  tableCaption("Table 11. Per-class test accuracy (1,000 images per class)."),
  table([
    ["Class", "Top-1 (TTA)", "Top-1 (single view)", "Class", "Top-1 (TTA)", "Top-1 (single view)"],
    ["airplane", "93.80 %", "93.50 %", "dog", "88.20 %", "87.90 %"],
    ["automobile", "**97.90 %**", "98.20 %", "frog", "96.20 %", "95.90 %"],
    ["bird", "90.90 %", "90.20 %", "horse", "94.20 %", "94.60 %"],
    ["cat", "**82.80 %**", "82.10 %", "ship", "96.50 %", "96.30 %"],
    ["deer", "93.40 %", "92.90 %", "truck", "94.80 %", "95.10 %"],
  ], [1400, 1800, 1900, 1400, 1800, 1060]),
  para("The confusion matrix shows a clear structure. The only class pair that is confused in both directions by more than 8 % is **cat <-> dog** (8.2 % of the cats are predicted as dogs and 8.1 % of the dogs as cats), which is the classic fine-grained problem of similar fur texture and pose. Beyond that pair, the dominant errors are truck -> automobile (2.9 %), airplane -> ship (2.2 %), bird -> frog (2.2 %) and deer -> frog (2.0 %). The classes with the highest accuracy — automobile, ship, frog and truck — are those with distinctive global shape and background, which supports the interpretation that the remaining errors are concentrated in shape-and-texture-similar quadruped and vehicle pairs rather than being spread randomly."),

  ...figure("training_curves.png", 620, 180, "Figure 1. Training curves: (a) cross-entropy loss, (b) top-1 accuracy with the best epoch marked, (c) the warm-up plus cosine learning-rate schedule."),
  ...figure("confusion_matrix.png", 470, 409, "Figure 2. Row-normalised confusion matrix on the test set (TTA protocol). The diagonal gives the per-class recall; the dominant off-diagonal mass is the cat/dog pair."),
  ...figure("per_class_accuracy.png", 520, 270, "Figure 3. Per-class test accuracy (TTA). The macro average is 92.87 %."),
  ...figure("data_samples.png", 620, 169, "Figure 4. Sample images from the CIFAR-10 validation split with their labels, used to verify the download, the normalisation statistics and the augmentation pipeline."),

  h2("4.6 Controlled comparison: the effect of numerical precision"),
  para("The first full-scale attempt was executed on a free GPU that supports only fp16 (not bfloat16). It completed all 60 epochs **without a single error message**, yet the test accuracy was 25.73 % and the training loss never moved below 2.15. Because the artefacts of the run are complete, the failure could be diagnosed exactly: the gradient scaler's dynamic loss scale had collapsed to 0.0156 (default 65536), and the Adam second-moment buffers of the first two stages were exactly zero — those weights had never received a gradient. The mechanism is that layer scale initialises the residual branches with a factor of 1e-6; the resulting gradients in the shallow blocks are around 1e-6, and once the loss scale has collapsed, multiplying them by 0.0156 produces values below the smallest representable fp16 magnitude, which are flushed to zero. The effective network then reduces to the stem, the normalisation layer and the classifier head — a shallow linear model, which explains the 25 % plateau. Re-running the identical recipe with bfloat16 (whose exponent range equals that of fp32) removes the problem entirely."),
  tableCaption("Table 12. The same recipe under two numerical precisions."),
  table([
    ["Aspect", "fp16 on a free GPU (failed run)", "bfloat16 on the local GPU (main result)"],
    ["Gradient-scaler loss scale", "collapsed to 0.0156", "grew from 6.55e4 to 6.71e7"],
    ["Shallow-block parameters", "second-moment buffers exactly zero — never updated", "updated normally"],
    ["Training loss over 60 epochs", "2.15 – 2.27 (ln 10 = 2.303)", "2.03 -> 0.87"],
    ["Best validation accuracy", "26.66 % (epoch 2)", "**93.50 %** (epoch 59)"],
    ["Test Top-1", "25.73 % (invalid)", "**92.66 %** (92.87 % with TTA)"],
  ], [2500, 3430, 3430]),
  para("The comparison is also a methodological lesson: a run that terminates without errors is not automatically a valid run. The project therefore now records the loss scale in every epoch, stores the precision metadata inside the results files, and prints an explicit warning whenever the scale collapses. In fact the code goes one step further and switches to full precision automatically if that happens, and the FP32 validation run (Table 13) confirms that the recipe itself is sound."),

  h2("4.7 CPU baseline for pipeline validation"),
  tableCaption("Table 13. Small-scale CPU run used to validate the complete pipeline before scaling up."),
  table([
    ["Item", "Value"],
    ["Experiment tag", "`mvp_atto_subset5k`"],
    ["Model / data", "ConvNeXt-atto (3.38 M parameters) / 4,500 training, 500 validation, 2,000 test images"],
    ["Training", "AdamW, lr 2e-3, 10 epochs, batch 128, no Mixup, CPU (20 logical cores), 18 min 41 s"],
    ["Result", "**Test Top-1 45.05 %**, Top-5 89.95 %, best validation 45.20 %"],
    ["Purpose", "Confirms that data loading, augmentation, optimisation, evaluation and plotting all work on real data before spending GPU time"],
  ], [2100, 7260]),

  h2("4.8 Discussion and limitations"),
  bullet("**Accuracy level.** Test Top-1 of 92.66 % (92.87 % with TTA) is inside the 93–95 % band that the project's design notes predicted for this recipe on CIFAR-10, at its lower end. The main reason is the limited training budget rather than the architecture: the validation accuracy was still improving at epoch 59 (92.9 % at epoch 50 -> 93.5 % at epoch 59), so the model had not saturated."),
  bullet("**Hardware-induced choices.** With 8 GB of video memory the batch size is 128; larger batches typically improve the accuracy of modernised ConvNets slightly. The evaluation therefore reflects a single-seed, single-configuration result rather than a tuned optimum."),
  bullet("**Strengths observed.** The model is robust: the gap between validation (93.50 %) and test (92.66 %) is only 0.84 percentage points, and no overfitting appears in 60 epochs despite 27.8 M parameters."),
  bullet("**Suggested follow-ups, in decreasing order of expected benefit.** (1) Train longer — 120 to 300 epochs — since the curve had not saturated. (2) Repeat with three seeds to report mean and standard deviation. (3) Add RandAugment, which the code already supports through a configuration switch. (4) Run the structural ablations that the project is already set up for (patchify stem versus 3x3 stem, atto/pico/tiny presets). (5) Compare against a transformer baseline under identical settings to test the paper's central claim on this dataset."),
);

// ============================== 5. WHAT I LEARNT =========================== //
CONTENT.push(
  h1("5. What I Have Learnt from This AI Assignment"),

  h2("5.1 An AI agent can own the entire pipeline, but verification remains the human contribution"),
  para("The agent wrote the model, the training loop, the evaluation, the plotting code, the Colab notebook generator, the download accelerator and the documentation — about 2,600 lines across 24 files — and none of it was hand-written. What the AI could not do by itself was decide whether the produced artefacts were trustworthy. The most valuable contributions I made were procedural: demanding that every run writes its configuration and per-epoch metrics to disk, that the same evaluation protocol is used for all reported numbers, and that a claim is only accepted once it is backed by a measurement. The fp16 incident is the clearest illustration: the code was correct and the logs contained no error, yet the result was worthless, and only the archived checkpoint made that visible."),

  h2("5.2 Numerical precision is an engineering problem, not a detail"),
  para("I had assumed that switching on mixed precision is a pure speed optimisation. In practice, 16-bit arithmetic can silently destroy a training run: because layer scale multiplies each residual branch by 1e-6, the shallow blocks have very small gradients, and when the dynamic loss scale of the gradient scaler collapsed, those gradients underflowed to exactly zero. The lesson generalises: whenever a network contains very small multiplicative constants, the safe default is bfloat16 (or full precision), and the loss scale should be logged and monitored rather than assumed healthy."),

  h2("5.3 Platform behaviour belongs to the experiment"),
  para("Two of the three problems encountered were specific to Windows. The operating system allows GPU memory to be over-committed, so a memory probe cannot be trusted unless it keeps a safety margin; and PyTorch's multi-process data loader slowly consumes system commit charge until the process dies after a couple of hours. Neither is a bug in the algorithm, yet both would have distorted the results if they had gone unnoticed. The practical habits I took away are to fix the environment explicitly in the report, to keep long runs resumable, and to treat sudden changes in per-epoch time as a signal worth investigating."),

  h2("5.4 Reproducibility is a design decision taken in advance"),
  para("The features that made the diagnosis possible — an experiment tag per run, a complete hyper-parameter snapshot, per-epoch history written after every epoch, checkpoints that contain the optimiser and scaler state, and metrics exported in a fixed column order — were all specified before the experiments were run. Adding them afterwards would have been impossible, because the evidence would already have been lost. I now consider this kind of instrumentation a first-class part of an experiment, not an optional extra."),

  h2("5.5 How to prompt an AI agent effectively"),
  bullet("**Convert the specification into verifiable constraints first.** Asking for a checklist before asking for a solution prevented the selection of non-compliant or infeasible algorithms."),
  bullet("**Ask for justification with checkable evidence.** Requiring paper identifiers, links and licences made the literature claims verifiable instead of merely plausible."),
  bullet("**Ask the AI to argue against its own proposal.** This is how the small-resolution stem problem was discovered, which turned out to be the single most important architectural adaptation."),
  bullet("**Ask for artefacts, not opinions.** Requests phrased as \u201Cwrite the config and the per-epoch metrics to disk\u201D produced evidence; requests phrased as \u201Cis it working?\u201D produced prose."),
  bullet("**Let the AI analyse the artefacts it produced.** The two decisive diagnoses in this project (the collapsed loss scale and the never-updated parameters) were obtained by asking the agent to inspect its own checkpoints, which it could do far more quickly and systematically than I could."),

  h2("5.6 What I would do differently next time"),
  bullet("Log the loss scale and mixed-precision metadata from the very first run instead of adding them after a failure (they are now part of the code)."),
  bullet("Validate on a small real-data subset before launching any multi-hour job — the CPU MVP run cost 19 minutes and would have caught environment problems immediately."),
  bullet("Prefer bfloat16-capable hardware, or simply disable mixed precision, when the network contains very small multiplicative constants."),
  bullet("Plan the comparison experiments (baselines, ablations, multiple seeds) from the start, so that the report can discuss them rather than only report a single configuration."),
);

// ============================ 6. SOURCE CODE LINK ========================== //
CONTENT.push(
  h1("6. Source Code Webpage Link"),
  para("The complete source code of this project is published at:"),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 120 },
    children: [new TextRun({ text: "[ Repository URL — to be filled in by the author before submission, e.g. https://github.com/<account>/convnext-cifar10 ]", bold: true, size: 22, color: "C00000" })],
  }),
  para("**Note on this item.** The assignment requires an accessible web link. Publishing the repository is the one step in the workflow that cannot be performed by the AI agent, so the link above is a placeholder that must be replaced by the author before submission; the complete project is meanwhile archived in the submitted folder structure described in Appendix A, from which the repository can be published without modification (`03_code/` together with `04_data/dataset_info.md` and the README files of `05_results/`)."),

  h2("6.1 What the repository contains"),
  bullet("`03_code/` — the complete PyTorch project (model, training, evaluation, utilities, downloader) plus the self-checking Colab notebook generator."),
  bullet("`04_data/` — the dataset description, the split index files and the download instructions (the 170 MB archive itself is not committed)."),
  bullet("`05_results/` — every experiment with its figures, logs, metrics and checkpoints, including the failed run and the diagnosis of it."),
  bullet("`06_report/` — this report, in Word and PDF format."),
  bullet("`02_ai_chats/` — the dialogue records and the algorithm profile that document how the AI was used."),

  h2("6.2 How to run the code"),
  ...code([
    "# 1) environment (PyPI's Windows wheels are CPU-only; use the CUDA index for GPU work)",
    "pip install --upgrade --extra-index-url https://download.pytorch.org/whl/cu130 \\",
    "            \"torch==2.14.1+cu130\" \"torchvision==0.29.1+cu130\"",
    "pip install -r 03_code/requirements.txt",
    "",
    "# 2) full-scale training (ConvNeXt-Tiny, 60 epochs, bfloat16, one GPU)",
    "cd 03_code",
    "python train.py --run-tag full_convnext_tiny_cifar10 --model tiny --epochs 60 \\",
    "    --batch-size 128 --stem-mode conv --device cuda --num-workers 0 \\",
    "    --data-root ../04_data/cifar10 --output-dir ../05_results",
    "",
    "# 3) evaluation with test-time augmentation",
    "python evaluate.py --ckpt ../05_results/full_convnext_tiny_cifar10/models/best.pt \\",
    "    --device cuda --tta --data-root ../04_data/cifar10 --output-dir ../05_results",
  ]),
  para("The repository also contains a self-contained Colab notebook (`03_code/notebooks/ConvNeXt_CIFAR10_Colab.ipynb`) which recreates the whole project inside Colab, downloads CIFAR-10, runs the training and displays the figures, so the experiments can be reproduced without any local setup."),
);

// ============================== REFERENCES ================================= //
CONTENT.push(
  h1("References"),
  numbered("Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell and S. Xie, \u201CA ConvNet for the 2020s\u201D, in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2022, pp. 11976–11986. arXiv:2201.03545."),
  numbered("Official ConvNeXt implementation, Facebook Research (MIT License): https://github.com/facebookresearch/ConvNeXt"),
  numbered("A. Krizhevsky, \u201CLearning multiple layers of features from tiny images\u201D, technical report, University of Toronto, 2009 (CIFAR-10 dataset)."),
  numbered("H. Zhang, M. Cisse, Y. N. Dauphin and D. Lopez-Paz, \u201Cmixup: Beyond empirical risk minimization\u201D, ICLR 2018 (Mixup)."),
  numbered("S. Yun, D. Han, S. Chun, S. Oh, Y. Yoo and J. Choe, \u201CCutMix: Regularization strategy to train strong classifiers with localizable features\u201D, ICCV 2019 (CutMix)."),
  numbered("G. Huang, Y. Sun, Z. Liu, D. Sedra and K. Weinberger, \u201CDeep networks with stochastic depth\u201D, ECCV 2016 (DropPath)."),
  numbered("I. Loshchilov and F. Hutter, \u201CDecoupled weight decay regularization\u201D, ICLR 2019 (AdamW)."),
  numbered("PyTorch documentation — automatic mixed precision and CUDA memory management: https://docs.pytorch.org/docs/stable/notes/cuda.html"),
);

// =============================== APPENDICES ================================ //
CONTENT.push(
  h1("Appendix A — Project Artefacts Used by This Report"),
  para("The submitted archive is organised in seven folders. The table below lists the artefacts that this report refers to and where they can be found."),
  table([
    ["Folder", "Key contents", "Used in"],
    ["`01_requirements/`", "Assignment statement, requirement summary, task breakdown", "Section 1"],
    ["`02_ai_chats/`", "Algorithm search dialogue (5 rounds), candidate list, algorithm profile, iteration summary, screenshots", "Sections 1, 2, 5"],
    ["`03_code/`", "24 Python source files, Colab notebook generator, download accelerator, requirements", "Section 3"],
    ["`04_data/`", "Dataset description, split index files, raw archive", "Section 4.1"],
    ["`05_results/full_convnext_tiny_cifar10/`", "4 figures, training and evaluation logs, history CSV/JSON, 2 checkpoints, 5 metric files, 5 index documents", "Section 4 and Figures 1–4"],
    ["`05_results/_failed_runs/`", "The fp16 run: its metrics, its diagnosis and the evidence (loss-scale and optimiser state)", "Sections 3.5 and 4.6"],
    ["`05_results/mvp_atto_subset5k/`", "CPU baseline run (test Top-1 45.05 %)", "Section 4.7"],
    ["`06_report/`", "This report (Word and PDF)", "—"],
    ["`07_submission/`", "Similarity and compliance checks for submission", "—"],
  ], [2600, 4860, 1900]),

  h1NoBreak("Appendix B — Reproduction Commands"),
  para("The two commands below reproduce the CPU validation run and the full-scale GPU run. Additional flags (for example `--resume auto` after an interruption) are documented in the project README."),
  ...code([
    "# CPU validation run (19 minutes, 4,500 training images, atto preset)",
    "python train.py --run-tag mvp_atto_subset5k --model atto --epochs 10 --batch-size 128 \\",
    "    --lr 2e-3 --warmup-epochs 2 --no-mixup --train-subset-size 5000 --test-subset-size 2000 \\",
    "    --data-root ../04_data/cifar10 --output-dir ../05_results \\",
    "    --exp-name mvp_atto_subset5k --plot-every 2 --num-workers 0 --seed 42",
    "",
    "# Full-scale run (about 3 hours on an 8 GB GPU)",
    "python train.py --run-tag full_convnext_tiny_cifar10 --model tiny --epochs 60 \\",
    "    --batch-size 128 --stem-mode conv --device cuda --num-workers 0 \\",
    "    --data-root ../04_data/cifar10 --output-dir ../05_results",
  ]),

  h1NoBreak("Appendix C — Evidence Index for AI-Generated Code"),
  table([
    ["Evidence", "Location", "What it proves"],
    ["Algorithm search dialogue (5 rounds, with the candidate table)", "`02_ai_chats/I0_algorithm_search_chat.md`", "Report item 1: how the AI was asked to find the algorithm"],
    ["Algorithm profile with venue, arXiv identifier and licence", "`02_ai_chats/convnext_algorithm_profile.md`", "Report item 2: verifiable algorithm description and provenance"],
    ["Sources embedded in the generated notebook, verbatim self-check", "`03_code/notebooks/`", "Report item 3: the code base was generated as a whole and stays consistent"],
    ["Per-epoch metrics, training logs, checkpoints, figures", "`05_results/full_convnext_tiny_cifar10/`", "Report item 4: experiment settings and results"],
    ["Failure diagnosis of the fp16 run (loss scale, optimiser moments)", "`05_results/_failed_runs/full_convnext_tiny_cifar10_amp_fp16_underflow/`", "Sections 3.5 and 4.6: verification discipline"],
    ["Dialogue interface screenshots", "`02_ai_chats/screenshots/`", "Report items 1, 2 and 3: the dialogues really took place"],
  ], [3200, 3400, 2760]),
);

// --------------------------------------------------------------------------- //
// Assemble document
// --------------------------------------------------------------------------- //
const doc = new Document({
  creator: "AI-driven ConvNeXt project (CISC3024)",
  title: "Reimplementing ConvNeXt for CIFAR-10 Image Classification",
  description: "AI Assignment #1 report — ConvNeXt on CIFAR-10",
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: FONT, color: ACCENT },
        paragraph: { spacing: { before: 260, after: 180 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, font: FONT, color: "2E5C8A" },
        paragraph: { spacing: { before: 220, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, font: FONT, color: "404040" },
        paragraph: { spacing: { before: 180, after: 100 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 360, hanging: 240 } } } }] },
      { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 400, hanging: 300 } } } }] },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF", space: 2 } },
          children: [new TextRun({ text: "AI Assignment #1 — ConvNeXt on CIFAR-10", size: 16, color: "808080" })],
        })],
      }),
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: "Page ", size: 16, color: "808080" }),
            new TextRun({ children: [PageNumber.CURRENT], size: 16, color: "808080" }),
            new TextRun({ text: " of ", size: 16, color: "808080" }),
            new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 16, color: "808080" }),
          ],
        })],
      }),
    },
    children: CONTENT,
  }],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  fs.writeFileSync(OUT, buffer);
  console.log("written:", OUT, buffer.length, "bytes");
});
