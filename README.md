# SteelDefectX Dataset

SteelDefectX is a vision-language dataset for steel surface defect analysis, containing **7,778 images** across **25 defect categories**. Built by unifying four public benchmarks (NEU, GC10, X-SDD, and S3D), it provides **multi-form textual annotations** at both class and sample levels, alongside pixel-level segmentation masks. The dataset is designed to enable vision-language learning, industrial anomaly detection, and cross-dataset transfer research.

## Dataset Structure

```text
SteelDefectX/
├── class_descriptions.json
├── train/
├── train_mask/
├── train-text.json
├── val/
├── val_mask/
└── val-text.json
```

## Files

- `train/` & `val/`: training and validation images (7:3 split, ~5,454 train / ~2,324 val).
- `train_mask/` & `val_mask/`: pixel-level binary segmentation masks.
- `train-text.json` & `val-text.json`: per-sample annotations with multiple text forms.
- `class_descriptions.json`: class-level textual descriptions (name, visual attributes, industrial causes).

## Annotation Format

Each sample-level annotation includes **three complementary text forms** to support diverse vision-language tasks:

### Multi-Form Text Annotations

- **T2 (Natural Language Description)**: Free-form, richly expressive descriptions generated via GPT-4o and refined through semantic similarity filtering and manual validation. Covers appearance, shape, size, depth, position, and contrast.
  
- **T3 (Structured Attributes)**: Compact, standardized representation with nine fields: defect type, shape, direction, spatial distribution, number of defects, position (3×3 grid), scale, polarity, and saliency. Derived from segmentation masks and constrained LLM prediction for reproducibility.

- **T4 (Template Sentence)**: Linearized structured attributes into a consistent sentence pattern. Balances semantic completeness with linguistic conciseness for controllable text-guided applications.

### Example Annotation

```json
{
  "image_name": "bs_01.jpg",
  "class_name": "Bright scratch",
  "natural_language_description": "The bright scratch defect appears as a thin, vertical line on the steel surface...",
  "structured_attributes": {
    "Defect type": "Bright scratch",
    "Shape": "linear",
    "Direction": "vertical",
    "Spatial Distribution": "isolated",
    "Number of Defects": "one",
    "Position": [["bottom-center", 0.467], ["center", 0.416]],
    "Scale": "tiny",
    "Polarity": "bright",
    "Saliency": "medium"
  },
  "template_sentence": "A tiny bright Bright scratch is observed on the steel surface. It has a linear shape..."
}
```

## Supported Defect Classes

25 categories: Bright scratch, Crazing, Crease, Crescent gap, Dark scratches, Finishing roll printing, Inclusion, Iron scale compression, Iron sheet ash, Oil spot, Oxide scale of plate system, Oxide scale of temperature system, Patches, Pitted surface, Punching, Red iron sheet, Rolled in scale, Rolled pit, Secondary rust skin, Silk spot, Slag inclusion, Waist folding, Water spot, Welding line, White rust.

## Technical Notes

- **Data Source**: Unified taxonomy from NEU, GC10, X-SDD, and S3D public benchmarks.
- **Image Resolution**: All images standardized to 256×256 pixels.
- **Mask Format**: Binary PNG images; foreground pixels denote defect regions.
- **Annotation Quality**: Sample-level T2 descriptions validated via ~275 hours of manual review for terminology consistency and completeness.


See the associated paper for detailed methodology on multi-form annotation generation and comprehensive benchmark results.
