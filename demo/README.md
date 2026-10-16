# CPU Image-Upload Demonstration

Open [`dtd_reference_free_generator_demo.ipynb`](dtd_reference_free_generator_demo.ipynb)
in VS Code or Jupyter and run the cells from top to bottom.

The notebook uses the final frozen checkpoints and defaults to CPU. It provides:

1. A browser upload widget.
2. A verified DTD fallback image.
3. A controlled clean-to-trigger-to-correction demonstration.
4. A deployment-path demonstration using an already-triggered image.
5. Prediction, amplitude and correction visualizations.
6. Saved PNG images and a JSON report under `Advanced_Outputs`.

The controlled demonstration is saved as five presentation-ready stages:

1. `01_original_image.png`: original image and prediction-confidence graph.
2. `02_trigger_injection.png`: before/after images and their frequency amplitudes.
3. `03_without_generator.png`: attacked image, prediction and confidence.
4. `04_with_generator.png`: triggered spectrum, applied correction and reconstruction.
5. `05_final_result.png`: final comparison and attack-target probability graph.

The notebook displays these stages in order, which makes it suitable for a
screen-recorded demonstration. It also saves `prediction_probability_graph.png`
as a standalone result graph for a presentation or paper.

No training or fine-tuning occurs in the notebook. A complete inference run
takes only a few seconds on the tested local CPU environment.
