# Build ASB-Benchmark from DTD

## Download

The Kaggle mirror may be used for convenience:

https://www.kaggle.com/datasets/jmexpert/describable-textures-dataset-dtd

The official source remains:

https://www.robots.ox.ac.uk/~vgg/data/dtd/

For this project, extract one complete copy below:

~~~text
Absolute_Dataset/dtd/
~~~

The builder searches nested folders automatically. A complete release must
contain:

~~~text
dtd/
+-- images/
|   +-- banded/
|   +-- blotchy/
|   +-- other class folders
+-- labels/
    +-- train1.txt
    +-- val1.txt
    +-- test1.txt
    +-- other official split files
~~~

## Verify and build manifests

Run this single command from the repository root:

~~~powershell
python -m scripts.build_asb_dtd_benchmark --data-root Absolute_Dataset --output-dir Absolute_Dataset/asb_dtd_v1 --split-number 1 --poison-ratio 0.10 --seed 42
~~~

Do not use the skip-checksums option for the final benchmark record.

The command verifies:

- 5,640 images;
- 47 classes;
- 120 images per class;
- 40 images per class in each official train, validation, and test split;
- every split entry resolves to an image;
- every indexed image appears in the official split.

It writes:

~~~text
Absolute_Dataset/asb_dtd_v1/
+-- clean_manifest.jsonl
+-- variant_manifest.jsonl
+-- trigger_catalog.json
+-- benchmark_summary.json
~~~

Images are not duplicated. Trigger variants are deterministic manifest
instructions and will be generated on demand.

## If labels are missing

Some third-party mirrors may contain only the images. If the builder reports
that the labels directory is missing, download the official labels archive from
the Oxford DTD page and place the extracted labels folder beside images.

Do not create a random split when the official split is available.
