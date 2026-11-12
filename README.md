# CS440 Assignment 3
Transformers for Retrieval and Translation

**Deadline:** November 24, 2026 at 10:00 PM Eastern Time. **Total:** 100 points. Work individually or in teams of up to four. Submit one `hw3.ipynb` per team on Gradescope and add all members. Late submissions receive 0 points. The handout contains the full integrity policy.

## Start here

1. Read `HW3.pdf`.
2. Run `bash setup.sh` with Python 3.12 installed. Select the CS440 HW3 kernel. The setup installs CPU PyTorch 2.14.1, NumPy 2.3.3 and matplotlib.
3. Open `hw3.ipynb` and complete the marked cells. Keep the supplied constructors, names and signatures.
4. Save before `check()`, which loads the saved code and answers and checks their formats.

| Task | Points |
|:---|---:|
| Image retrieval | 50 |
| English to Spanish translation | 50 |

The figures, 53 catalog photos and translation data are included. A photo may show objects that are missing from its tag list. Detailed requirements and complete question options are in the handout.

## Examples (optional)

Set `RUN_DEMOS=True` in the notebook to inspect retrieval results and compare generated translations with references.

## Sources

François Chollet, *Deep Learning with Python*, second edition, Chapter 11. [Official companion code](https://github.com/fchollet/deep-learning-with-python-notebooks/tree/master/second_edition), MIT license.

Open Images V7 annotations are CC BY 4.0. Photos are CC BY 2.0 with author, title, source and resizing credits in `code/hw3lib/data/catalog.json`. Translation text is from ManyThings and Tatoeba contributors, CC BY 2.0 FR. Full source records and licenses are in `sources/`.

Editable PDF, SVG and PNG figures are in `figures/`. Their source is `make_figures.py`.
