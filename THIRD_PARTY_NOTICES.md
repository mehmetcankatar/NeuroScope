# Third-party notices

## SPaRCNet

Source: [bdsp-core/IIIC-SPaRCNet](https://github.com/bdsp-core/IIIC-SPaRCNet), pinned revision `d374d5137fe30e595d686ecc1b584a2e551f1aa0`.

Reference: Jing J, Ge W, Hong S, et al. *Development of Expert-Level Classification of Seizures and Rhythmic and Periodic Patterns During EEG Interpretation*. Neurology. 2023;100(17):e1750–e1762. [doi:10.1212/WNL.0000000000207127](https://doi.org/10.1212/WNL.0000000000207127).

`backend/neuroscope/sparcnet_network.py` contains the architecture class definitions from the author's `SPaRCNet/runSPaRCNet.py`, with formatting and import organization adapted for a module. The author checkpoint is downloaded separately by setup; it is not committed here. The setup records original and converted hashes and checks equivalent outputs. NeuroScope's channel mapping, EDF ingestion, chunk processing, UI and postprocessing are separate application work.

The upstream repository includes the [Open Data Commons Attribution License v1.0](third-party/SPaRCNet-LICENSE.md), preserved verbatim here. It is a database license and explicitly excludes computer programs from its scope. Do not interpret NeuroScope's MIT license or the existence of this public download as a separate permissive software/checkpoint license or a grant of commercial rights. Consult the authors for clarification of rights for redistribution or commercial deployment. Attribution is retained; no author endorsement is implied.

## YASA

[YASA](https://github.com/raphaelvallat/yasa), version 0.6.5, uses the BSD 3-Clause license, copyright Raphael Vallat. The installed package includes its license and pretrained classifier files; those files are not vendored in this repository.

Reference: Vallat R, Walker MP. *An open-source, high-performance tool for automated sleep staging*. eLife. 2021;10:e70092. [doi:10.7554/eLife.70092](https://doi.org/10.7554/eLife.70092). NeuroScope uses YASA's pretrained classifiers and automated feature extraction without fitting on the test examples.

## Public evaluation recordings

- Guttag J. *CHB-MIT Scalp EEG Database*, version 1.0.0. PhysioNet. [doi:10.13026/C2K01R](https://doi.org/10.13026/C2K01R).
- Kemp B, Zwinderman AH, Tuk B, Kamphuisen HAC, Oberyé JJL. *Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG*. IEEE Transactions on Biomedical Engineering. 2000;47(9):1185–1194. [Sleep-EDF release](https://physionet.org/content/sleep-edfx/1.0.0/).

Downloaded recordings and annotation files remain outside source control. Consult each PhysioNet release for its data license and required citations. `configs/public-examples.json` records exact download URLs and hashes. These examples are not a representative clinical evaluation cohort and development-cohort independence has not been established.

## Optional MORGOTH and other dependencies

MORGOTH is developed by bdsp-core and its research collaborators. Source: https://github.com/bdsp-core/morgoth. Pinned integration revision: `3e449003287465127b441f492263579c805c741d`. The upstream repository's license is **Creative Commons Attribution-NonCommercial 4.0**. Commercial use is prohibited under that license. No MORGOTH source or checkpoints are included in the tracked NeuroScope repository; the setup script downloads source separately and preserves its license. Redistribution/use of checkpoints and datasets must follow their respective terms.

Research reference: Sun C, Karakis I, Herlopian A, et al. *Toward unified and comprehensive automated electroencephalogram interpretation: a multicentre development and validation of an electroencephalogram foundation model*. The Lancet Digital Health (2026). DOI: https://doi.org/10.1016/j.landig.2026.101039. See the official release for the complete authorship and citation: https://bdsp.io/content/morgoth1/1.0.0/.

The project uses MNE-Python for EDF reading; FastAPI/Uvicorn for the API; NumPy/SciPy for numerical processing; PyTorch/timm/einops and upstream dependencies for optional inference; React, TypeScript, Vite, and Lucide for the interface. Their licenses are distributed with the corresponding installed packages. No third-party brand assets or patient data are bundled.

The visual identity and synthetic demo are original to this project. OpenAI Codex assisted with implementation and documentation.
