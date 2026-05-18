# Applications of ERA

This directory contains research papers, preprints, and case studies showcasing real-world scientific applications of **Empirical Research Assistance (ERA)**. 

As introduced in  *"AI system designed to help scientists write expert-level empirical software,"* ERA is an AI-powered system designed to write, optimize, and execute software for empirical discovery. By systematically searching scientific literature, writing code, testing solution strategies, combining techniques, and rigorously evaluating outcomes across thousands of potential options, ERA achieves expert-level performance across diverse benchmarks.

Over the past several months, Google Research scientists and collaborative academic institutions have deployed ERA across critical disciplines—demonstrating how AI democratizes expert computational modeling, resolves outstanding theoretical puzzles, and extracts unprecedented utility from observational data.

---

## Summaries of Published & Submitted Research

### 1. Public Health: Hospitalization Forecasts for Flu, COVID-19, and RSV
* **Project Title**: Epidemiology Forecasting
* **External Link**: *Pending arXiv link*
* **Local PDF**: [Epidemiology_Forecasting.pdf](pdfs/Epidemiology_Forecasting.pdf) *(embargo placeholder)*

**Summary**: 
Following up on the retrospective predictive models established in the foundational ERA study, this effort establishes real-time, prospective weekly hospitalization forecasts across the United States at a state level. Forecasting up to four weeks in advance for influenza, COVID-19, and Respiratory Syncytial Virus (RSV), the ERA-developed models consistently perform at or near the top of public forecasting leaderboards (such as the CDC FluSight and CovidHub prediction hubs). The flexible AI approach demonstrates how state-of-the-art epidemiological forecasting can be rapidly generalized to new infections and broader geographic regions.

---

### 2. Neuroscience: Discovering Mechanistic Models of Neural Activity
* **Project Title**: Discovering Mechanistic Models of Neural Activity: System Identification in an in Silico Zebrafish
* **External Link**: [arXiv:2602.04492](https://arxiv.org/abs/2602.04492)
* **Local PDF**: [2602.04492.pdf](pdfs/2602.04492.pdf)

**Summary**:
Moving beyond black-box predictive models, this research harnesses ERA for system identification. Using `simZFish` (a neuromechanical simulator of zebrafish), ERA was provided with connectivity of neural circuits, without the underlying governing equations. Tasked with connecting visual stimuli to motor responses, ERA discovered interpretable, mechanistically accurate models that successfully recover ground-truth dynamics and generalize robustly to unobserved stimuli.

---

### 3. Theoretical Physics: Solving Singularities in Cosmic Strings Power Spectrum
* **Project Title**: Solving an Open Problem in Theoretical Physics using AI-Assisted Discovery
* **External Link**: [arXiv:2603.04735](https://arxiv.org/abs/2603.04735)
* **Local PDF**: [2603.04735.pdf](pdfs/2603.04735.pdf)

**Summary**:
This paper uses ERA together with Gemini Deepthink to find both exact solutions and also asymptotic limits for an integral that arose in the study of cosmic strings. The scoring function for these problems validated the mathematical formulas -- given in python -- against numerical solutions of the integrals. With this, we found six complete general solutions and an asymptotic limit formula valid to high order. Beyond the integral itself, this shows how ERA with an associated scorable task can be usedto calculate asymptotic expansions and exact formulas for problems that would take significant human labor and ingenuity.

---

### 4. Climate & Sustainability: High-Frequency CO2 Monitoring from Weather Satellites
* **Project Title**: Quantification of Atmospheric Carbon Dioxide from the Geostationary Operational Environmental Satellite (GOES East)
* **External Link**: [GitHub Project Paper](https://github.com/asonabend/DeepXCO2/blob/main/Quantification%20of%20atmospheric%20carbon%20dioxide%20from%20the%20Geostationary%20Operational%20Environmental%20Satellite%20(GOES%20East).pdf)
* **Local PDF**: [DeepXCO2.pdf](pdfs/DeepXCO2.pdf)

**Summary**:
Current dedicated greenhouse gas monitoring satellites provide highly precise observations but scan narrow swaths of the Earth with 16-day revisit times. Conversely, geostationary weather satellites like GOES-East monitor entire hemispheres every 10 minutes but lack dedicated CO2 instrumentation. Using ERA, researchers constructed a physics-guided neural network that fuses 16 GOES-East wavelength bands with meteorological and solar angle data to successfully distill column-averaged CO2 concentrations at unprecedented spatial and 10-minute temporal resolution.

---

### 5. Solar Energy Engineering: 3D Solar Topography Maximization
* **Project Title**: Optimized Three-Dimensional Photovoltaic Structures with LLM guided Tree Search
* **External Link**: [arXiv:2605.16191](https://arxiv.org/abs/2605.16191)
* **Local PDF**: [2605.16191.pdf](pdfs/2605.16191.pdf)

**Summary**:
Exploring architectural optimizations for next-generation solar energy harvesting, this study leverages ERA alongside Google Antigravity to optimize 3D solar cell panel topographies. The automated optimization search discovered that a highly specialized 500-triangle volumetric fan arrangement successfully traps scattered solar radiation with near-zero backward shading, significantly multiplying energy absorption efficiency.

---

### 6. Hydrology: Runoff Forecasting for Snow-Fed River Basins
* **Project Title**: Runoff Forecasting across California Snow-Fed River Basins
* **External Link**: *Pending arXiv link*
* **Local PDF**: [Runoff_Forecasting.pdf](pdfs/Runoff_Forecasting.pdf) *(embargo placeholder)*

**Summary**:
Accurate predictions of spring snowmelt runoff are critical for urban water resource allocation and agricultural planning. Researchers utilized ERA to synthesize a seasonal runoff forecasting model across California's river basins. The generated empirical model produces significantly more precise long-horizon spring runoff forecasts than California's official state baseline outlook (Bulletin 120 / B120).

---

### 7. Macroeconomics: Weekly Retail Sales Forecasting
* **Project Title**: Retail Sales Forecasting
* **External Link**: *Pending Link*
* **Local PDF**: [Retail_Sales_Forecasting.pdf](pdfs/Retail_Sales_Forecasting.pdf) *(embargo placeholder)*

**Summary**:
Applying empirical optimization to economic tracking, this project constructs highly accurate weekly retail forecasting models. Feeding on public U.S. economic indicators, Google Trends density, consumer sentiment metrics, and historical demand cycles, the ERA-devised model successfully matches or surpasses leading commercial consensus estimates and the Chicago Fed Advance Retail Trade Summary (CARTS).

---

### 8. Combinatorics: Proof Constructions in Knuth's Cycles
* **Project Title**: Simple Even-Case Constructions in Knuth's Cycles and Their Gemini-Generated Proofs
* **External Link**: [GitHub Project Paper](https://github.com/dpwoodru/knuthCycles/blob/main/Simple%20Even-Case%20Constructions%20in%20Knuth's%20Cycles%20and%20Their%20Gemini-Generated%20Proofs.pdf)
* **Local PDF**: [KnuthCycles.pdf](pdfs/KnuthCycles.pdf)

**Summary**:
An exploration of classical combinatorial problems using large language models. ERA was used to generate initial ideas for further exploration to Knuth's even cycle problem. These were then significantly expanded using theorem verification to design simple, elegant even-case constructions within Knuth's cycles, complete with comprehensive, formally validated Gemini-generated proofs.
