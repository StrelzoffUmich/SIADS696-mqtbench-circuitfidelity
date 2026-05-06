###  SIADS 696 Milestone II Project : MQT Benchmark dataset - pre-hardware Fidelity Analysis 

Project Statement : We are investigating the features which contribute to respective fidelity values for different algorithmic implementations on the FakeBrisbane Quantum simulator from IBM.

Background : We plan to analyze the existing MQT Benchmark dataset, available via the mqt.bench module, differentiating ourselves from the existing mqt.predictor module. 
The existing toolkit at mqt.predictor is a classification based tool : it uses various features such as the num_qubits, along with pre-calibrated noise snapshots from existing hardware to predict compilation choices : the best possible combination of device, compiler, and opt_level parameters.

We differentiate ourselves via adding new features, derived from topology and other available data, in the hopes of discovering important as of yet unknown contributing factors to the accuracy of simulated quantum hardware. 
Fidelity is a measurement of the accuracy of the quantum computation; We use Hellinger fidelity, comparing the ideal simulation to the FakeBrisbane output to determine the accuracy of the quantum computation. 
This differentiates our project from mqt.predictor, which treats quantum circuits as a classification problem. 

**Our goal : given a circuit, predict the Hellinger fidelity under a direct, noisy simulation, and hope to extract features which produce reliable outputs.
**

For the purposes of pre-project analysis, we have developed a smaller 132 circuit corpus, with plans to scale upwards, likely using HPC resources or Colab hours.

**Our planned work is to use various statistical machine learning methods to effectively predict the fidelity values of FakeBrisbane's software emulation on various algorithms.**

This document will serve as the base for the finalized file structure and important methodological notes of the SIADS 696 project. 
