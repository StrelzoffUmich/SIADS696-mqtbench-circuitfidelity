(TO COMPLETE FULLY LATER as project develops)

## Purpose
This file docuemnts code ownership and AI assistance for academic honesty purposes under SIADS 696's relevant policy on Generative AI usage. 

## Authors and Roles
Benjamin Strelzoff - 
Steven Flack - 
Krystian Barlozewski - 

## Per File Authorship Ledger
- table goes here

## Utilized code snippets

## AI Assistance disclosure
Per the SIADS 696 generative-AI policy, this section is required for academic honesty compliance :
    - Model : Anthropic's Claude Opus 4.7 running in Agentic CLI mode.                                 
    - Usage : Pre-project analysis of features, code development and debugging, plot generation, file organization.                                                                   
    - Not assisted : the proposal prose, the final report prose, peer review responses, standup recordings.                                                                                     
    - Verification of AI generated code : The team will review all ai-generated code before pushing to this repository. 
      Smoke tests and awareness of the faults of generative AI have been employed/considered and will continue to be.
      
## Data Provenance
Data sources : 
    - QASM Corpus : generated via the mqt.bench package, open source license. 132 circuits across 18 algorhtims at the target-independent abstraction level
      (Fidelity may vary based on the target), qubits ranging from 3-9, seeded.
    - Fidelity : Hellinger fidelity computed by simulating each circuit ideally and under FakeBrisbane noise profiles (package is 'qiskit-aer' 0.17.2, 'qiskit-ibm-runtine 0.46.1.
      FakeBrisbane  = IBM Eagle r3 127-qubit noise model simulator. 
    - Reproducibility : Run python src/load/loader.py followed by python src/load/fidelity.py to regenerate utilized data from a clean repo clone.

This project is not building on a previous Milestone I project. 
      
