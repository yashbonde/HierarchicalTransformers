# VaayuVidha Research Paper

## Introduction

- What is Climate Modelling?
- Why is it so important?
  - Applications in the Industry
  - Ways it affect average life of humans
  - Climate change
- Why ground observations?
- Availability of datasets
- Allow small researchers to work on it

## Related Work

- Ground base data
  - Considers only one feature onver one city
  - does not represent an actual activity
  - Effect of weather in one part affects other part, locust attack in Africa and Monsoons

- Satellite Imagery
  - Compute Intensive
  - No open source data available

- ML models
  - Language models
  - long term modelling

## Dataset
- How we decompose the problem to graph prediction
- How the connections work
- Dataset source
- Great Circle Distance
- Train, Test datasets

## Graph Neural Networks
- Graph Neural Networks brief
- Graph Neural Networks function from Inductive biases paper

### Transformers
- What are transformers
- How transformers are GNNs with formulation

### Heirarchical Transformers
- Encoder decoder blocks
- attention with Edge matrix
- How the data is blocked [B, T, N, E], [B, T, G]

### [Paragraph] Training Strategy
- System used
- Model configurations
- LR Scheduling and training

## Results
- How we validate
- What are the results
- Samples

## Conclusion
- What we learned

### Scaling to a global model
- replace the architecture with GNNs
- Introduce more nodes in the graph
  - Allow fo ever scaling
  - Nodes carrying different information

## References
- Transformer Network
- TransfoXL paper
- Inductive Biases paper
- Message passing GNN
- huggingface repo
- AdamW
- OnceCycleLR