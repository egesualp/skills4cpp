flowchart TD
    subgraph Input
        A[Job Title] 
        B[ESCO Label]
    end

    subgraph Encoder
        A --> C[Shared Encoder fθ]
        B --> C
    end

    subgraph Symmetric Setup
        C --> D[Embeddings in same space]
        D --> E[Contrastive Loss: align positives, push negatives apart]
    end

    subgraph Asymmetric Setup
        A --> F[Encoder Head fθ_job]
        B --> G[Encoder Head fθ_esco]
        F --> H[Job Embedding Space]
        G --> H
        H --> I[Contrastive Loss across spaces]
    end

    style Symmetric Setup fill:#DFF,stroke:#09C
    style Asymmetric Setup fill:#FFD,stroke:#C90