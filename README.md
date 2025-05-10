# PyTorch Workflow

```mermaid
flowchart TD
    A[User Question] --> B{Has Context?}
    B -- Yes --> C[Extractive QA]
    B -- No --> D[Generative QA]
    C --> E[Extracted Answer]
    E --> F{Needs Rephrasing?}
    F -- Yes --> G[Rephrase with Generative Model]
    F -- No --> H[Return Raw Answer]
    G --> H
    D --> H
    H --> I[Final Answer]