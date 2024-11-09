# How it works?
```mermaid
flowchart TD
    Q["Question/Math Problem"] --> CS["Current State"]
    
    CS --> P["Prover"]
    CS --> SP["Sneaky Prover"]
    
    %% Regular Prover Path
    P --> PS["Generate Solution Step"]
    PS --> PV{"Verifier Check"}
    PV -->|"Correct (+1)"| PR["Prover Reward"]
    PV -->|"Incorrect (-1)"| PR
    PV -->|"Neutral (0)"| PR
    
    %% State Update Logic
    PV -->|"Correct/Neutral"| US1["Update State with Prover's Step"]
    PV -->|"Incorrect"| US2["Update State with Prover's Step + Verifier's Explanation"]
    
    %% Sneaky Prover Path
    SP --> SS["Generate Alternative Step with Deliberate Error"]
    SS --> SE["Explain Error"]
    SE --> SV{"Verifier Check"}
    
    %% Verifier Training Paths
    SV -->|"Caught Error (+1)"| VR["Verifier Reward"]
    SV -->|"Missed Error (-1)"| VR
    SV -->|"Missed Error"| VT["Train Verifier with Sneaky Prover's Explanation (+1)"]
    
    %% Sneaky Rewards
    SV -->|"Caught Error (-1)"| SR["Sneaky Prover Reward"]
    SV -->|"Missed Error (+1)"| SR
    
    %% Continue or End
    US1 & US2 --> CO{"Prover's Step contains [EOS]?"}
    CO -->|"No"| CS
    CO -->|"Yes"| End["End Training"]
    
    %% Custom CSS for Better Readability with Black Text
    classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px,font-size:16px,color:black
    classDef question fill:#ffccff,stroke:#333,stroke-width:2px,font-size:18px,color:black
    classDef prover fill:#ccffcc,stroke:#333,stroke-width:2px,font-size:18px,color:black
    classDef sneaky fill:#ffcccc,stroke:#333,stroke-width:2px,font-size:18px,color:black
    classDef state fill:#e6e6ff,stroke:#333,stroke-width:2px,font-size:18px,color:black
    classDef verifier fill:#ffe6cc,stroke:#333,stroke-width:2px,font-size:18px,color:black
    
    class Q question
    class P,PS,PR prover
    class SP,SS,SE,SR sneaky
    class CS,US1,US2 state
    class VR,VT verifier
```

# To do:
1. Finish the selfplay.py file:
   - ~~add stop tokens for the critic~~
   - ~~make the models actually load (I think we can use the same base model for all of those)~~
   - ~~make the loop run until the prover finished solving the question~~
   - verify if there aren't any errors in the code
2. Figure out a way how to make the model split the task into more steps (original prover-verifier games paper does that so maybe we can find some useful info there)
3. Add final solutions verification too (maybe at start we can use some better llm, later we would have to do something else, we could also verify some solutions by ourselves, especially the ones that the critic wasn't fully certain about)
4. Make a dataset with increasingly harder questions

## Long term:
1. How to improve the model in the domains without easily verifiable ground truth?

