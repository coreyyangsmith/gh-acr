# Motivation
This repository aims to evaluate the efficacy of multi-agent systems for automating software engineering tasks through their application in resovling Github merge conflicts.

Multi-agent systems can follow any number of paradigms including collaborative frameworks, centralized frameworks, or combative debate-based frameworks. Oftentimes, in a single sequential multi-agent pipeline or overall system, there many be one or multiple combinations of the aforementioned sections tha contribute to the overall system.

As large language models have been known to be agreeable, and struggle when dealing with negative examples with few-shot learning, inherently, debate-based frameworks seems to provide the most resistance in completing any automation task. By having multiple agents actively work against eachother, to reach any resolution must indicate either (a) common ground through reasoning (or otherwise) has truly been reached, (b) depending on the type of agent, they have become so agreeable as to be submissive to others in the debate chain, or (c) a stand-still has been reached and a manual control flow has been triggered to break the tie.

Resolving merge conflict, inevitably then, lie in this category of debate-based reasoning amongst multi-agents, where each agent (developer) advocates for the logic inclusion of their statements above others, and must both (a) negotiate and (b) communicate their changes in order to have a chance at inclusion.

Thus, the better that (1) their argument (qualitative), and (2) their code (empirical?), the better the chance they have at inclusion. Conversely, who they are 'matched against' (competition) also plays a significant factor, consider how (1) combative vs pliant they are, or perhaps (2) cooperative/agreeable vs stubborn.

These factors in unison, in the presence of a larger framework and wrapped in the overall context of problem statement and achievable goal, must then allow/force an agent framework to resolve the task given the indeterminancy of the (1) agents provided and the (2) external structure of the framework (overall problem statement) and (3) internal structure (i.e. methodology/implementation of multi-agentic structure to solve (2)).

Automated conflict resolution provides a challenging structure over other software engineering tasks by inherently posing developers contributions against each other in an environment where negotiation and reasoning over a broader code base and standards is paramount. This allows us to independently evaluate the (1) influence of different agent types/personalities/prompts in influencing the overall network, and (2) how the greater system/collaboration of agents contributes to or challenges convential belief.

Considering all of us, we will be able to asses the performance of different agent types and architectures, and evaluate the challenges of implementing multi-agent systems in challenging, negotitation-driven environments.

# Dataset
We consider the [ConGra](https://arxiv.org/abs/2409.14121) dataset