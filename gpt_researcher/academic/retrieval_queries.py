from __future__ import annotations

from .schemas import EvaluationDimension


DIMENSION_RETRIEVAL_QUERIES: dict[EvaluationDimension, str] = {
    "research_question_clarity": (
        "Where does the paper state its research problem, objectives, "
        "scope and motivation?"
    ),
    "methodological_rigor": (
        "What evidence describes and justifies the method, assumptions, "
        "technical design and implementation details?"
    ),
    "experimental_design": (
        "What datasets, evaluation protocols, controls, metrics, ablations "
        "and sample sizes are used in the experiments?"
    ),
    "baseline_quality": (
        "Which baselines are compared, and is there evidence that the "
        "comparison conditions are relevant, competitive and fair?"
    ),
    "result_validity": (
        "What results, uncertainty analyses and negative findings support "
        "or limit the paper's conclusions?"
    ),
    "reproducibility": (
        "Does the paper report data, model versions, random seeds, "
        "hyperparameters, hardware, code and evaluation procedures?"
    ),
    "novelty": (
        "Where does the paper distinguish its contributions from prior "
        "work and explain what is technically new?"
    ),
    "limitations_discussion": (
        "What assumptions, failure cases, generalisation limits, ethical "
        "concerns and unresolved questions does the paper acknowledge?"
    ),
    "citation_quality": (
        "Which evidence shows that important claims are connected to "
        "relevant and balanced prior work?"
    ),
}
