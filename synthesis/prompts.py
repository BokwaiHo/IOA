"""
Prompts for Synthetic Data Generation

This module contains all prompting templates for the Adapter module
as described in Section 3.4 and Appendix I/J.

The five adaptation dimensions are:
1. Abstract Concept Concretization - concrete analogies before formalism
2. Complex Reasoning Decomposition - step-by-step breakdown
3. Cognitive Load Management - controlled complexity
4. Representation Format Optimization - templated formats
5. Linguistic Complexity Reduction - simplified language

References:
- Section 3.4: Adapter knowledge representation strategies
- Appendix I: Knowledge Representation Adaptation Prompts
- Appendix J: Practical Prompting Guidelines
"""

from typing import List, Dict, Any, Optional


# =============================================================================
# System Prompts
# =============================================================================

SYSTEM_PROMPT_SYNTHESIS = """You are a teacher LLM generating pedagogically adapted synthetic data for a student model (SLM). Your goal is to align knowledge representation with the student's cognitive capacity.

Strictly enforce the following adaptation requirements:

1) Abstract Concept Concretization: Begin with concrete analogies or real-world examples before introducing formal definitions or symbolic notation. Make abstract concepts tangible.

2) Complex Reasoning Decomposition: Present explicit, small-step reasoning. Break down multi-step processes into atomic cognitive operations with clear transitions.

3) Cognitive Load Management: Start minimal and increase difficulty gradually. Control information density and complexity. Begin with simple cases before complex ones.

4) Representation Format Optimization: Use a consistent stepwise template. Provide standardized solution scaffolds that enable pattern extraction.

5) Linguistic Complexity Reduction: Prefer simple words, short sentences, and clear connectors (e.g., "first", "next", "therefore"). Replace jargon with simpler equivalents.

If reasoning or verification fails, discard the example and regenerate. All outputs MUST follow the JSON schema provided by the user prompt."""


SYSTEM_PROMPT_REMEDIAL = """You are generating remedial synthetic data for a student model that has NOT achieved mastery on certain knowledge modules. The student needs additional practice on basic concepts.

Requirements:
- Reduce linguistic and structural complexity even further than standard synthesis
- Keep instance size minimal; remove distracting details
- Focus ONLY on the weak sub-skills identified
- Maintain explicit step decomposition and verification
- Use even more concrete examples and simpler language

Your goal is to help the student master prerequisite concepts before advancing."""


SYSTEM_PROMPT_BRIDGING = """You are generating bridging synthetic data for a student model that has achieved mastery and is ready for slightly more challenging material.

Requirements:
- Increase complexity by ONLY one notch (do NOT skip levels)
- Bounded difficulty increments following Zone of Proximal Development
- Maintain all five adaptation requirements
- Provide appropriate scaffolding for the new difficulty level

Your goal is to smoothly transition the student to more advanced concepts."""


# =============================================================================
# User Prompts for Data Synthesis
# =============================================================================

def get_synthesis_user_prompt(
    domain: str,
    stage_id: str,
    knowledge_modules: List[str],
    prerequisites: List[str],
    num_examples: int,
    size_cap: str = "medium",
    complexity_cap: str = "moderate",
    baseline_ratio: float = 0.5
) -> str:
    """
    Generate the user prompt for synthetic data generation.
    
    This implements the User Prompt Template from Appendix J.4.
    
    Args:
        domain: Target domain (e.g., "math_problem_solving")
        stage_id: Current curriculum stage ID
        knowledge_modules: Modules to generate data for
        prerequisites: Prerequisite modules
        num_examples: Number of examples to generate
        size_cap: Problem size constraint
        complexity_cap: Complexity constraint
        baseline_ratio: Student's baseline performance ratio
    
    Returns:
        Formatted user prompt string
    """
    modules_str = ", ".join(knowledge_modules)
    prereqs_str = ", ".join(prerequisites) if prerequisites else "None"
    
    return f"""Target Domain: {domain}
Curriculum Stage: {stage_id}
Knowledge Units: {modules_str}
Prerequisites: {prereqs_str}
Student Baseline (relative to teacher): {baseline_ratio:.2f}

Please generate {num_examples} new synthetic examples adapted to this stage.

Requirements:
- Obey the five adaptation dimensions (concretization, decomposition, cognitive load, format, language).
- Cognitive load constraints: max problem size {size_cap}, max symbolic/arithmetic complexity {complexity_cap}.
- Provide a problem that transitions from concrete analogy to symbolic form.
- Provide a full solution with explicit step-by-step reasoning and verification.
- Output MUST conform to the JSON schema below.

JSON Schema (all keys required):
[
    {{
        "module": "<knowledge unit identifier>",
        "prereq": ["<prereq1>", "<prereq2>"],
        "difficulty_tag": "<introductory|intermediate|advanced>",
        "problem": "<text: concrete analogy -> symbolic formulation>",
        "solution": {{
            "steps": ["Step 1: ...", "Step 2: ...", "..."],
            "final_answer": "<canonical answer>",
            "verification": "<independent check; describe or show test>"
        }},
        "adapter_flags": {{
            "concretization": true,
            "decomposition": true,
            "cognitive_load": {{
                "scale": "<e.g., '2x2 system' or 'small input size'>",
                "notes": "<what was simplified and why>"
            }},
            "format_template": "Stepwise-3",
            "simplified_language": true
        }},
        "metadata": {{
            "stage_id": "{stage_id}",
            "seed_style_ref": "<seed reference if applicable>"
        }}
    }}
]

Generate exactly {num_examples} examples. Ensure diversity in surface features while maintaining structural consistency."""


def get_remedial_prompt(
    stage_id: str,
    knowledge_modules: List[str],
    weak_subskills: List[str],
    num_examples: int
) -> str:
    """
    Generate prompt for remedial data synthesis.
    
    This implements the Remedial Prompt from Appendix J.6.
    
    Args:
        stage_id: Current stage ID
        knowledge_modules: Modules in the stage
        weak_subskills: Specific weak areas to target
        num_examples: Number of examples to generate
    
    Returns:
        Remedial prompt string
    """
    modules_str = ", ".join(knowledge_modules)
    weak_str = ", ".join(weak_subskills)
    
    return f"""The student did NOT achieve mastery for {stage_id}/{modules_str}.

Generate {num_examples} simplified remedial examples focusing ONLY on these weak sub-skills:
{weak_str}

Constraints:
- Reduce linguistic and structural complexity further.
- Keep instance size minimal; remove distracting details.
- Maintain explicit step decomposition and verification.
- Use the same JSON schema as before.

The goal is to help the student build foundational understanding before attempting more complex problems."""


def get_bridging_prompt(
    stage_id: str,
    knowledge_modules: List[str],
    num_examples: int
) -> str:
    """
    Generate prompt for bridging data synthesis.
    
    This implements the Bridging Prompt from Appendix J.6.
    
    Args:
        stage_id: Current stage ID
        knowledge_modules: Modules in the stage
        num_examples: Number of examples to generate
    
    Returns:
        Bridging prompt string
    """
    modules_str = ", ".join(knowledge_modules)
    
    return f"""The student ACHIEVED mastery for {stage_id}/{modules_str}.

Generate {num_examples} bridging examples with SLIGHTLY increased complexity (only one notch higher in scale/coefficients/constraints).

Constraints:
- Bounded difficulty increments; do NOT skip levels.
- Keep the five adaptation requirements.
- Use the same JSON schema as before.

The goal is to smoothly transition the student to the next difficulty level within their Zone of Proximal Development."""


# =============================================================================
# Individual Adaptation Dimension Prompts (from Appendix I)
# =============================================================================

PROMPT_CONCRETIZATION = """Explain the abstract concept of {concept} using a concrete analogy or real-world example (e.g., speed of a car for derivatives). Avoid formal definitions at first, and gradually transition to the symbolic or mathematical expression."""

PROMPT_DECOMPOSITION = """Solve the problem step by step. Break down the reasoning into small sub-steps:
(1) Extract relevant information
(2) Identify relationships
(3) Formulate equations
(4) Solve step by step
(5) Verify the solution

Provide each step explicitly."""

PROMPT_COGNITIVE_LOAD = """Reformulate the problem into a simpler version of the same type (e.g., start with a 2×2 system of equations before moving to larger systems). Ensure each sub-problem is self-contained and introduce incremental difficulty only after demonstrating mastery."""

PROMPT_FORMAT_OPTIMIZATION = """Present the solution in a consistent, structured format using the following template:
Step 1: [Action]
Step 2: [Action]
Step 3: [Action]

Use the same template across multiple examples to highlight structural patterns while varying the surface details."""

PROMPT_LINGUISTIC_SIMPLIFICATION = """Rewrite the explanation of {problem} in simpler language. Use short, direct sentences. Replace advanced terms with simpler synonyms where possible. Use clear connectors such as "first", "next", "therefore". Ensure the reasoning remains correct but linguistically accessible."""


# =============================================================================
# Domain-Specific Templates
# =============================================================================

def get_math_synthesis_template(
    module: str,
    difficulty: str
) -> Dict[str, str]:
    """
    Get math-specific synthesis template.
    
    Args:
        module: Math knowledge module
        difficulty: Difficulty level
    
    Returns:
        Template dictionary with problem and solution scaffolds
    """
    templates = {
        "algebra/linear": {
            "problem_scaffold": "Story analogy (apples, money, etc.) → translate to variables → form equation → solve",
            "solution_template": "Step 1: Identify unknowns\nStep 2: Write equation\nStep 3: Solve for variable\nStep 4: Verify by substitution",
            "verification": "Substitute answer back into original equation"
        },
        "algebra/quadratic": {
            "problem_scaffold": "Area/motion analogy → set up quadratic → factor or use formula → check",
            "solution_template": "Step 1: Identify coefficients a, b, c\nStep 2: Calculate discriminant\nStep 3: Apply quadratic formula\nStep 4: Verify both roots",
            "verification": "Substitute roots back into original equation"
        },
        "calculus/derivatives": {
            "problem_scaffold": "Rate of change analogy (speed) → define function → compute derivative → interpret",
            "solution_template": "Step 1: Understand rate of change concept\nStep 2: Apply differentiation rules\nStep 3: Simplify\nStep 4: Numerical approximation check",
            "verification": "Compare symbolic result with numerical approximation"
        }
    }
    
    # Return template or default
    return templates.get(module, {
        "problem_scaffold": "Concrete analogy → formal definition → solve → verify",
        "solution_template": "Step 1: Setup\nStep 2: Apply method\nStep 3: Compute\nStep 4: Verify",
        "verification": "Check answer against problem constraints"
    })


def get_code_synthesis_template(
    module: str,
    difficulty: str
) -> Dict[str, str]:
    """
    Get code-specific synthesis template.
    
    Args:
        module: Code knowledge module
        difficulty: Difficulty level
    
    Returns:
        Template dictionary
    """
    templates = {
        "python/string-processing": {
            "problem_scaffold": "Assembly line analogy for processing → pseudocode → implementation",
            "solution_template": "Step 1: Parse input\nStep 2: Process elements\nStep 3: Assemble output\nStep 4: Test with examples",
            "verification": "Run unit tests with edge cases"
        },
        "algorithms/sorting": {
            "problem_scaffold": "Physical sorting analogy (cards) → describe algorithm → implement → analyze",
            "solution_template": "Step 1: Understand algorithm logic\nStep 2: Write pseudocode\nStep 3: Implement in Python\nStep 4: Test and analyze complexity",
            "verification": "Test with small inputs and verify correctness"
        }
    }
    
    return templates.get(module, {
        "problem_scaffold": "Real-world analogy → pseudocode → implementation → test",
        "solution_template": "Step 1: Understand requirement\nStep 2: Design approach\nStep 3: Implement\nStep 4: Test",
        "verification": "Run tests with various inputs"
    })


# =============================================================================
# Helper Functions
# =============================================================================

def format_adapter_flags(
    concretization: bool = True,
    decomposition: bool = True,
    cognitive_load_scale: str = "standard",
    cognitive_load_notes: str = "",
    format_template: str = "Stepwise-3",
    simplified_language: bool = True
) -> Dict[str, Any]:
    """
    Create adapter flags dictionary for JSON output.
    
    Args:
        concretization: Whether concretization was applied
        decomposition: Whether decomposition was applied
        cognitive_load_scale: Scale of cognitive load
        cognitive_load_notes: Notes on simplification
        format_template: Template name used
        simplified_language: Whether language was simplified
    
    Returns:
        Adapter flags dictionary
    """
    return {
        "concretization": concretization,
        "decomposition": decomposition,
        "cognitive_load": {
            "scale": cognitive_load_scale,
            "notes": cognitive_load_notes
        },
        "format_template": format_template,
        "simplified_language": simplified_language
    }


def get_difficulty_constraints(difficulty: str) -> Dict[str, str]:
    """
    Get size and complexity constraints based on difficulty.
    
    Args:
        difficulty: Difficulty level
    
    Returns:
        Dictionary with size_cap and complexity_cap
    """
    constraints = {
        "introductory": {
            "size_cap": "small (2 variables, single-digit numbers)",
            "complexity_cap": "basic (single operations, no nested logic)"
        },
        "intermediate": {
            "size_cap": "medium (3-4 variables, double-digit numbers)",
            "complexity_cap": "moderate (multiple operations, simple nesting)"
        },
        "advanced": {
            "size_cap": "large (5+ variables, larger numbers)",
            "complexity_cap": "high (complex operations, deep nesting)"
        }
    }
    
    return constraints.get(difficulty, constraints["intermediate"])


def create_few_shot_examples(
    domain: str,
    num_examples: int = 2
) -> str:
    """
    Create few-shot examples for prompting.
    
    Args:
        domain: Target domain
        num_examples: Number of examples to include
    
    Returns:
        Formatted few-shot examples string
    """
    if domain == "math_problem_solving":
        return """
Example 1 (Introductory Linear Equation):
{
    "module": "algebra/linear",
    "prereq": ["arithmetic/basic"],
    "difficulty_tag": "introductory",
    "problem": "You and your friend together have 7 apples, and you have 1 more apple than your friend. Let x be your apples and y be your friend's apples. Write equations and solve for both.",
    "solution": {
        "steps": [
            "Step 1: Translate to equations: x + y = 7 and x - y = 1",
            "Step 2: Add equations: 2x = 8, so x = 4",
            "Step 3: Substitute back: 4 + y = 7, so y = 3",
            "Step 4: Verify: 4 - 3 = 1 ✓ and 4 + 3 = 7 ✓"
        ],
        "final_answer": "x = 4, y = 3",
        "verification": "Both equations are satisfied: 4 + 3 = 7 and 4 - 3 = 1"
    },
    "adapter_flags": {
        "concretization": true,
        "decomposition": true,
        "cognitive_load": {"scale": "2x2 system", "notes": "Integer coefficients only"},
        "format_template": "Stepwise-3",
        "simplified_language": true
    },
    "metadata": {"stage_id": "Math-S1", "seed_style_ref": "Seed-Math-001"}
}

Example 2 (Intermediate Derivative):
{
    "module": "calculus/derivatives",
    "prereq": ["algebra/functions", "limits/basic"],
    "difficulty_tag": "intermediate",
    "problem": "Think of a car's speedometer showing instantaneous velocity. If a car's position is f(t) = 3t² - 2t meters at time t seconds, find the instantaneous velocity at t = 4.",
    "solution": {
        "steps": [
            "Step 1: Velocity is rate of change of position, like speedometer reading",
            "Step 2: Compute derivative: f'(t) = 6t - 2",
            "Step 3: Evaluate at t=4: f'(4) = 6(4) - 2 = 22 m/s",
            "Step 4: Verify numerically: (f(4.001) - f(4))/0.001 ≈ 22"
        ],
        "final_answer": "22 m/s",
        "verification": "Numerical approximation confirms: ≈22.003"
    },
    "adapter_flags": {
        "concretization": true,
        "decomposition": true,
        "cognitive_load": {"scale": "single derivative", "notes": "Quadratic function, no chain rule"},
        "format_template": "Stepwise-3",
        "simplified_language": true
    },
    "metadata": {"stage_id": "Math-S2", "seed_style_ref": "Seed-Math-012"}
}
"""
    elif domain == "code_generation":
        return """
Example 1 (String Processing):
{
    "module": "python/string-processing",
    "prereq": ["python/basics"],
    "difficulty_tag": "introductory",
    "problem": "Imagine an assembly line that trims edges and spaces out items evenly. Write a function normalize_spaces(s) that removes leading/trailing spaces and ensures words are separated by exactly one space.",
    "solution": {
        "steps": [
            "Step 1: Strip leading and trailing whitespace",
            "Step 2: Split string into words by whitespace",
            "Step 3: Join words with a single space",
            "Step 4: Test with edge cases"
        ],
        "final_answer": "def normalize_spaces(s):\\n    return ' '.join(s.split())",
        "verification": "assert normalize_spaces(' a  b ') == 'a b'; assert normalize_spaces('') == ''"
    },
    "adapter_flags": {
        "concretization": true,
        "decomposition": true,
        "cognitive_load": {"scale": "small input size", "notes": "No regex, basic string methods only"},
        "format_template": "Stepwise-3",
        "simplified_language": true
    },
    "metadata": {"stage_id": "Code-S1", "seed_style_ref": "Seed-Code-010"}
}
"""
    else:
        return ""


if __name__ == "__main__":
    # Test prompt generation
    prompt = get_synthesis_user_prompt(
        domain="math_problem_solving",
        stage_id="Math-S1",
        knowledge_modules=["algebra/linear", "algebra/systems"],
        prerequisites=["arithmetic/basic"],
        num_examples=5,
        baseline_ratio=0.6
    )
    
    print("Generated User Prompt:")
    print("-" * 50)
    print(prompt)