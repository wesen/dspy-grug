# DSPy Signatures

DSPy Signatures are declarative specifications of input/output behavior for DSPy modules. They provide a way to define the structure and semantics of tasks for language models without specifying implementation details.

## Overview

DSPy Signatures are a fundamental building block in the DSPy framework, providing a declarative way to specify the input/output behavior of language model (LM) tasks. They offer several key benefits and features for developers:

1. **Declarative Task Specification**: Signatures allow you to define what the LM needs to do, rather than how to do it. This abstraction enables more modular and cleaner code.

2. **Semantic Role Definition**: Field names in signatures are semantically meaningful, allowing you to express roles in plain English (e.g., 'question', 'answer', 'context').

3. **Optimization Ready**: Signatures are designed to work with DSPy's optimization tools, enabling automatic generation of high-quality prompts or fine-tuning strategies.

4. **Flexibility in Definition**: Signatures can be defined inline as simple strings or as more detailed class-based structures for complex tasks.

5. **Integration with DSPy Modules**: Signatures form the basis for creating DSPy modules, which can be compiled into optimized prompts or fine-tuned models.

6. **Support for Multiple Fields**: Signatures can handle multiple input and output fields, accommodating complex task structures.

7. **Type Annotations**: Custom types can be specified for fields using Python type annotations, defaulting to `str` if not specified.

8. **Instructions and Constraints**: Class-based signatures allow for the inclusion of task instructions (as docstrings) and field-specific descriptions or constraints.

9. **Auxiliary Information**: Some DSPy modules automatically expand signatures to include additional fields for auxiliary information (e.g., 'rationale' in `ChainOfThought`).


## Basic Structure

A signature consists of input and output fields, separated by an arrow (`->`). Field names are semantically meaningful and define roles for inputs and outputs.

```python
classify = dspy.Predict('sentence -> sentiment')
```

## Multiple Fields

Signatures can have multiple input and output fields, separated by commas.

```python
qa = dspy.ChainOfThought('context, question -> reasoning, answer')
```

## Class-based Signatures

For more complex tasks, class-based signatures allow for additional specifications like instructions and field descriptions.

```python
class Emotion(dspy.Signature):
    """Classify emotion among sadness, joy, love, anger, fear, surprise."""
    
    sentence = dspy.InputField()
    sentiment = dspy.OutputField()
```

## Field Types

By default, fields are of type `str`. Custom types can be specified using Python type annotations.

```python
class CustomSignature(dspy.Signature):
    number: int = dspy.InputField()
    result: float = dspy.OutputField()
```

## Instructions

Signatures can include instructions as a docstring, which some DSPy optimizers can use to generate more effective prompts.

```python
class Summarize(dspy.Signature):
    """Provide a concise summary of the given text."""
    
    text = dspy.InputField()
    summary = dspy.OutputField()
```

## Field Descriptions

Input and output fields can have descriptions to provide more context or constraints.

```python
class FactCheck(dspy.Signature):
    context = dspy.InputField(desc="Verified facts")
    claim = dspy.InputField(desc="Statement to be fact-checked")
    verdict = dspy.OutputField(desc="True/False/Partially True")
```

## Signature Creation

Signatures can be created dynamically using the `make_signature` function or the `Signature` class.

```python
dynamic_sig = dspy.Signature("input1, input2 -> output", "Process inputs and generate output")
```

## Integration with DSPy Modules

Signatures are used to define the behavior of DSPy modules, which can then be compiled into optimized prompts or fine-tuned models.

```python
summarizer = dspy.Module(Summarize)
compiled_summarizer = dspy.Compile(summarizer)
```

By using signatures, DSPy enables modular, adaptive, and reproducible development of language model applications.