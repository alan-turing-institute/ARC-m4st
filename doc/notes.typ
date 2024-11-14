#import "@preview/showybox:2.0.3": showybox
#set math.equation(numbering: "(1)")

= Formalising the problem

Qualitatively, the idea is that you find a dataset with $(x, y)$ pairs (source, human translation), then select a model for translation $cal(T)$.
The model produces a translation $x^prime = cal(T)(x)$, so now you want to know about the quality of the translation.
In order to tell whether one translation is better than another, ideally you would ask a translator to score the pair $(x^prime, y)$ using some absolute scale.
We are interested in the case where the scoring must be done automatically by a computer.
The metric $cal(M)$ is assumed to be able to compare $x^prime$ to $y$ and estimate the translation quality.
(If you have a good way to estimate translation quality, you can train a model against it and learn how to translate well?)

We want to study the behaviour of $cal(M)$ when the source text is "corrupted" by filler words.

#showybox(
  [*Assumption*: The translation model $cal(T)$ will preserve the semantic meaning.]
)

If this assumption is not true, then we might have to think harder about separating metric variance due to semantic change vs due to change in style.
Either way, the translation will be influenced by the domain shift due to filler words, and we can sample metric evaluations $M$, as shown below.

//In our case, they want to know which metric is the most robust against filler words. 
//This is not an adversarial case, since filler words occur naturally.
//We don't know the real world distribution of filler words, but we could use a LLM to sample from $bb(P)(hat(x) | x)$, where $x$ is the clean input, and $hat(x)$ is the filler-word-corrupted input.

The translation model can be defined as $cal(T): x arrow x^prime$, where $x^prime$ is the translated text.
The metric can be defined as $cal(M): x^prime, x, {y_i}_(i=1)^N arrow bb(R)$, where $y_i$ are reference translations provided by $N$ translators.
In our use case $N=1$.

We are generally not interested in benchmarking different models, so we can assume that $cal(T)$ is given.
The focus is on ranking a set of metrics ${cal(M)_i}$, which we should also propose.

== Quantifying Robustness

In particular, robustness against the distributional shift $hat(X) tilde bb(P)(hat(x) | x)$. $hat(x)$ is a randomly corrupted version of $x$ - by corruption I mean the addition of filler words randomly.
We could call $lambda$ the degree of shift away from $X$, such that $lambda = 0$ means that there is no corruption, and $hat(x) = x$.
For simplicity, we could define multiple levels of corruption, as they've done in @hendrycks2019benchmarkingneuralnetworkrobustness. For example $lambda = 0,1,2,3$, and build different algorithm of corruption for each level, or prompt the corrupting LLM differently (e.g. $lambda=1$: "Add a filler word", $lambda=3$, "Add lots of filler words").

We can analyse how the metric behaves as a function of $lambda$.
Per $(lambda, cal(M))$, we could plot the mean and/or variance of the metric, evaluated on a given dataset.
It might be useful to plot this for cases where the metrics must show poor translation, such as when we select the wrong translation on purpose (negative x,y pairs).
This is to explore whether the metric can still tell that a translation is bad, even when corrupted.

Below is the recipe for computing $M$ as a random variable, using the functions $cal(T)$, $cal(M)$, the dataset $cal(D)$, and $bb(P)(hat(X) | X, lambda)$ is the distribution of corrupted versions of $X$, from which we can sample.

$ (X, Y) tilde cal(D) $

$ hat(X) tilde bb(P)(hat(X) | X, lambda) $

$ hat(X)^prime = cal(T)(hat(X)) $

$ X^prime = cal(T)(X) $

$ M = cal(M)(hat(X)^prime, X^prime, Y) $ <eq:sample_metric>

Ideally, we can observse a $cal(M)$ that consistently outputs the same metric irrespective of $lambda$, for both positive and negative pairs.
We would have to quantify this based on summary statistics of $M$, such as $angle.l M angle.r$ and $"Var"(M)$.

#showybox(
    [*Definition*: A robust metric will on average provide the same mean with no increase in variance in some dataset, with respect to a corruption of the input.]
)

This definition of robustness follows from the filler word function $bb(P)(hat(x) | x, lambda)$ only changing the style, but preserving the semantic meaning.
This might be wrong, though?
If a translation model preserves the meaning and all grammatical correctness, then a metric should produce a similar mean and variance for the dataset.
Producing a lower mean for positive pairs (or higher mean for negatives) with low variance would mean that the metric is confidently wrong.

We must write code to rank the metrics $cal(M)$ according to the mean and variance of $M$.
The recipe for sampling $M$ is given in @eq:sample_metric, and an example list of experiments is shown in @table:experiments.

#figure(
    table(
      columns: (auto, auto, auto, auto),
      inset: 10pt,
      align: horizon,
      table.header(
        [$cal(M)$: Metric], [$bb(P)(x)$: Source dataset], [$cal(T)$: Translation model],[$bb(P)(hat(x) | x, lambda)$: Corruption]
      ),
      "BLASER 1.0",
      "French",
      [French $arrow$ English],
      [Some LLM],
      "BLASER 2.0",
      "French",
      [French $arrow$ English],
      [Some LLM],
      "BLEU",
      [...],
      [],
      []
    ),
    caption: [Example table of experiments.],
) <table:experiments>


#bibliography("bibliography.bib")
