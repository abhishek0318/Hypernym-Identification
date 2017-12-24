******************************************************
******************************************************
*                                                    *
*              The BLESS test set                    *
*                                                    *
******************************************************
******************************************************


Sections in this README:

- DESCRIPTION OF THE BLESS TEST SET

- RECOMMENDED EVALUATION PROCEDURE

- LICENSE



### DESCRIPTION OF THE BLESS TEST SET ###

The BLESS (Baroni & Lenci's Evaluation of Semantic Similarity) test
set is a benchmark for computational models that attempt to capture
semantic relations among (concepts expressed by single) words.

The BLESS data are presented in the following tab-delimited format:

- CONCEPT: the target concept holding a certain type of relation with
  the relatum concept

- CLASS: the broader semantic class of the target concept

- RELATION: the type of relation linking the target concept and the
  relatum

- RELATUM: the relatum linked to the target concept

All target concepts are nouns, whereas the relata are nouns, verbs and
adjectives. In the BLESS file, nouns are suffixed with -n, verbs are
suffixed with -v and adjectives, suffixed with -j.

BLESS includes 200 distinct concrete nouns as target concepts. These
concepts can be grouped into 17 broader classes:

- amphibian_reptile [that is: amphibian OR reptile], 5 nouns, e.g.:
  alligator-n

- appliance, 11 nouns, e.g.: toaster-n

- bird, 15 nouns, e.g.: crow-n

- building, 9 nouns, e.g.: cottage-n

- clothing, 12 nouns, e.g.: sweater-n

- container, 6 nouns, e.g.: bottle-n

- fruit, 15 nouns, e.g.: banana-n

- furniture, 9 nouns, e.g.: chair-n

- ground_mammal, 21 nouns, e.g.: beaver-n

- insect, 8 nouns, e.g.: cockroach-n

- musical_instrument, 8 nouns, e.g.: violin-n

- tool [that is: manipulable tool or device], 15 nouns, e.g.: hammer-n

- tree, 9 nouns, e.g.: birch-n

- vegetable, 16 nouns, e.g.: cabbage-n

- vehicle, 18 nouns, e.g.: bus-n

- water_animal [including fish and sea mammals], 11 nouns, e.g.:
  herring-n

- weapon, 12 nouns, e.g.: dagger-n


For each concept noun, BLESS includes several relatum words, linked to
the concept by one of the following 5 relations:

- attri: the relatum is an adjective expressing an attribute of the
		concept, e.g.:
		alligator-n attri aquatic-j

- coord: the relatum is a noun that is a co-hyponym (coordinate) of
		the concept, i.e., they belong to the same (narrowly
		or broadly defined) semantic class, e.g.:
		alligator-n coord lizard-n
				
- event: the relatum is a verb referring to an
		action/activity/happening/event the concept is
		involved in or is performed by/with the concept e.g.:
		alligator-n event swim-v
		
- hyper: the relatum is a noun that is a hypernym of the concept e.g.:
		alligator-n hyper animal-n

- mero: the relatum is a noun referring to a
		part/component/organ/member of the concept, or
		something that the concept contains or is made of
		e.g.: 
		alligator-n mero mouth-n


The relata were selected by the authors using a number of sources,
including WordNet, ConceptNet, Wikipedia and the semantic norms of
McRae and colleagues. Both authors validated the full list, and a
randomly sampled subset of 1,000 concept-relation-relatum tuples were
also validated with Amazon Mechanical Turk via the CrowdFlower
interface.

The relata are expressed in lemma form, but, in order to maximize
coverage, we tried to include spelling variants (e.g., organise-v and
organize-v) and to account for cases where different lemmatizers might
map to different forms (e.g., scissors-n and scissor-n).

For each concept, there are at least 2 relata per relation type. On
average, there are 14 attri, 18 coord, 19 event, 7 hyper and 15 mero
relata per concept. In total, BLESS contains 14,4000
concept-relation-relatum tuples where the relation is among those
listed above.

The BLESS set includes moreover 12,154 control tuples with random
relata (nouns, verbs, and adjectives, with the corresponding relations
encoded as random-n, random-v and random-j, respectively). Random
relata were chosen to approximately match the frequency of occurrence
of the true relata (frequencies computed on the concatenated
WaCkypedia_EN and ukWaC corpora, see below). A series of t-tests
confirmed that there is no significant difference in frequency between
any POS-specific random set and the set of true relata with the same
POS. Each concept is paired on average with 34 random nouns (minimally
16), 11 random adjectives (minimally 3) and 16 random verbs (minimally
4).

All random relata tuples were validated with Amazon Mechanical Turk
(via CrowdFlower). All the random tuples received at least 2 judgments
that excluded the presence of any meaningful relation between the
concept and the relatum.

Counting both true and random relata, BLESS contains in total 26,554
concept-relation-relatum tuples.



### RECOMMENDED EVALUATION PROCEDURE ###

The BLESS data set contains multiple examples of each relation for
each target concept, to account for variations in the specific
words/concepts that instantiate a certain semantic relation. For
example, a model more oriented towards technical terminology might
assign a high similarity to the whale-n/cetacean-n pair, whereas
another model might assign high similarity to
whale-n/animal-n. Despite the difference in specific terms, both
models are picking appropriate hypernyms for whales. In this
perspective, the authors recommend computing the cosine (or other
similarity measure) of the target concepts with each relatum and
picking, for each set of true and random relations, the nearest
neighbours to each concept. A comparison across relation types can
then be performed by looking at the similarity of these nearest
neighbours across concepts.

More specifically, we suggest the following evaluation procedure:

- compute the similarity of each concept with all its relata and pick
  the closest relatum per relation;

- in order to account for concept-specific effects (some concepts
  might have sparser neighbourhoods than others), normalize the 8
  similarity scores associated to each target concept by converting
  them to z scores (subtract their average and divide by their
  standard deviation);

- summarize the distribution of similarities across relations by
  plotting the z scores grouped by relations (200 scores for each
  relation) in a box-and-whisker plot;

- verify the statistical significance of the differences in
  similarities across relations by performing a Tukey Honestly
  Significant Difference test (to account for the multiple pairwise
  comparisons to be performed)

Note that z scaling, plotting and statistical testing can easily be
performed using the free R statistical package
(http://www.r-project.org/).

All the concepts and nearly all the relata in BLESS are attested in
the ukWaC and WaCkypedia_EN corpora that are freely available from
http://wacky.sslmit.unibo.it/ (all concepts have at least 2 relata per
relation type -- typically many more -- that are attested with
frequency of at least 100 occurrences in the concatenated ukWaC/
WaCkypedia_EN). These corpora come with lemma, POS and dependency
information. If your model can be trained on any corpus, we strongly
recommend you use the concatenated ukWaC/WaCkypedia_EN, for optimal
coverage and comparability.



### LICENSE ###

Copyright 2011, Marco Baroni and Alessandro Lenci.

The BLESS data set is released under the Creative Commons
Attribution-ShareAlike license. In short, you are free to copy,
distribute, transmit and edit the data set, as long as you credit the
original authors (Marco Baroni and Alessandro Lenci) and, if you
choose to distribute the resulting work, you release it under the same
license.

For more information, see:

http://creativecommons.org/licenses/by-sa/3.0/

