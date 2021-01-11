# 100DaysOfMLCode

# Introduction

In this repository there will be my code and progress 
into the 100DaysofMLCode Challenge which I started from
this day of 17th December of 2020. I will also write about
my progress into my [Twitter](https://twitter.com/EmmanuelLykos) account

# Progress

## Day 1

I set a goal to learn to make myself familiar with PyTorch
framework after I learned Keras, so today I made myself 
familiar with tensor handling, I implemented R-Squared using tensors
and I took a look at creating a simple neural network through the 
60-Minute Blitz.

## Day 2

I continued  PyTorch 60-Minute Blitz and I finally implemented 
the CIFAR10 classifier(with 50% accuracy) and an MNIST 
Classifier(around 97% accuracy) using ConvNets. Moreover, 
I created my own transform.

## Day 3

Today I wanted to deal with the practical part of Reinforcement 
Learning(RL), thus, I continued the [Easy21](https://www.davidsilver.uk/wp-content/uploads/2020/03/Easy21-Johannes.pdf)
Project of David Silver's 
RL course and I implemented the Sarsa(λ) algorithm. 
I will upload the code when I finish the question.

## Day 4

I completed the TD-Learning question of Easy 21 Project by implementing Sarsa(λ) algorithm
and drawing the necessary graphs. However might it seem little, yesterday I wrote the
Forward view Sarsa(λ) while the project asking to implement the backward view. Tomorrow
I will probably finish the question about Value Function Approximation, thus, all the Easy21
project and it will be uploaded also to a separate repo.

## Day 5

I completed the coding on Easy21 Project. The code is on the corresponding
directory on 100DaysMLOfCode repo, but I will upload it on another repo
where I will keep all my RL Projects and the code will be better

## Day 6

Today I made myself familiar with OpenAI
 Gym and I tried to implement an RL Algorithm that could solve the CartPole problem. 
But, MC and Semi-Gradient Sarsa with linear approximation did not work. I believe that
this happens because my features are not good because I cannot find another reason that
neither bootrapping and not bootstrapping do not work. 
The fixed code will be uploaded in GitHub on another day.

## Day 7

I tried to see why my code about solving CartPole was not working. I fixed some bugs
that I had however, the results were not encouraging at all because still my agent
does not learn. I tried the following adjustments:

1. Having features for Q(S,A) the observation and the number of action was wrong
thus I did one-hot encoded the action feature but this did not work neither having
4 weights for each action worked and the features will be the observations.

2. Normalizing the weights at each step.

3. Rounding the features.

4. Decaying the policy by a constant factor but I saw that exploration did not helped at all,
because when epsilon became low the agent did not seem to exploit the knowledge that he
obtained through exploration.

I will upload my code only for someone to see the implemented algorithms.

## Day 8

Back in action!!! Today, I started working on Berkeley's Pacman Project 3 and I solved the first 3 questions!!!

## Day 9 

Today I continued UC Berkeley's Pacman Project 3 and I reached  Question 7 where I still have to figure out why I get a Timeout exception at that question.
