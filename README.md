![Logo](logo.png)

# Stable Diffusion cloud training tutorial
> Train Stable Diffusion (textual inversion) with your own images

This repository contains tutorials to train your own Stable Diffusion `.ckpt`
model using Google Cloud Platform (GCP) and Amazon Web Services (AWS).

## Getting started

For platform specific instructions, see the following:

* [Google Cloud Platform (GCP)](GCP.md)
* [Amazon Web Services (AWS)](AWS.md)

## Cost estimate

It's very cheap to train a Stable Diffusion model on GCP or AWS. Prepare to
spend $5-10 of your own money to fully set up the training environment and to
train a model.

As a comparison, my total budget at GCP is now at $14, although I've been
playing with it a lot (including figuring out how to deploy it in the first
place).

The spot price of A100 instance at GCP is under $2 per hour, and it takes under
half an hour to train 500 steps. Smallest P3 instance is at ~$3/hour, and you
should be able to train 500 steps in an hour.

### On limits

Note that both AWS and GCP have limits on the number of GPU instances you can
run if you have a new account. You'll need to follow their instructions to
request a limit increase.

On my own testing it took 2 minutes for GCP to increase my limit from 0 to 1,
while it took 7 days to do the same for AWS

## Licensing

This project is licensed under MIT license.
