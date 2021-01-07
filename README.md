Generative Adversarial Nets for Social Scientists
================
Marcel Neunhoeffer
November 04, 2020

## Running on Windows

On Windows you might encounter the following error when first using
torch:

> `Error in cpp_lantern_init(normalizePath(install_path())):
> C:\Users\User\Documents\R\R-4.0.2\library\torch\deps\lantern.dll - The
> specified module could not be found.`

You can find instructions on how to fix this
[here](https://torch.mlverse.org/docs/articles/installation.html).

## Introduction to Generative Adversarial Nets

You can find an introduction to Generative Adversarial Nets (GANs) in my
slides
[here](https://github.com/SocialScienceDataLab/generative-adversarial-nets/blob/main/SSDL_GANs.pdf).

And you can find a video of the talk
[here](https://youtu.be/KVJ1rVW53Wk).

The following code implements a simple GAN.

## The data

Let’s start with real data that we want to copy using a GAN.

``` r
a <- rnorm(1000)

b <- rnorm(1000, a^2, 0.3)

train_samples <- cbind(a, b)

plot(
      train_samples,
      bty = "n",
      col = viridis(2, alpha = 0.7)[1],
      pch = 19,
      xlab = "Var 1",
      ylab = "Var 2",
      main = "The Data",
      las = 1
    )
```

![](ssdl_GAN_files/figure-gfm/Setting%20up%20real%20data-1.png)<!-- -->

## Setting up the Neural Nets for the GAN with torch in R

The following code will rely on the torch for R package. You can find an
excellent introduction at [the RStudio
mlverse](https://torch.mlverse.org/).

In a GAN we need two competing neural networks – a Generator network and
a Discriminator network. The Generator is supposed to produce real
looking output from random noise. The Discriminator should distinguish
real and fake examples.

``` r
# First, we check whether a compatible GPU is available for computation.
use_cuda <- torch::cuda_is_available()

# If so we would use it to speed up training.
device = ifelse(use_cuda, "cuda", "cpu")

# The Generator Network will contain so called residual blocks. These pass the output and the input of a layer to the next layer
ResidualBlock <- nn_module(
  initialize = function(i, o) {
    # We will use a fully connected (fc) linear layer
    self$fc <- nn_linear(i, o)
    # Followed by a leakyReLU activation.
    self$leaky_relu <- nn_leaky_relu()
  },
  forward = function(input) {
    # A forward pass will take the input and pass it through the linear layer
    out <- self$fc(input)
    # Then on each element of the output apply the leaky_relu activation
    out <- self$leaky_relu(out)
    # To pass the input through as well we concatenate (cat) the out and input tensor.
    torch_cat(list(out, input), dim = 2)
  }
)

# Now we can define the architecture for the Generator as a nn_module.
Generator <- nn_module(
  initialize = function(noise_dim, # The length of our noise vector per example
                        data_dim, # The number of columns in our data
                        hidden_units = list(128, 128), # A list with the number of neurons per layer. If you add more elements to the list you create a deeper network.
                        dropout_rate = 0.5 # The dropout probability
                        ) {
    # Initialize an empty nn_sequential module
    self$seq <- nn_sequential()
   
    # For the hidden layers we need to keep track of our input and output dimensions. The first input will be our noise vector, therefore, it will be noise_dim
    dim <- noise_dim
    
    # i will be a simple counter to keep track of our network depth
    i <- 1
    
    # Now we loop over the list of hidden units and add the hidden layers to the nn_sequential module
    for (neurons in hidden_units) {
      # First, we add a ResidualBlock of the respective size.
      self$seq$add_module(module =  ResidualBlock(dim, neurons),
                          name = paste0("ResBlock_", i))
      # And then a Dropout layer.
      self$seq$add_module(module = nn_dropout(dropout_rate),
                          name = paste0("Dropout_", i))
      # Now we update our dim for the next hidden layer.
      # Since it will be another ResidualBlock the input dimension will be dim+neurons
      dim <- dim + neurons
      # Update the counter
      i <- i + 1
    }
    # Finally, we add the output layer. The output dimension must be the same as our data dimension (data_dim).
    self$seq$add_module(module = nn_linear(dim, data_dim), 
                        name = "Output")
  },
  forward = function(input) {
    input <- self$seq(input)
    input
  }
)


# And we can define the architecture for the Discriminator as a nn_module.
Discriminator <- nn_module(
  initialize = function(data_dim, # The number of columns in our data
                        hidden_units = list(128, 128), # A list with the number of neurons per layer. If you add more elements to the list you create a deeper network.
                        dropout_rate = 0.5 # The dropout probability
                        ) {

    # Initialize an empty nn_sequential module
    self$seq <- nn_sequential()
    
     # For the hidden layers we need to keep track of our input and output dimensions. The first input will be our noise vector, therefore, it will be noise_dim
    dim <- data_dim
    
    # i will be a simple counter to keep track of our network depth
    i <- 1
    
     # Now we loop over the list of hidden units and add the hidden layers to the nn_sequential module
    for (neurons in hidden_units) {
      # We start with a fully connected linear layer
      self$seq$add_module(module = nn_linear(dim, neurons),
                          name = paste0("Linear_", i))
      # Add a leakyReLU activation
      self$seq$add_module(module = nn_leaky_relu(), 
                          name = paste0("Activation_", i))
      # And a Dropout layer
      self$seq$add_module(module = nn_dropout(dropout_rate),
                          name = paste0("Dropout_", i))
      # Update the input dimension to the next layer
      dim <- neurons
      # Update the counter
      i <- i + 1
    }
    # Add an output layer to the net. Since it will be one score for each example we only need a dimension of 1.
    self$seq$add_module(module = nn_linear(dim, 1), 
                        name = "Output")
    
  },
  forward = function(input) {
    data <- self$seq(input)
    data
  }
)


# We will use the kl GAN loss
# You can find the paper here: https://arxiv.org/abs/1910.09779
# And the original python implementation here: https://github.com/ermongroup/f-wgan

kl_real <- function(dis_real) {
  loss_real <- torch_mean(nnf_relu(1 - dis_real))
  
  return(loss_real)
}

kl_fake <- function(dis_fake) {
  dis_fake_norm = torch_exp(dis_fake)$mean()
  dis_fake_ratio = torch_exp(dis_fake) / dis_fake_norm
  dis_fake = dis_fake * dis_fake_ratio
  loss_fake = torch_mean(nnf_relu(1. + dis_fake))
  
  return(loss_fake)
}

kl_gen <- function(dis_fake) {
  dis_fake_norm = torch_exp(dis_fake)$mean()
  dis_fake_ratio = torch_exp(dis_fake) / dis_fake_norm
  dis_fake = dis_fake * dis_fake_ratio
  loss = -torch_mean(dis_fake)
  return(loss)
}
```

## Initializing the Networks

So far we only created functions to help us to set up our networks. To
train them we need to create and initialize actual networks.

``` r
# We need to pass our noise_dim and data_dim to create concrete networks
noise_dim <- 2
data_dim <- ncol(train_samples)

# Now, we can set up a Generator net and send it to our device (cpu or gpu)
g_net <-
  Generator(noise_dim, data_dim)$to(device = device)

# Get a quick overview of how many parameters our network has
g_net
```

    ## An `nn_module` containing 17,670 parameters.
    ## 
    ## ── Modules ───────────────────────────────────────────────────────────────────────
    ## ● seq: <nn_sequential> #17,670 parameters

``` r
# To update the parameters of the network we need setup an optimizer. Here we use the adam optimizer with a learning rate of 0.0002
g_optim <- optim_adam(g_net$parameters, lr = 0.0002)

# Now, we also need a Discriminator net.
d_net <-
  Discriminator(data_dim = ncol(train_samples))$to(device = device)

#To update the parameters of the network we need setup an optimizer. Here we use the adam optimizer with a learning rate of 0.0002 * 4
# This heuristic comes from the idea of using two time-scales (aka different learning rates) for the Generator and Discriminator. You can find more in this paper: https://arxiv.org/abs/1706.08500
d_optim <- optim_adam(d_net$parameters, lr = 0.0002 * 4)
```

## Setting up the data for torch

Torch needs the data in the form of tensors on the device of your model.
(More on that in the introduction: <https://torch.mlverse.org/>).

It is easy to transform an R matrix/array to a torch tensor by using
`torch_tensor`.

``` r
# We need our real data in a torch tensor
x <-
  torch_tensor(train_samples)$to(device = device)

# To observe training we will also create one fixed noise data frame.
# # torch_randn creates a torch object filled with draws from a standard normal distribution
fixed_z <-
  torch_randn(c(nrow(train_samples), noise_dim))$to(device = device)
```

Next we write a simple function to sample synthetic data from the GAN
and get it back as a R array/matrix.

``` r
sample_synthetic_data <-
  function(z,
           device
           ) {
    # Pass the noise through the Generator to create fake data
    fake_data <-  g_net(z)
    
    # Create an R array/matrix from the torch_tensor
    synth_data <- as_array(fake_data$detach()$cpu())
    return(synth_data)
  }
```

## The Training Loop

We will iteratively update both networks. Before we can get started with
our training loop, we need to set some training hyperparameters.

``` r
# Batch Size: How many samples do we use per update step?
batch_size <- 50

# Steps: How many steps do we need to make before we see the entire data set (on average).
steps <- nrow(train_samples) %/% batch_size

# Epochs: How many times (on average) do we want to pass the entire data set.
epochs <- 100

# Iters: What's the total number of update steps?
iters <- steps * epochs
```

Then, we can get started with training our GAN.

``` r
for (i in 1:(epochs * steps)) {
  
  ###########################
  # Sample Batch of Data
  ###########################
  
  # For each training iteration we need a fresh (mini-)batch from our data.
  # So we first sample random IDs from our data set.
  batch_idx <- sample(nrow(train_samples), size = batch_size)
  
  # Then we subset the data set (x is the torch version of the data) to our fresh batch.
  real_data <- x[batch_idx]$to(device = device)
  
  ###########################
  # Update the Discriminator
  ###########################
  
  # In a GAN we also need a noise sample for each training iteration.
  # torch_randn creates a torch object filled with draws from a standard normal distribution
  z = torch_randn(c(batch_size, noise_dim))$to(device = device)
  
  # Now our Generator net produces fake data based on the noise sample.
  # Since we want to update the Discriminator, we do not need to calculate the gradients of the Generator net.
  fake_data <- with_no_grad(g_net(input = z))
  
  # The Discriminator net now computes the scores for fake and real data
  dis_real <- d_net(real_data)
  dis_fake <- d_net(fake_data)
  
  # We combine these scores to give our discriminator loss
  d_loss <- kl_real(dis_real) + kl_fake(dis_fake)
  d_loss <- d_loss$mean()
  
  # What follows is one update step for the Discriminator net
  
  # First set all previous gradients to zero
  d_optim$zero_grad()
  
  # Pass the loss backward through the net
  d_loss$backward()
  
  # Take one step of the optimizer
  d_optim$step()
  
  ###########################
  # Update the Generator
  ###########################
  
  # To update the Generator we will use a fresh noise sample.
  # torch_randn creates a torch object filled with draws from a standard normal distribution
  z <- torch_randn(c(batch_size, noise_dim))$to(device = device)
  
  # Now we can produce new fake data
  fake_data <- g_net(z)
  
  # The Discriminator now scores the new fake data
  dis_fake <- d_net(fake_data)
  
  # Now we can calculate the Generator loss
  g_loss = kl_gen(dis_fake)
  g_loss = g_loss$mean()
  
  # And take an update step of the Generator
  
  # First set all previous gradients to zero
  g_optim$zero_grad()
  
  # Pass the loss backward through the net
  g_loss$backward()
  
  # Take one step of the optimizer
  g_optim$step()
  
  # This concludes one update step of the GAN. We will now repeat this many times.
  
  ###########################
  # Monitor Training Progress
  ###########################
  
  # During training we want to observe whether the GAN is learning anything useful.
  # Here we will create a simple message to the console and a plot after each epoch. That is when i %% steps == 0.
  
  if (i %% steps == 0) {
    # Print the current epoch to the console.
    cat("Epoch: ", i %/% steps, "\n")
    
    # Create synthetic data for our plot. This synthetic data will always use the same noise sample -- fixed_z -- so it is easier for us to monitor training progress.
    synth_data <-
      sample_synthetic_data(fixed_z, device)
    # Now we plot the training data.
    plot(
      train_samples,
      bty = "n",
      col = viridis(2, alpha = 0.7)[1],
      pch = 19,
      xlab = "Var 1",
      ylab = "Var 2",
      main = paste0("Epoch: ", i %/% steps),
      las = 1
    )
    # And we add the synthetic data on top.
    points(
      synth_data,
      bty = "n",
      col = viridis(2, alpha = 0.7)[2],
      pch = 19
    )
    # Finally a legend to understand the plot.
    legend(
      "topleft",
      bty = "n",
      pch = 19,
      col = viridis(2),
      legend = c("Real", "Synthetic")
    )
  }
}
```

How did our training do?

## Look at our Synthetic Data

Finally, we can sample synthetic data from our trained GAN. Since we
used dropout in the generator we could even pass the same noise through
the net and would get slightly different synthetic data points. Dropout
is not only a form of training regularization, but can also be used to
assess the uncertainty in our training (see:
<https://arxiv.org/abs/1506.02142>). For the plot we will use fresh
noise for each draw.

``` r
# First plot the training data
plot(
  train_samples,
  bty = "n",
  col = viridis(2, alpha = 0.7)[1],
  pch = 19,
  xlab = "Var 1",
  ylab = "Var 2",
  main = "Real and Synthetic Data"
)
# Now we take 100 draws of synthetic data from our generator
for (i in 1:100) {
  # Fresh noise z
  z <- torch_randn(c(nrow(train_samples), noise_dim))$to(device = device)
  # Sample synthetic data
  synth_data <-
    sample_synthetic_data(z, device)
  # Plot it with alpha to "make the uncertainty" visible
  points(
    synth_data,
    bty = "n",
    col = viridis(2, alpha = 0.02)[2],
    pch = 19
  )
}
# Add a legend
legend(
  "topleft",
  bty = "n",
  pch = 19,
  col = viridis(2),
  legend = c("Real", "Synthetic")
)
```

![](ssdl_GAN_files/figure-gfm/Synthetic%20Examples-1.png)<!-- -->

## Summary

In this tutorial we learned how to train a GAN in R with torch. Can you
think of cool applications?
