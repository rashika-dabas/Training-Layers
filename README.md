# Ex-6: Trainable Layers in Neural Networks

## Overview
In this exercise, you will learn to create and modify neural network layers.

## Objectives
1. Define dense models with multiple hidden layers.
2. Set specific layers as trainable or non-trainable.

## The Assignment

### define_dense_model_with_hidden_layers
Modify your solution from the previous exercise for `define_dense_model_with_hidden_layers`. This function should accept an array of hidden layer sizes and an array of the activation functions. Keep the following points in mind when coding:

* The first layer is different because it should accept an `input_layer` parameter.
* The output layer should use the `output_function` and `output_length` parameters.

### set_layers_to_trainable
The function `set_layers_to_trainable` should take a model and an array of layer numbers. It should set the layers in the array to `trainable` and the other layers to non-trainable.

---

## Validating and Evaluating Your Results

### Online
1. After committing and pushing your code, check the mark on the top line (near the commit ID).
2. If some tests are failing, click on the ❌ to open up a popup, which will show details about the errors.
3. You can click the [Details]() link to see what went wrong. Pay special attention to lines with the words "Failed" or "error".

![screnshot](images/details_screenshot.png)

4. Near the bottom of the [Details]() page, you can see your score. Here are examples of 0/5 and 5/5:

![score](images/score.png) ![success](images/success.png)

5. When you achieve a perfect score, you will see a green checkmark near the commit ID.

![green](images/green.png)

### Locally
1. You can test your code locally by installing and running `pytest` (`pip install pytest` or `conda install pytest`).
2. Run the tests using the command `pytest` in your terminal. This will show the status of each test and any errors that occurred.

Good luck!