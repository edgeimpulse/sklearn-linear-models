# Custom Scikit Learn linear model ML block examples for Edge Impulse

Documentation on the inner workings of these models is found on scikit-learns website [here](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model).

As a primer, read the [Custom learning blocks](https://docs.edgeimpulse.com/docs/edge-impulse-studio/learning-blocks/adding-custom-learning-blocks) page in the Edge Impulse docs and see another example [here](https://github.com/edgeimpulse/example-custom-ml-block-scikit) which also shows how to test the block locally.


### Pushing the block to Edge Impulse

1. Install the [Edge Impulse CLI](https://docs.edgeimpulse.com/docs/edge-impulse-cli/cli-installation) v1.16.0 or higher.
2. Navigate to the directory with the linear model you want to push to edge impulse.
3. Initialize the block:

    ```
    $ edge-impulse-blocks init
    # Answer the questions, select "Classification" or "Regression" based on the block you wish to install for 'What type of data does this model operate on?'
    ```
4. Push the block:

    ```
    $ edge-impulse-blocks push
    ```
5. The block is now available under any of your projects. Depending on the data your block operates on, you can add it via:
    * Classification: **Create impulse > Add learning block > Classification**, then select the block via 'Add an extra layer' on the 'Classifier' page.
    * Regression: **Create impulse > Add learning block > Regression**, then select the block via 'Add an extra layer' on the 'Regression' page.
    
    Or you can select the block on the "Impulse design" page


If you wish to change any of the model training hyperparameters such as alpha for Ridge and RidgeClassifier you can change them in the `train.py` script and then push the block to your Edge Impulse account.