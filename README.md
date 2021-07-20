# Iris Classification with RedisAI

## Step 0: Setup RedisAI

To use RedisAI, well, you need RedisAI. I've found the easiest way to do this is with Docker. First, pull the redismod image—it contains Redis with several popular modules ready to go:

    $ docker image pull redislabs/redismod

Then run the image:

    $ docker run -p 6379:6379 --name redismod redislabs/redismod

And, you've got RedisAI up and running!

## Step 1: Setup Python Environment

You need a Python environment to make this all work. I used Python 3.9—the latest, greatest, and most updatest at the time of this writing. I also used `venv` to manage my environment.

I'll assume you can download and install Python 3.9 on your own. So lets go ahead and setup the environment:

    $ python3.9 -m venv .venv

Once `venv` is installed, you need to activate it:

    $ . ./.venv/bin/activate

Now when you run `python` from the command line, it will always point to Python3.9 and any libraries you install will only be for this specific environment. Usually, this includes a dated version of pip so go ahead an update that as well:

    $ pip install --upgrade pip


If you want to deactivate this environment, you can do so from anywhere with the following command:

    $ deactivate

## Step 2: Install Dependencies

Next, let's install all the dependencies. These are all listed in `requirements.txt` and can be installed with `pip` like this.

    $ pip install -r requirements.txt

Run that command, and you'll have all the dependencies installed and will be ready to run the code.

## Step 3: Build the TorchScript Model

Load and train a Sklearn LogisticRegression model using the Iris Data Set. Use Microsoft's Hummingbird.ml to convert the Sklearn model into a TorchScript model for loading into RedisAI. Run the `build.py` Python script to generate the `iris.pt` model file:

    $ python build.py

## Step 4: Deploy the Model into RedisAI

NOTE: This requires redis-cli. If you don't have redis-cli, I've found the easiest way to get it is to download, build, and install Redis itself. Details can be found at the [Redis quickstart](https://redis.io/topics/quickstart) page:

    $ redis-cli -x AI.MODELSTORE iris TORCH CPU BLOB < iris.pt
    OK

## Step 5: Make Some Predictions

Launch redis-cli:

    $ redis-cli

Set the input tensor with 2 sets of inputs of 4 values each:

    > AI.TENSORSET iris:in FLOAT 2 4 VALUES 5.0 3.4 1.6 0.4 6.0 2.2 5.0 1.5

Make the predictions (inferences) by executing the model:

    > AI.MODELEXECUTE iris INPUTS 1 iris:in OUTPUTS 2 iris:inferences iris:scores

Check the predictions:

    > AI.TENSORGET iris:inferences VALUES
    1) (integer) 0
    2) (integer) 2

Check the scores:

    > AI.TENSORGET iris:scores VALUES
    1) "0.96567678451538086"
    2) "0.034322910010814667"
    3) "3.4662525649764575e-07"
    4) "0.00066925224382430315"
    5) "0.45369619131088257"
    6) "0.54563456773757935"

### References

* https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
* https://pytorch.org
* https://pytorch.org/docs/stable/jit.html
* https://microsoft.github.io/hummingbird/
* https://github.com/microsoft/hummingbird