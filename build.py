from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from hummingbird.ml import convert
from zipfile import ZipFile
import shutil
import os

TORCH_FILE = 'iris.torch'
TORCH_ARCHIVE = f'{TORCH_FILE}.zip' # the output of torch model save()
TORCHSCRIPT_BLOB_SRC = 'deploy_model.zip' # internal (in zip) torchscript blob
TORCHSCRIPT_BLOB_DEST = 'iris.pt' # output name for extracted torchscript blob

# prepare the train and test data
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

# train the model - using logistic regression classifier
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# use hummingbird.ml to convert sklearn model to torchscript model (torch.jit backend)
torch_model = convert(model, 'torch.jit', test_input=X_train, extra_config={})

# save the model
torch_model.save(TORCH_FILE)

# extract the TorchScript binary payload
with ZipFile(TORCH_ARCHIVE) as z:
    with z.open(TORCHSCRIPT_BLOB_SRC) as zf, open(TORCHSCRIPT_BLOB_DEST, 'wb') as f:
        shutil.copyfileobj(zf, f)

# clean up - remove the zip file
if os.path.exists(TORCH_ARCHIVE):
    os.remove(TORCH_ARCHIVE)
