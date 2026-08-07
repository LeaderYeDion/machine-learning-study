import torch
import matplotlib.pyplot as plt
from torch import nn
from pathlib import Path


class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        # model training starts with a group of random data and updates
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float32))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weights + self.bias

start = 0
end = 1
step = 0.02
# PyTorch 在 Apple 芯片上通过 MPS（Metal Performance Shaders）使用 GPU
X = torch.arange(start, end, step, device="cpu").unsqueeze(1)

# liner regression formula: y = a + bX. b symbols for weights, a symbols for bias
weights = 0.7
bias = 0.3
y = bias + weights * X

# split the input into training set and test set
train_split = int(len(X) * 0.8)
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]


def do_model_training(model_1:nn.Module, draw_plot=True):
    epochs = 1000
    torch.manual_seed(10)
    # model_1 = LinearRegression()
    print(f"""actual weights: {weights}, actual bias: {bias}.""")
    print(f"""model 1: {model_1.state_dict()}""")
    # create loss function and optimizer for our linear regression
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.SGD(model_1.parameters(), lr=0.01)

    for epoch in range(epochs):
        # 1. switch into training mode
        model_1.train()

        # 2. forward pass the training data
        y_pred = model_1(X_train)

        # 3. calculate the loss
        loss = loss_fn(y_pred, y_train)
        # print(f"""loss: {loss}""")

        # 4. set gradients to zero to prevent from accumulation
        optimizer.zero_grad()

        # 5. perform backpropagation on loss
        loss.backward()

        # 6. step the optimizer
        optimizer.step()

    print(f"""new model_1: {model_1.state_dict()}""")

    # --- now test the model

    # 7. switch off all unnecessary configurations during evaluation procedure
    model_1.eval()
    with torch.no_grad():
        y_pred = model_1(X_test)
        loss = loss_fn(y_pred, y_test)
        print(f"""test loss: {loss}""")

        if draw_plot:
            plot_predictions(X_train, y_train, X_test, y_test, predictions=y_pred)


def do_liner_regression():
    """
    in machin learning, data are split into three kinds of datasets:
    1. training data, 60% - 80% of all
    2. validation data (optional), 10% - 20% of all
    3. test data, 10% - 20% of all
    """

    torch.manual_seed(42)
    model_0 = LinearRegression()

    with torch.inference_mode():
        # equals to model_0.__call__()
        y_pred = model_0(X_test)
        # the same to above
        y_pred = model_0.__call__(X_test)

    plot_predictions(X_train, y_train, X_test, y_test, predictions=y_pred)

def plot_predictions(train_data, train_labels, test_data, test_labels, predictions=None):
    """
    Plots training data, test data and compare predictions
    """

    plt.figure(figsize=(10, 7))

    plt.scatter(train_data, train_labels, c="b", s=4, label="training data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="test data")
    if predictions is not None:
        plt.scatter(predictions, test_labels, c="r", s=4, label="predictions")

    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Linear Regression Data")
    plt.legend()
    plt.show()

MODEL_PATH = Path("./model")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "01_pytorch_workflow.pt"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
def save_model(model_to_save:nn.Module, weights_only=True):
    """
    Saves model to disk
    """
    torch.save(model_to_save.state_dict() if weights_only else model_to_save, MODEL_SAVE_PATH)
    print(MODEL_SAVE_PATH)

def load_model(path_to_load: str, weights_only=True) -> nn.Module | dict:
    """
    Loads model from disk
    """
    return torch.load(path_to_load, weights_only=weights_only)

if __name__ == "__main__":
    # 1. training model and draw a picture to visualize
    model_1 = LinearRegression()
    do_model_training(model_1, False)

    persis_weights_only = True

    # 2. saving the model as binary to disk for future loading
    save_model(model_1, persis_weights_only)

    # 3. loading existed model state dict from disk to use
    loaded = load_model(str(MODEL_SAVE_PATH), persis_weights_only)

    if persis_weights_only:
        loaded_model = LinearRegression()
        loaded_model.load_state_dict(loaded)
    else:
        loaded_model = loaded

    print(loaded_model)
    print(loaded_model.state_dict())

    print(f"""result from loaded: {loaded_model(X_test)}""")
    print(f"""expected: {y_test}""")
