import yaml
import copy
import argparse
from two_layer_nn import TwoLayerNeuralNet
from softmax_regression import SoftmaxRegression
from optimizer import SGD
from common import utils, plot_utils, load_mnist_trainval, load_mnist_test, generate_batched_data, train, evaluate, plot_curves

def train_model(yaml_config_file):
  args = {}
  with open(yaml_config_file) as f:
      config = yaml.full_load(f)

  for key in config:
      for k, v in config[key].items():
          args[k] = v

  # Prepare MNIST data

  train_data, train_label, val_data, val_label = load_mnist_trainval()
  test_data, test_label = load_mnist_test()

  # Prepare model and optimizer

  if args["type"] == 'SoftmaxRegression':
      model = SoftmaxRegression()
  elif args["type"] == 'TwoLayerNet':
      model = TwoLayerNeuralNet(hidden_size=args["hidden_size"])
  optimizer = SGD(learning_rate=args["learning_rate"], reg=args["reg"])

  # Training Code

  train_loss_history = []
  train_acc_history = []
  valid_loss_history = []
  valid_acc_history = []
  best_acc = 0.0
  best_model = None
  for epoch in range(args["epochs"]):
      batched_train_data, batched_train_label = generate_batched_data(train_data, train_label, batch_size=args["batch_size"], shuffle=True)
      epoch_loss, epoch_acc = train(epoch, batched_train_data, batched_train_label, model, optimizer, args["debug"])

      train_loss_history.append(epoch_loss)
      train_acc_history.append(epoch_acc)
      # evaluate on test data
      batched_test_data, batched_test_label = generate_batched_data(val_data, val_label, batch_size=args["batch_size"])
      valid_loss, valid_acc = evaluate(batched_test_data, batched_test_label, model, args["debug"])
      if args["debug"]:
          print("* Validation Accuracy: {accuracy:.4f}".format(accuracy=valid_acc))

      valid_loss_history.append(valid_loss)
      valid_acc_history.append(valid_acc)

      if valid_acc > best_acc:
          best_acc = valid_acc
          best_model = copy.deepcopy(model)

  # Testing Code
  
  batched_test_data, batched_test_label = generate_batched_data(test_data, test_label, batch_size=args["batch_size"])
  _, test_acc = evaluate(batched_test_data, batched_test_label, best_model)
  if args["debug"]:
      print("Final Accuracy on Test Data: {accuracy:.4f}".format(accuracy=test_acc))

  return train_loss_history, train_acc_history, valid_loss_history, valid_acc_history

def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["softmax", "twolayer"],
        required=True,
    )

    args = parser.parse_args()

    config_files = {
        "softmax": "MNISTique/experiments/config_softmax.yaml",
        "twolayer": "MNISTique/experiments/config_twolayer.yaml"
    }

    config_file = config_files[args.model]

    # Train the model
    train_loss_history, train_acc_history, valid_loss_history, valid_acc_history = train_model(config_file)

    # Plot results
    plot_curves(train_loss_history, train_acc_history, valid_loss_history, valid_acc_history)


if __name__ == "__main__":

    initialize()