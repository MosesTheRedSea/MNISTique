
import matplotlib.pyplot as plt

def plot_curves(train_loss_history, train_acc_history, valid_loss_history, valid_acc_history):
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss_history, label='train', color='blue')
    plt.plot(valid_loss_history, label='valid', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_curve.png')
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(train_acc_history, label='train', color='blue')
    plt.plot(valid_acc_history, label='valid', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('accuracy_curve.png')
    plt.close()
