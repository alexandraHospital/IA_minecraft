from pathlib import Path
import matplotlib.pyplot as plt
import os




# def walk_through_dir(dir_path):
#     for dirpath, dirnames, filenames in os.walk(dir_path):
#         logger.info(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


def plot_loss_curves(model_results, out_plot_name):
  
    results = dict(list(model_results.items()))
    out = Path("output/plots")
    out.mkdir(parents=True, exist_ok=True)

    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();
    
    plt.savefig(f"{out}/{out_plot_name}")
    plt.close()
    