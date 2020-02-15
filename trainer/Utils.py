import json
import pandas
import csv
import matplotlib.pyplot as plt


def save_results(results, file_path):
    pandas.DataFrame.from_dict(
            results, 
            orient = 'columns',
     ).to_csv(f'{file_path}.csv')

    with open(f'{file_path}.json', 'w', encoding='utf-8') as fd:
         json.dump(results, fd, ensure_ascii=False, indent=4)


def plot_convergence_graph(results, legend=None):
    if not legend:
        legend = range(len(results))
    
    losses = []
    for i, result in enumerate(results):
      losses.append([])
      losses[i].append([])
      losses[i].append([])
      for epoch_result in result:
        losses[i][0].append(epoch_result["train loss"])
        losses[i][1].append(epoch_result["test loss"])

    for i, losses in enumerate(losses):
        train_loss, test_loss = losses

        plt.figure(i)
        plt.plot(range(1, NUMBER_OF_EPOCHS+1),train_loss)
        plt.plot(range(1, NUMBER_OF_EPOCHS+1),test_loss, color='r')
        plt.title(legend[i] + " Loss (blue-train, red-test)")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.grid()

        plt.show()
 

def show_image(images, classes):
    np_images = images.numpy()
    for i, np_img in enumerate(np_images):
        plt.subplot(1, len(np_images), i+1)
        plt.imshow(np_img[0], cmap='gray')
        plt.title(CLASSES[classes[i]])

    plt.show()