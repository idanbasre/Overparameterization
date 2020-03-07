import json
import pandas
import csv
import matplotlib.pyplot as plt


def create_architecture_description(layers_width, P, number_of_epochs, lr):
    # create a string that holds net architectur
    if len(layers_width) - 1 == 1:
        net_arch_str = ''
    else:
        net_arch_str = 'width_'
        for cl in layers_width[1:-1]:
            net_arch_str = ''.join([net_arch_str, '{}_'.format(cl)])

    decription = "{}_layer_{}L{}_{}_epochs_lr_{}_".format(len(layers_width)-1, net_arch_str, P, number_of_epochs, lr)
    return decription


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
    
    accuracies = []
    for i, result in enumerate(results):
      accuracies.append([])
      accuracies[i].append([])
      for epoch_result in result:
        accuracies[i][0].append(epoch_result["train loss"])

    for i, accuracy in enumerate(accuracies):
        train_acc = accuracy[0]

        plt.figure(i)
        plt.plot(range(1, NUMBER_OF_EPOCHS+1),train_acc)
        plt.title(" Loss (blue-train)")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.grid()
        plt.ylim((0, 1))

        plt.show()
