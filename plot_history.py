import os
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Plots the composition model training history (since the model is trained in multiple checkpoints)
    with open(os.path.join(os.getcwd(), "History.txt"), "r") as f:
        lines = f.readlines()
        f.close()
    loss = [None]
    note_outputs_loss = [None]
    duration_outputs_loss = [None]
    for i in range(len(lines)):
        if i % 2 == 0:
            continue
        if lines[i] == "\n" or len(lines[i]) <= 3:
            break
        split_line = lines[i].split(" ")
        loss.append(float(split_line[7]))
        note_outputs_loss.append(float(split_line[10]))
        duration_outputs_loss.append(float(split_line[13].strip()))
    plt.plot(loss, label="loss")
    plt.plot(note_outputs_loss, label="note_outputs_loss")
    plt.plot(duration_outputs_loss, label="duration_outputs_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs. Epoch")
    plt.ylim(0, 6.5)
    for i in range(0, len(loss), 50):
        if i == 0:
            i += 1
        plt.text(i, loss[i], "{:.2f}".format(loss[i]), ha="center", va="bottom", fontsize=8)
        plt.text(i, note_outputs_loss[i], "{:.2f}".format(note_outputs_loss[i]), ha="center", va="bottom", fontsize=8)
        offset = 0.2 if i != 1 else 0
        plt.text(i, duration_outputs_loss[i]-offset, "{:.2f}".format(duration_outputs_loss[i]),
                 ha="center", va="bottom", fontsize=8)
    plt.legend()
    plt.show()
    # plt.savefig(os.path.join(os.getcwd(), "Images/tenor_intro_model_history.png"))
    pass
