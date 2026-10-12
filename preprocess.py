from utils.dataloader import get_dataloader

loader = get_dataloader()

for images, labels in loader:

    print("Image Shape :", images.shape)
    print("Label Shape :", labels.shape)

    break