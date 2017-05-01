# Milestone 4 README

## Connecting to and Setting Up AWS

1. Use the *cs109b-Group24AMI* and make sure that you have at least 64GB of memory on the instance. This AMI comes loaded with the relevant packages as well as all of the posters we use.
2. Transfer the *train_w_poster.csv*, *test_w_poster.csv*, *run_model.py*, and *convert_images_to_array.py* to the main directory of the instance
3. Create a *logs* folder from the main directory and a *model_0* subfolder.
```
mkdir logs
cd logs
mkdir model_0
```
4. Create a predictions folder from the main directory. `mkdir predictions`

## Creating Data Files

You need convert the posters and metadata into numpy arrays that keras / tensorflow can process. To do this, enter the following command in the ssh terminal:
```
python convert_to_keras_arrays.py train_w_poster.csv test_w_poster.csv <img_height> <img_width>
```
You can choose whatever image height and width you want, though I would recommend using 150 and 100 due to RAM limitations on the GPU

## Running the Model

### Non Pre-Trained

Enter the following command in the ssh terminal:
```
python run_model.py train_predictor_image_keras.npy test_predictor_image_keras.npy train_outcome_keras.npy test_outcome_keras.npy <learning_rate> <batch_size> <epochs>
```

You can play around with the last 3 parameters as needed, though I would recommend .001 for the learning rate, 10 for the batch size, and 200 for epochs. There is a learning rate decay function in the model and a early stopping function so it will end way before 200 epochs.

If you want to change the actual structure of the network, edit the *run_model.py* file as necessary on your local machine and transfer it to the instance

### Pre-Trained

Enter the following command in the ssh terminal:
```
python run_model_pretrained.py train_predictor_image_keras.npy test_predictor_image_keras.npy train_outcome_keras.npy test_outcome_keras.npy <learning_rate> <batch_size> <epochs>
```

You can play around with the last 3 parameters as needed, though I would recommend .001 for the learning rate, 10 for the batch size, and 200 for epochs. There is a learning rate decay function in the model and a early stopping function so it will end way before 200 epochs.

If you want to change the actual structure of the network, edit the *run_model_pretrained.py* file as necessary on your local machine and transfer it to the instance

## Output Files

Tensorboard logs are in the *logs* folder and can be accessed by the following command in terminal:

tensorboard --logdir=<logs_folder_directory>
```

Predictions for networks that ran to completion can be found in the *predictions* folder
## Confusion Matrix

To computer and visualize a confusion matrix, use confusion_matrix.py and give the filepaths of your y_test and y_prediction

```
python confusion_matrix.py <y_test_filepathy.npy> <y_prediction_filepath.npy>
```

