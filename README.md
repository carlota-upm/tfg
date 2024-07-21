# 🧠 Neural-Network Utility Functions in Game Theory

Welcome to the repository for our framework of neural network utility functions, designed to analyze and predict outcomes based on behavioral data. To visualise the utility of our framework, we focus on two primary examples: **Allais** and **Ellsberg** Paradoxes.

## 🚀 Getting Started

### Dependencies

Before you begin, make sure you have the following libraries installed:

- `pandas`
- `numpy`
- `tensorflow`
- `scikit-learn`
- `matplotlib`

### 📁 File Structure

Here's a quick overview of the key components in our project:

- **Model Training**: `nn_model(MODEL_TYPE, N, TRAINING_COLUMNS, TEST_COLUMN, EPOCHS=200, BATCH_SIZE=10)`
- **Results Generation**: `result(MSE_THRESHOLD, MODEL_TYPE, N, NN_NUMBER, TRAINING_COLUMNS, TEST_COLUMN, EPOCHS, BATCH_SIZE)`
- **Plotting Functions**: `plot_inconsistencies`, `plot_risk`, `plot_selection`

## 🧩 Code Explanation

### 🏗️ Model Training

The `nn_model` function trains a neural network based on the specified `MODEL_TYPE`:

```python
def nn_model(MODEL_TYPE, N, TRAINING_COLUMNS, TEST_COLUMN, EPOCHS=200, BATCH_SIZE=10):
    if MODEL_TYPE == 'allais':
        df = allais_data(N)
    else:
        print('Invalid model type')
        return

    # Shuffle the dataframe
    df = shuffle(df)

    # Split the data into training and testing sets
    X = df[TRAINING_COLUMNS].values
    y = df[TEST_COLUMN].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create the neural network
    model = Sequential()
    model.add(Dense(1, input_dim=4, activation='relu'))  # First hidden layer
    model.add(Dense(1, activation='linear'))  # Output layer
    
    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    
    # Train the model
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val), verbose=0)
    
    return model, history
```

The `MODEL_TYPE` parameter specifies the dataset to use, while the `TRAINING_COLUMNS` and `TEST_COLUMN` parameters specify the columns to use for training and testing, respectively. There are also optional parameters for the number of `EPOCHS` and `BATCH_SIZE` to use.

This function can be altered to train models for different datasets and neural network architectures.

### 📊 Results Generation

The `result` function generates results based on various parameters. It outputs a CSV file with the results.


### 📈 Plotting Functions

We provide functions to visualize the different results from where we can draw conclusions.  We decided to plot the inconsistencies, risk, and selection plots as we believe they are the most relevant to the analysis of the Allais and Ellsberg Paradoxes.

## 📷 Visualization Examples

Here are some examples of the plots you can generate:

### Inconsistencies Plot

<u> Inconsistencies Plot of the Allais Paradox varying the Dataset Size </u>

| ![30 epochs](/epochs/epochs_30/images/allais/inconsistencies_plot_dataset_size_allais.png) | ![50 epochs](/epochs/epochs_50/images/allais/inconsistencies_plot_dataset_size_allais.png) | ![100 epochs](/epochs/epochs_100/images/allais/inconsistencies_plot_dataset_size_allais.png) |
|:---:|:---:|:---:|
| 30 epochs | 50 epochs | 100 epochs |

<u> Inconsistencies Plot of the Allais Paradox varying the Model Complexity </u>

| ![30 epochs](/epochs/epochs_30/images/allais/inconsistencies_plot_model_complexity_allais.png) | ![50 epochs](/epochs/epochs_50/images/allais/inconsistencies_plot_model_complexity_allais.png) | ![100 epochs](/epochs/epochs_100/images/allais/inconsistencies_plot_model_complexity_allais.png) |
|:---:|:---:|:---:|
| 30 epochs | 50 epochs | 100 epochs |

### Risk Aversion Plot

<u> Risk Aversion Plot of the Allais Paradox varying the Dataset Size </u>
| ![30 epochs](/epochs/epochs_30/images/allais/risk_aversion_plot_dataset_size_allais.png) | ![50 epochs](/epochs/epochs_50/images/allais/risk_aversion_plot_dataset_size_allais.png) | ![100 epochs](/epochs/epochs_100/images/allais/risk_aversion_plot_dataset_size_allais.png) |
|:---:|:---:|:---:|
| 30 epochs | 50 epochs | 100 epochs |

<u> Risk Aversion Plot of the Allais Paradox varying the Model Complexity </u>
| ![30 epochs](/epochs_30/images/allais/risk_aversion_plot_model_complexity_allais.png) | ![50 epochs](/epochs_50/images/allais/risk_aversion_plot_model_complexity_allais.png) | ![100 epochs](/epochs_100/images/allais/risk_aversion_plot_model_complexity_allais.png) |
|:---:|:---:|:---:|
| 30 epochs | 50 epochs | 100 epochs |

### Selection Plot

<u> Selection Plot for Experiment 1 varying the Dataset Size </u>
| ![30 epochs](/epochs/epochs_30/images/allais/experiment1_selections_dataset_size_allais.png) | ![50 epochs](/epochs/epochs_50/images/allais/experiment1_selections_dataset_size_allais.png) | ![100 epochs](/epochs/epochs_100/images/allais/experiment1_selections_dataset_size_allais.png) |
|:---:|:---:|:---:|
| 30 epochs | 50 epochs | 100 epochs |

<u> Selection Plot for Experiment 2 varying the Dataset Size </u>
| ![30 epochs](/epochs/epochs_30/images/allais/experiment2_selections_dataset_size_allais.png) | ![50 epochs](/epochs/epochs_50/images/allais/experiment2_selections_dataset_size_allais.png) | ![100 epochs](/epochs/epochs_100/images/allais/experiment2_selections_dataset_size_allais.png) |
|:---:|:---:|:---:|
| 30 epochs | 50 epochs | 100 epochs |

<u> Selection Plot for Experiment 1 varying the Model Complexity </u>
| ![30 epochs](/epochs/epochs_30/images/allais/experiment1_selections_model_complexity_allais.png) | ![50 epochs](/epochs/epochs_50/images/allais/experiment1_selections_model_complexity_allais.png) | ![100 epochs](/epochs/epochs_100/images/allais/experiment1_selections_model_complexity_allais.png) |
|:---:|:---:|:---:|
| 30 epochs | 50 epochs | 100 epochs |

<u> Selection Plot for Experiment 2 varying the Model Complexity </u>
| ![30 epochs](/epochs/epochs_30/images/allais/experiment2_selections_model_complexity_allais.png) | ![50 epochs](/epochs/epochs_50/images/allais/experiment2_selections_model_complexity_allais.png) | ![100 epochs](/epochs/epochs_100/images/allais/experiment2_selections_model_complexity_allais.png) |
|:---:|:---:|:---:|
| 30 epochs | 50 epochs | 100 epochs |

## 🤝 Contributing

We welcome contributions! Please feel free to submit a pull request or open an issue.

## 📄 License

We believe that all knowledge generated in a public university should be pub-
lic and free. That is why this work is published with the license Creative Com-
mons Attribution - No Derivative Works - Noncommercial - Share Alike, in
such a way that it is guaranteed that the work is shared with the same free-
doms and thus, everyone can use this work as they want.

## 📞 Contact

Feel free to reach out if you have any questions or suggestions. Happy coding! 💻✨
