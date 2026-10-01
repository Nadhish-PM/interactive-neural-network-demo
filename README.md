# 🧠 Interactive Neural Network Learning Demo

## Activity — Artificial Neural Network

A web-based interactive neural network learning demo built with **Streamlit**.

### Features

- Three datasets: Moons, Circles and Classification
- Adjustable hidden-layer architecture
- ReLU, Tanh and Logistic activation functions
- Adjustable learning rate
- Adjustable training iterations
- Adjustable dataset noise
- Train/test split control
- Dataset visualization
- Neural-network architecture display
- Training loss curve
- Test accuracy
- Confusion matrix
- Classification report
- Interactive prediction for a new data point

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The application will open in your browser.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py` and `requirements.txt`.
3. Open Streamlit Community Cloud.
4. Select your GitHub repository.
5. Select `app.py` as the main file.
6. Deploy.

## Suggested demonstration flow

1. Explain what a neural network does.
2. Select the Moons dataset.
3. Show the initial dataset.
4. Explain input → hidden layers → output.
5. Change hidden layers from `4` to `8,8`.
6. Change the activation function.
7. Change learning rate or noise.
8. Show the training-loss curve.
9. Show test accuracy and confusion matrix.
10. Enter a new point and demonstrate prediction.

## Academic objective

The demo allows learners to observe how neural-network architecture and hyperparameters influence training and classification performance.
