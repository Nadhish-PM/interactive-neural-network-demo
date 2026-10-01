import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(
    page_title="Neural Network Playground",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Neural Network Playground")
st.markdown(
    "### Interactive learning demo — see how a neural network learns a classification problem."
)

st.sidebar.header("⚙️ Model Controls")

dataset_name = st.sidebar.selectbox(
    "Choose Dataset",
    ["Moons", "Circles", "Classification"]
)

hidden_layers = st.sidebar.text_input(
    "Hidden Layers",
    value="8,8",
    help="Example: 8,8 means two hidden layers with 8 neurons each."
)

activation = st.sidebar.selectbox(
    "Activation Function",
    ["relu", "tanh", "logistic"]
)

learning_rate = st.sidebar.slider(
    "Learning Rate",
    min_value=0.0005,
    max_value=0.1,
    value=0.01,
    step=0.0005,
    format="%.4f"
)

epochs = st.sidebar.slider(
    "Training Iterations",
    min_value=100,
    max_value=2000,
    value=500,
    step=100
)

noise = st.sidebar.slider(
    "Dataset Noise",
    min_value=0.0,
    max_value=0.5,
    value=0.15,
    step=0.05
)

test_size = st.sidebar.slider(
    "Test Size",
    min_value=0.1,
    max_value=0.4,
    value=0.2,
    step=0.05
)

random_state = st.sidebar.number_input(
    "Random Seed",
    min_value=0,
    max_value=9999,
    value=42,
    step=1
)

def create_dataset(name, noise_value, seed):
    if name == "Moons":
        X, y = make_moons(
            n_samples=500,
            noise=noise_value,
            random_state=seed
        )
    elif name == "Circles":
        X, y = make_circles(
            n_samples=500,
            noise=noise_value,
            factor=0.5,
            random_state=seed
        )
    else:
        X, y = make_classification(
            n_samples=500,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_clusters_per_class=1,
            class_sep=1.2,
            flip_y=noise_value,
            random_state=seed
        )
    return X, y

def parse_layers(value):
    try:
        layers = tuple(int(x.strip()) for x in value.split(",") if x.strip())
        if not layers or any(x < 1 or x > 100 for x in layers):
            raise ValueError
        return layers
    except ValueError:
        return (8, 8)

X, y = create_dataset(dataset_name, noise, int(random_state))
layers = parse_layers(hidden_layers)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size,
    random_state=int(random_state),
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = MLPClassifier(
    hidden_layer_sizes=layers,
    activation=activation,
    solver="adam",
    learning_rate_init=learning_rate,
    max_iter=epochs,
    random_state=int(random_state),
    early_stopping=False
)

with st.spinner("Training neural network..."):
    model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dataset",
    "🧠 Network",
    "📈 Training",
    "🎯 Evaluation"
])

with tab1:
    st.subheader("Dataset Visualization")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap="viridis",
        edgecolors="black",
        alpha=0.75
    )
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title(f"{dataset_name} Dataset")
    ax.grid(alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        f"This dataset contains {len(X)} samples. "
        f"The neural network receives two numerical features and predicts one of two classes."
    )

with tab2:
    st.subheader("Neural Network Architecture")

    architecture = ["Input (2 features)"] + [
        f"Hidden Layer {i}: {n} neurons"
        for i, n in enumerate(layers, start=1)
    ] + ["Output (2 classes)"]

    cols = st.columns(len(architecture))
    for i, (col, label) in enumerate(zip(cols, architecture)):
        with col:
            st.metric(f"Layer {i}", label)

    st.markdown("---")
    st.write("**Selected configuration:**")
    st.write(f"- Activation: `{activation}`")
    st.write(f"- Learning rate: `{learning_rate}`")
    st.write(f"- Training iterations: `{epochs}`")
    st.write(f"- Hidden layers: `{layers}`")

    st.caption(
        "The hidden layers transform the input features through learned weights. "
        "During training, the optimizer adjusts these weights to reduce prediction loss."
    )

with tab3:
    st.subheader("Training Loss")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(model.loss_curve_, linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Neural Network Learning Curve")
    ax.grid(alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)

    st.success(
        f"Training completed in {model.n_iter_} iterations. "
        f"Final loss: {model.loss_curve_[-1]:.4f}"
    )

with tab4:
    st.subheader("Model Evaluation")

    c1, c2, c3 = st.columns(3)
    c1.metric("Test Accuracy", f"{accuracy * 100:.2f}%")
    c2.metric("Training Samples", len(X_train))
    c3.metric("Test Samples", len(X_test))

    st.markdown("#### Confusion Matrix")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    ax.set_title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    st.pyplot(fig)
    plt.close(fig)

    st.markdown("#### Classification Report")
    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(3), use_container_width=True)

st.markdown("---")
st.subheader("🎯 Try a New Data Point")

c1, c2 = st.columns(2)
with c1:
    feature_1 = st.number_input(
        "Feature 1",
        value=float(X[:, 0].mean()),
        format="%.3f"
    )
with c2:
    feature_2 = st.number_input(
        "Feature 2",
        value=float(X[:, 1].mean()),
        format="%.3f"
    )

new_point = np.array([[feature_1, feature_2]])
prediction = model.predict(scaler.transform(new_point))[0]
probability = model.predict_proba(scaler.transform(new_point))[0]

st.success(f"Predicted Class: **{prediction}**")

prob_df = pd.DataFrame({
    "Class": [f"Class {i}" for i in range(len(probability))],
    "Probability": probability
})
st.bar_chart(prob_df.set_index("Class"))

st.caption(
    "Change the controls in the sidebar and observe how the architecture, "
    "loss curve and evaluation results change. This demonstrates the effect "
    "of neural-network hyperparameters interactively."
)

st.markdown("---")
st.caption("Activity 2 — Interactive Neural Network Learning Demo")
