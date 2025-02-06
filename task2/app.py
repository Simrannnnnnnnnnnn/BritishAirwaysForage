import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Display British Airways Logo
st.image("C:/Users/kaurs/Downloads/download.png", width=300)

# App Title
st.markdown("<h3 style='text-align: center; color: #1a2c5b;'>✈️ British Airways Booking Prediction</h3>", unsafe_allow_html=True)
st.title("🎯 Flight Booking Prediction App")

# Sidebar for user inputs
st.sidebar.header("🔍 User Input Features")

# Load dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("customer_booking.csv", encoding='latin1')
        return df.copy()
    except FileNotFoundError:
        st.error("⚠️ Error: Dataset file not found. Please check the file path.")
        return None

df = load_data()

if df is not None:
    if st.sidebar.checkbox("📜 Show raw data"):
        st.subheader("Raw Data")
        st.write(df.head())

    if st.sidebar.checkbox("📊 Show basic statistics"):
        st.subheader("Basic Statistics")
        st.write(df.describe())

    # Data Preprocessing
    def preprocess_data(df):
        df = df.copy()
        df.dropna(inplace=True)

        if 'route' in df.columns:
            df.drop(columns=['route'], inplace=True)

        if 'flight_day' in df.columns:
            day_map = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}
            df['flight_day'] = df['flight_day'].map(day_map).fillna(-1).astype(int)

        if 'flight_hour' in df.columns:
            df['time_of_day'] = pd.cut(df['flight_hour'], bins=[0, 6, 12, 18, 24], 
                                       labels=['Night', 'Morning', 'Afternoon', 'Evening'], right=False)

        categorical_cols = ['sales_channel', 'trip_type', 'time_of_day', 'booking_origin', 'origin', 'destination']
        existing_categorical_cols = [col for col in categorical_cols if col in df.columns]
        df = pd.get_dummies(df, columns=existing_categorical_cols, drop_first=True, dtype=int)

        return df

    df = preprocess_data(df)

    if 'booking_complete' in df.columns:
        X = df.drop(columns=['booking_complete'])
        y = df['booking_complete']

        st.write("Class Balance:", y.value_counts())

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        num_features = X_train.select_dtypes(include=['int64', 'float64']).columns
        X_train[num_features] = scaler.fit_transform(X_train[num_features])
        X_test[num_features] = scaler.transform(X_test[num_features])

        # Model Training
        @st.cache_resource
        def train_model(X_train, y_train):
            param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20], 'min_samples_split': [2, 5]}
            model = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'), 
                                 param_grid, cv=5, scoring='accuracy')
            model.fit(X_train, y_train)
            return model

        if st.sidebar.checkbox("🎯 Train Model"):
            model = train_model(X_train, y_train)
            st.success("✅ Model Trained Successfully!")

            st.write("Best Parameters:", model.best_params_)
            y_pred = model.best_estimator_.predict(X_test)

            st.subheader("📊 Evaluation Metrics")
            st.write(f"🔹 **Accuracy:** {model.best_estimator_.score(X_test, y_test):.4f}")
            st.write(f"🔹 **AUC-ROC Score:** {roc_auc_score(y_test, y_pred):.4f}")
            st.text("📜 Classification Report:")
            st.text(classification_report(y_test, y_pred))

            st.subheader("📉 Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)

            st.subheader("📈 Feature Importance")
            feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.best_estimator_.feature_importances_})
            feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
            fig, ax = plt.subplots()
            sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10), ax=ax)
            ax.set_title('Top 10 Feature Importances')
            st.pyplot(fig)

        # User Prediction Input
        st.sidebar.subheader("🔮 Make a Prediction")

        def user_input_features():
            user_data = {
                'num_passengers': st.sidebar.slider("🧑‍✈️ Number of Passengers", 1, 10, 1),
                'purchase_lead': st.sidebar.slider("📅 Purchase Lead (days)", 0, 365, 30),
                'length_of_stay': st.sidebar.slider("🏨 Length of Stay (days)", 0, 30, 5),
                'flight_hour': st.sidebar.slider("⏰ Flight Hour", 0, 23, 12),
                'flight_duration': st.sidebar.slider("🕒 Flight Duration (hours)", 1, 24, 2),
                'wants_extra_baggage': st.sidebar.selectbox("🛄 Wants Extra Baggage", [0, 1]),
                'wants_preferred_seat': st.sidebar.selectbox("💺 Wants Preferred Seat", [0, 1]),
                'wants_in_flight_meals': st.sidebar.selectbox("🍽️ Wants In-Flight Meals", [0, 1]),
            }
            return pd.DataFrame(user_data, index=[0])

        user_input = user_input_features()

        # Aligning user input
        missing_cols = [col for col in X_train.columns if col not in user_input.columns]
        missing_df = pd.DataFrame(0, index=user_input.index, columns=missing_cols)
        user_input = pd.concat([user_input, missing_df], axis=1)
        user_input = user_input[X_train.columns]
        user_input[num_features] = scaler.transform(user_input[num_features])

        if st.sidebar.button("🔎 Predict"):
            prediction_prob = model.best_estimator_.predict_proba(user_input)
            prediction = (prediction_prob[:, 1] > 0.4).astype(int)  # Adjusted threshold

            st.subheader("🎯 Prediction Result")
            st.write(f"🔹 **Probability of booking completion:** {prediction_prob[0][1]:.4f}")

            if prediction[0] == 1:
                st.success("✅ Booking will likely be completed!")
            else:
                st.error("❌ Booking is unlikely to be completed!")

            st.write("User Input Features:", user_input)
