import pandas as pd
import ast
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import joblib

# Load your dataset (adjust filename if needed)
filename = 'prefixes_with_ts.csv'
df = pd.read_csv(filename)
df['Prefix'] = df['Prefix'].apply(ast.literal_eval)
df['PrefixTimestamps'] = df['PrefixTimestamps'].apply(ast.literal_eval)

prefixes = df['Prefix'].tolist()
labels = df['Status'].tolist()
prefix_lens = [len(trace) for trace in prefixes]  # Track each prefix length

# Encode events
all_events = [event for trace in prefixes for event in trace]
event_encoder = LabelEncoder()
event_encoder.fit(all_events)
encoded_prefixes = [event_encoder.transform(trace) for trace in prefixes]
max_len = max(len(p) for p in encoded_prefixes)
padded_prefixes = pad_sequences(encoded_prefixes, maxlen=max_len, padding='post', value=0)

# Encode labels
status_encoder = LabelEncoder()
y = status_encoder.fit_transform(labels)
y_cat = to_categorical(y)

# Train/test split
X_train, X_test, y_train, y_test, prefix_train, prefix_test = train_test_split(
    padded_prefixes, y, prefix_lens, test_size=0.2, random_state=42, stratify=y
)
y_cat_train = to_categorical(y_train)
y_cat_test = to_categorical(y_test)

# Build and train LSTM model
model = Sequential()
model.add(Embedding(input_dim=len(event_encoder.classes_), output_dim=32, input_length=max_len))
model.add(LSTM(64))
model.add(Dense(y_cat.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(
    X_train, y_cat_train,
    epochs=120, batch_size=64, validation_split=0.2,
    callbacks=[early_stopping], verbose=2
)

# Predict on test set
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)

# Prefix length-wise accuracy plot
results_df = pd.DataFrame({
    'Length': prefix_test,
    'True': y_test,
    'Pred': y_pred
})
accuracy_by_len = results_df.groupby('Length').apply(lambda g: (g['True'] == g['Pred']).mean()).reset_index(name='Accuracy')
accuracy_by_len.sort_values('Length', inplace=True)

plt.figure(figsize=(9,5))
plt.plot(accuracy_by_len['Length'], accuracy_by_len['Accuracy'], marker='o', color='navy')
plt.xlabel('Prefix Length')
plt.ylabel('Accuracy')
plt.title('Prefix Length-wise Accuracy for LSTM')
plt.ylim(0, 1.05)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Print sample prefix length accuracy for reference
print(accuracy_by_len.head(20))

# Save the trained LSTM model
model.save('lstm_prefix_model.h5')

# Save the event encoder and status encoder
joblib.dump(event_encoder, 'event_encoder.pkl')
joblib.dump(status_encoder, 'status_encoder.pkl')

# Also save max_len so backend knows padding length
joblib.dump(max_len, 'max_len.pkl')