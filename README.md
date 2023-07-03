
<body>
  <h1>Sentiment Analysis on Turkish Tweets</h1>

  <h2>Dependencies</h2>
  <pre>
    pandas
    re
    numpy
    matplotlib
    seaborn
    tensorflow
    scikit-learn
  </pre>

  <h2>Data</h2>
  <p> [Kaggle/Turkish Tweets Dataset](https://www.kaggle.com/datasets/anil1055/turkish-tweet-dataset)  The dataset contains tweets in Turkish along with their corresponding labels.</p>


  <h2>Data Preprocessing</h2>
  <p>The code performs several preprocessing steps on the text data before training the models.</p>
  <ol>
    <li>Data Exploration: The code displays a count plot of the sentiment labels to visualize the distribution of sentiments in the dataset.</li>
    <li>Label Mapping: The sentiment labels are mapped to numerical values for model training.</li>
    <li>Text Cleaning: The code defines a function to clean the text by removing unwanted patterns and special characters.</li>
    <li>Lowercasing: The text is converted to lowercase to ensure consistent tokenization.</li>
    <li>Stopword Removal: Turkish stopwords are removed from the text using the nltk library.</li>
  </ol>

  <h2>Vectorization</h2>
  <p>The code performs vectorization on the preprocessed text data using TF-IDF vectorization and count vectorization.</p>

  <h2>Machine Learning Models</h2>
  <p>The code trains and evaluates two machine learning models: Naive Bayes and Support Vector Machine (SVM).</p>
  <h3>Naive Bayes</h3>
  <p>The code uses the Bernoulli Naive Bayes classifier from scikit-learn to train the Naive Bayes model.</p>
  <h3>Support Vector Machine (SVM)</h3>
  <p>The code uses the SVM classifier from scikit-learn to train the SVM model.</p>

  <h2>Deep Learning Models</h2>
  <p>The code trains and evaluates two deep learning models: CNN (Convolutional Neural Network) and LSTM (Long Short-Term Memory).</p>
  <h3>CNN</h3>
  <p>The code defines a CNN model using the Keras API.</p>
  <h3>LSTM</h3>
  <p>The code defines an LSTM model using the Keras API.</p>

  <h2>Evaluation</h2>
  <p>The code evaluates the trained models on the test data and computes accuracy scores for each model. It also visualizes the training and validation loss and accuracy for the deep learning models.</p>

  <h2>Testing</h2>
  <p>The code includes a function to test the trained models on new tweets.</p>

  <h2>Example Usage</h2>
  <p>To test the trained models on a new tweet, you can call the function.</p>
</body>
</html>
